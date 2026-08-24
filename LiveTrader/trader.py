#!/usr/bin/env python3
"""CoreEquity live-trader CLI.

Default is dry-run: compute the locked CoreEquity book on the live IBKR daily panel
(Yahoo only if TWS/historical fails), optionally size against the IB account, never
send an order unless paper-submit / live-submit. The pack itself is not modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from book import (  # noqa: E402
    CoolState,
    SLEEVE_A,
    active_sleeve_symbols,
    assign_order_types,
    clamp_buys_to_cash,
    flatten_intents,
    journal_key,
    merge_marks,
    orders_to_targets,
    park_sleeve_a,
    round_to_whole_shares,
    session_rebalance_flags,
    spendable_cash,
    weight_drift,
)
from config import LiveConfig, load_config  # noqa: E402
from data import cache_dir, last_panel_source, load_live_panel, panel_to_px  # noqa: E402
from ce_strategy import (  # noqa: E402
    BOOK_SYMBOLS,
    P,
    STRATEGY_ID,
    apply_sleeve_a_to_targets,
    latest_targets,
    portfolio_weights,
    sleeve_a_live_state,
)
from ibkr_client import (  # noqa: E402
    AccountSnapshot,
    BrokerError,
    IBKRBroker,
)
from preflight import Check, blocking_failures, evaluate  # noqa: E402


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _state_path() -> Path:
    return cache_dir() / "state.json"


def _journal_path() -> Path:
    return cache_dir() / "order_intents.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _force_refresh(args: argparse.Namespace) -> bool:
    """Yahoo panel always refreshes unless --no-refresh-data."""
    return not bool(getattr(args, "no_refresh_data", False))


def _journal_submitted(account: str, asof: str) -> bool:
    path = _journal_path()
    if not path.is_file():
        return False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("submitted"):
                continue
            if rec.get("fills_ok") is False:
                continue
            if str(rec.get("asof")) == str(asof) and str(rec.get("account") or "") == str(account or ""):
                return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def load_cool() -> CoolState:
    st = _read_json(_state_path())
    ltd = st.get("last_trade_date")
    return CoolState(
        peak_equity=float(st.get("peak_equity") or 0.0),
        flat_a=bool(st.get("flat_a") or False),
        cool_remaining=int(st.get("cool_remaining") or 0),
        last_trade_date=str(ltd) if ltd else None,
    )


def save_cool(cool: CoolState, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "peak_equity": cool.peak_equity,
        "flat_a": cool.flat_a,
        "cool_remaining": cool.cool_remaining,
        "last_trade_date": cool.last_trade_date,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    _write_json(_state_path(), payload)


def _connect(cfg: LiveConfig):
    broker = IBKRBroker(cfg)
    broker.connect()
    return broker


def _connect_optional(cfg: LiveConfig, *, required: bool = False):
    """Connect for IBKR data/orders. Yahoo fallback when not required."""
    try:
        return _connect(cfg)
    except BrokerError:
        if required:
            raise
        if str(cfg.data_source) == "ibkr" and not cfg.yahoo_fallback:
            raise
        return None


def _offline_snapshot(aum: float) -> AccountSnapshot:
    return AccountSnapshot(
        account="OFFLINE",
        net_liquidation=float(aum),
        cash=float(aum),
        buying_power=float(aum),
        positions={},
        port=0,
        server_name="offline",
    )


def build_plan(
    *,
    cfg: LiveConfig,
    snapshot: AccountSnapshot | None,
    force_refresh: bool = True,
    as_of: date | None = None,
    broker: Any | None = None,
) -> dict[str, Any]:
    dates, closes, ohlc = load_live_panel(
        force_refresh=force_refresh, broker=broker, cfg=cfg
    )
    panel_dates, px, panel_ohlc, panel_marks = panel_to_px(dates, closes, ohlc, as_of=as_of)
    data_source = last_panel_source()
    hot = str(getattr(P, "eq_sym", "QQQ")).upper()
    if hot not in panel_ohlc or "QQQ" not in panel_ohlc:
        raise RuntimeError(f"live panel missing {hot}/QQQ OHLC")
    targets_raw = latest_targets(panel_dates, px, panel_ohlc[hot], panel_ohlc["QQQ"], P)
    sleeve = sleeve_a_live_state(panel_dates, px, panel_ohlc[hot], P)
    targets_raw = apply_sleeve_a_to_targets(targets_raw, sleeve, P)
    weights = portfolio_weights(targets_raw, P)
    if sleeve.flat:
        weights = park_sleeve_a(weights, str(targets_raw.get("dual_asset") or "GLD"))
    signal_asof = panel_dates[-1]
    calendar_today = as_of or date.today()
    wk, me = session_rebalance_flags(
        panel_dates, len(panel_dates) - 1, calendar_today=calendar_today
    )
    trade_asof = signal_asof
    if calendar_today > signal_asof and calendar_today.weekday() < 5:
        lag = (calendar_today - signal_asof).days
        if 0 < lag <= 3:
            trade_asof = calendar_today

    ib_last = dict(snapshot.last_prices) if snapshot is not None else {}
    marks = merge_marks(panel_marks, ib_last)
    if ib_last:
        mark_source = f"ib_last+{data_source}"
    else:
        mark_source = data_source

    aum = float(cfg.aum_override or 0.0)
    positions: dict[str, float] = {}
    if snapshot is not None:
        if aum <= 0:
            aum = float(snapshot.net_liquidation or snapshot.cash or 0.0)
        positions = {str(k).upper(): float(v) for k, v in snapshot.positions.items()}
    if aum <= 0:
        aum = 1000.0

    stored = load_cool()
    cool = CoolState(
        peak_equity=sleeve.peak,
        flat_a=sleeve.flat,
        cool_remaining=sleeve.cool_remaining,
        last_trade_date=stored.last_trade_date,
    )
    es_park = bool(
        sleeve.flat
        and any(abs(float(positions.get(s, 0.0) or 0.0)) > 1e-8 for s in SLEEVE_A)
    )

    book_empty = not any(
        abs(float(v)) > 1e-8 for k, v in positions.items() if str(k).upper() in set(BOOK_SYMBOLS)
    )
    seed = bool(cfg.seed_if_flat and book_empty)
    already = str(cool.last_trade_date or "") == str(trade_asof)
    account = str((snapshot.account if snapshot else "") or cfg.account or "")
    if (not already) and account and _journal_submitted(account, str(trade_asof)):
        already = True
    due = bool(wk or me or seed or es_park)
    skip_reason = ""
    rebal = False
    if already:
        skip_reason = "already_traded"
    elif not due:
        skip_reason = "not_due"
    else:
        rebal = True

    allow = None
    if cfg.sleeve_split:
        allow = active_sleeve_symbols(week_end=wk, month_end=me, seed=seed, es_park=es_park)

    orders = []
    if rebal:
        orders = orders_to_targets(
            aum,
            positions,
            marks,
            weights,
            min_notional=cfg.min_notional,
            allow_symbols=allow,
        )
        orders = assign_order_types(
            orders,
            whole_share_type=cfg.whole_share_order_type,
            fractional_type=cfg.fractional_order_type,
        )
        if not cfg.allow_fractional:
            orders = round_to_whole_shares(orders, min_notional=cfg.min_notional)
            orders = assign_order_types(
                orders,
                whole_share_type=cfg.whole_share_order_type,
                fractional_type=cfg.fractional_order_type,
            )
        if cfg.cap_buys_to_cash and snapshot is not None:
            orders = clamp_buys_to_cash(
                orders,
                spendable_cash(snapshot.cash, snapshot.buying_power),
                min_notional=cfg.min_notional,
            )

    return {
        "strategy_id": STRATEGY_ID,
        "asof": str(trade_asof),
        "signal_asof": str(signal_asof),
        "data_source": data_source,
        "mark_source": mark_source,
        "account": account,
        "aum": aum,
        "week_end": wk,
        "month_end": me,
        "seed_if_flat": seed,
        "es_park": es_park,
        "sleeve_symbols": sorted(allow) if allow is not None else sorted(BOOK_SYMBOLS),
        "rebalance": rebal,
        "skip_reason": skip_reason,
        "already_traded": already,
        "flat_a": sleeve.flat,
        "cool_remaining": sleeve.cool_remaining,
        "peak_equity": sleeve.peak,
        "sleeve_a_equity": sleeve.equity,
        "target_weights": weights,
        "targets": {k: v for k, v in targets_raw.items() if k != "params"},
        "positions": positions,
        "marks": marks,
        "orders": [o.as_dict() for o in orders],
        "n_orders": len(orders),
        "journal_keys": [journal_key(account, str(trade_asof), o.symbol) for o in orders],
        "note": (
            "weekly QQQ close + month-end GLD/TLT dual; residual BIL; "
            "sleeve-A ES parks to BIL immediately; "
            "no 2x/3x ETFs; fractional legs use MKT, whole shares use MOC"
        ),
        "_orders": orders,
        "_cool": cool,
        "_panel": (panel_dates, px, panel_ohlc, marks, weights),
        "_snapshot": snapshot,
    }


def _public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in plan.items() if not str(k).startswith("_")}


def _eval_for(cfg: LiveConfig, plan: dict[str, Any], snapshot: AccountSnapshot | None, **kwargs):
    panel_dates, px, panel_ohlc, marks, weights = plan["_panel"]
    return evaluate(
        cfg=cfg,
        dates=panel_dates,
        px=px,
        ohlc=panel_ohlc,
        marks=marks,
        weights=weights,
        snapshot=snapshot,
        orders=plan["_orders"],
        already_traded=bool(plan.get("already_traded")),
        skip_reason=str(plan.get("skip_reason") or ""),
        **kwargs,
    )


def cmd_verify_data(args: argparse.Namespace) -> int:
    cfg = load_config()
    broker = None
    try:
        broker = _connect_optional(cfg)
        dates, closes, ohlc = load_live_panel(
            force_refresh=_force_refresh(args), broker=broker, cfg=cfg
        )
        panel_dates, px, panel_ohlc, marks = panel_to_px(dates, closes, ohlc)
        hot = str(getattr(P, "eq_sym", "QQQ")).upper()
        live = latest_targets(panel_dates, px, panel_ohlc[hot], panel_ohlc["QQQ"], P)
        live_w = portfolio_weights(live, P)
        source = last_panel_source()
        frozen_asof = None
        frozen_w = None
        try:
            from rlbot.pack_core_equity import latest_portfolio_weights, paper_plan

            frozen = paper_plan(aum=1000.0)
            frozen_asof = frozen.get("asof")
            frozen_source = frozen.get("data_source")
            frozen_w = latest_portfolio_weights(aum=1000.0)
            live_via_pack = paper_plan(
                aum=1000.0,
                dates=panel_dates,
                closes=px,
                ohlc=panel_ohlc,
            )
            pack_live_w = latest_portfolio_weights(
                aum=1000.0, dates=panel_dates, closes=px, ohlc=panel_ohlc
            )
        except Exception as exc:  # noqa: BLE001
            frozen_source = f"unavailable: {exc}"
            live_via_pack = {}
            pack_live_w = {}
        qqq_r = float("nan")
        if "QQQ" in px and len(px["QQQ"]) >= 2:
            qqq_r = float(px["QQQ"][-1] / px["QQQ"][-2] - 1.0)
        payload = {
            "today": str(date.today()),
            "strategy_id": STRATEGY_ID,
            "live_asof": live.get("asof"),
            "live_source": source,
            "live_weights": live_w,
            "live_last_day_QQQ": qqq_r,
            "frozen_pack_asof": frozen_asof,
            "frozen_pack_source": frozen_source,
            "frozen_pack_weights": frozen_w,
            "pack_on_live_panel_asof": live_via_pack.get("asof"),
            "pack_on_live_panel_source": live_via_pack.get("data_source"),
            "live_matches_pack_on_live": (
                bool(pack_live_w)
                and all(abs(live_w.get(k, 0.0) - float(v)) < 1e-9 for k, v in pack_live_w.items())
                and set(live_w) == set(pack_live_w)
            ),
            "marks": {k: marks.get(k) for k in ("SPY", "QQQ", "GLD", "TLT", "BIL")},
        }
        print(json.dumps(payload, indent=2, default=str))
        if payload["live_asof"] == payload["frozen_pack_asof"]:
            print(
                "[verify] WARNING: live asof still equals frozen pack asof — panel may be stale",
                file=sys.stderr,
            )
            return 1
        if payload["live_matches_pack_on_live"] is False:
            print(
                "[verify] WARNING: LiveTrader weights diverged from pack on the live panel",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        if broker is not None:
            broker.disconnect()


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = load_config()
    snap = None
    broker = None
    try:
        broker = _connect_optional(cfg, required=bool(args.connect) and not cfg.yahoo_fallback)
        if broker is not None:
            snap = broker.snapshot()
        elif cfg.aum_override:
            snap = _offline_snapshot(float(cfg.aum_override))
        plan = build_plan(
            cfg=cfg,
            snapshot=snap,
            broker=broker,
            force_refresh=_force_refresh(args),
            as_of=_parse_date(args.as_of) if args.as_of else None,
        )
        print(json.dumps(_public_plan(plan), indent=2, default=str))
        return 0
    finally:
        if broker is not None:
            broker.disconnect()


def cmd_preflight(args: argparse.Namespace) -> int:
    cfg = load_config()
    snap = None
    broker = None
    connect = not bool(args.offline)
    plan: dict[str, Any] = {}
    checks: list[Check] = []
    try:
        if connect:
            broker = _connect(cfg)
            snap = broker.snapshot()
            qualified = broker.qualify_book()
        else:
            qualified = []
        plan = build_plan(
            cfg=cfg,
            snapshot=snap or _offline_snapshot(float(cfg.aum_override or 1000.0)),
            broker=broker,
            force_refresh=_force_refresh(args),
        )
        checks = _eval_for(cfg, plan, snap, want_connect=connect, want_submit=False)
        if connect:
            checks.append(
                Check(
                    "qualify_book",
                    set(qualified) >= set(BOOK_SYMBOLS),
                    "qualified=" + ",".join(qualified) if qualified else "none",
                )
            )
    except BrokerError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    finally:
        if broker is not None:
            broker.disconnect()
    failed = blocking_failures(checks)
    print(
        json.dumps(
            {
                "ok": not failed,
                "mode": cfg.mode,
                "asof": plan.get("asof"),
                "aum": plan.get("aum"),
                "rebalance": plan.get("rebalance"),
                "skip_reason": plan.get("skip_reason"),
                "n_orders": plan.get("n_orders"),
                "target_weights": plan.get("target_weights"),
                "snapshot": snap.as_dict() if snap else None,
                "checks": [c.as_dict() for c in checks],
            },
            indent=2,
            default=str,
        )
    )
    return 1 if failed else 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    del args
    cfg = load_config()
    broker = _connect(cfg)
    try:
        snap = broker.snapshot()
        print(json.dumps(snap.as_dict(), indent=2, default=str))
    finally:
        broker.disconnect()
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    cfg = load_config()
    snap = None
    broker = None
    try:
        if args.connect or cfg.data_source == "ibkr":
            broker = _connect_optional(cfg, required=bool(args.connect) and not cfg.yahoo_fallback)
            if broker is not None:
                snap = broker.snapshot()
        elif cfg.aum_override:
            snap = _offline_snapshot(float(cfg.aum_override))
        plan = build_plan(
            cfg=cfg,
            snapshot=snap,
            broker=broker,
            force_refresh=_force_refresh(args),
            as_of=_parse_date(args.as_of) if args.as_of else None,
        )
        checks = _eval_for(cfg, plan, snap, want_connect=bool(args.connect), want_submit=False)
        rec = {
            **_public_plan(plan),
            "submitted": False,
            "dry_run": True,
            "checks": [c.as_dict() for c in checks],
        }
        if not args.no_journal:
            _append_jsonl(
                _journal_path(),
                {
                    **{
                        k: rec[k]
                        for k in (
                            "asof",
                            "account",
                            "aum",
                            "rebalance",
                            "skip_reason",
                            "orders",
                            "target_weights",
                            "journal_keys",
                        )
                    },
                    "dry_run": True,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
            save_cool(plan["_cool"], extra={"last_signal_date": plan["asof"], "equity": plan["aum"]})
        print(json.dumps(rec, indent=2, default=str))
        failed = blocking_failures(checks)
        return 1 if failed else 0
    finally:
        if broker is not None:
            broker.disconnect()


def _submit(args: argparse.Namespace, *, expected_mode: str, arm_live: bool = False) -> int:
    cfg = load_config()
    if cfg.mode != expected_mode and not getattr(args, "force_mode", False):
        raise SystemExit(
            f"config execution.mode is {cfg.mode!r}; set it to {expected_mode!r} "
            f"or pass --force-mode"
        )
    broker = _connect(cfg)
    try:
        snap = broker.snapshot()
        plan = build_plan(
            cfg=cfg, snapshot=snap, broker=broker, force_refresh=_force_refresh(args)
        )
        if getattr(args, "if_due", False) and not plan["rebalance"]:
            print(
                json.dumps(
                    {
                        "submitted": False,
                        "reason": plan.get("skip_reason") or "not_due",
                        "plan": _public_plan(plan),
                    },
                    indent=2,
                    default=str,
                )
            )
            return 0
        checks = _eval_for(
            cfg,
            plan,
            snap,
            want_connect=True,
            want_submit=True,
            arm_live=arm_live,
            confirm_env=os.environ.get("LIVE_TRADER_CONFIRM", ""),
        )
        failed = blocking_failures(checks)
        if failed or not plan["_orders"]:
            print(
                json.dumps(
                    {
                        "submitted": False,
                        "reason": [c.as_dict() for c in failed] if failed else (
                            plan.get("skip_reason") or "no_orders"
                        ),
                        "plan": _public_plan(plan),
                        "checks": [c.as_dict() for c in checks],
                    },
                    indent=2,
                    default=str,
                )
            )
            return 1 if failed else 0
        if args.preview_only:
            print(json.dumps({"submitted": False, "preview": _public_plan(plan)}, indent=2, default=str))
            return 0
        what = broker.what_if(
            plan["_orders"], order_type=cfg.whole_share_order_type, tif=cfg.tif
        )
        if not what.get("ok", False):
            print(json.dumps({"submitted": False, "reason": "what_if_failed", "what_if": what}, indent=2, default=str))
            return 1
        fills = broker.submit(
            plan["_orders"],
            order_type=cfg.whole_share_order_type,
            tif=cfg.tif,
        )
        fills, fills_ok = broker.wait_for_fills(fills, timeout_s=cfg.fill_timeout_s)
        after = broker.snapshot()
        drift = weight_drift(
            float(after.net_liquidation or plan["aum"]),
            after.positions,
            merge_marks(plan["marks"], after.last_prices),
            plan["target_weights"],
        )
        rec = {
            **_public_plan(plan),
            "submitted": True,
            "fills": fills,
            "fills_ok": fills_ok,
            "what_if": what,
            "reconcile_drift": drift,
            "positions_after": after.positions,
            "mode": cfg.mode,
        }
        _append_jsonl(
            _journal_path(),
            {
                **{k: rec[k] for k in ("asof", "account", "aum", "orders", "fills", "journal_keys") if k in rec},
                "submitted": True,
                "fills_ok": fills_ok,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        cool = plan["_cool"]
        extra = {"equity": plan["aum"]}
        if fills_ok:
            cool.last_trade_date = str(plan["asof"])
            extra["last_trade_date"] = plan["asof"]
        save_cool(cool, extra=extra)
        print(json.dumps(rec, indent=2, default=str))
        return 0 if fills_ok else 1
    finally:
        broker.disconnect()


def cmd_paper_submit(args: argparse.Namespace) -> int:
    return _submit(args, expected_mode="paper")


def cmd_live_submit(args: argparse.Namespace) -> int:
    return _submit(args, expected_mode="live", arm_live=bool(getattr(args, "arm_live", False)))


def cmd_run_if_due(args: argparse.Namespace) -> int:
    cfg = load_config()
    args.if_due = True
    args.preview_only = bool(getattr(args, "preview_only", False))
    args.force_mode = True
    if cfg.mode == "dry_run":
        args.connect = True
        args.no_journal = False
        args.as_of = ""
        return cmd_dry_run(args)
    if cfg.mode == "paper":
        return _submit(args, expected_mode="paper")
    live_launchd = str(os.environ.get("LIVE_TRADER_LAUNCHD_LIVE") or "") == "1"
    return _submit(
        args,
        expected_mode="live",
        arm_live=live_launchd and bool(cfg.allow_live),
    )


def cmd_reconcile(args: argparse.Namespace) -> int:
    cfg = load_config()
    broker = _connect(cfg)
    try:
        snap = broker.snapshot()
        plan = build_plan(
            cfg=cfg, snapshot=snap, broker=broker, force_refresh=_force_refresh(args)
        )
        drift = weight_drift(
            float(snap.net_liquidation or plan["aum"]),
            snap.positions,
            plan["marks"],
            plan["target_weights"],
        )
        print(
            json.dumps(
                {
                    "asof": plan["asof"],
                    "account": snap.account,
                    "aum": plan["aum"],
                    "target_weights": plan["target_weights"],
                    "positions": snap.positions,
                    "drift": drift,
                    "ok": not drift,
                },
                indent=2,
                default=str,
            )
        )
        return 0 if not drift else 1
    finally:
        broker.disconnect()


def cmd_cancel_open(args: argparse.Namespace) -> int:
    del args
    cfg = load_config()
    broker = _connect(cfg)
    try:
        cancelled = broker.cancel_open_book_orders()
        print(json.dumps({"cancelled": cancelled}, indent=2))
        return 0
    finally:
        broker.disconnect()


def cmd_flatten(args: argparse.Namespace) -> int:
    cfg = load_config()
    broker = _connect(cfg)
    try:
        snap = broker.snapshot()
        dates, closes, ohlc = load_live_panel(
            force_refresh=_force_refresh(args), broker=broker, cfg=cfg
        )
        _d, _px, _oh, panel_marks = panel_to_px(dates, closes, ohlc)
        marks = merge_marks(panel_marks, snap.last_prices)
        orders = flatten_intents(
            snap.positions,
            marks,
            include_foreign=bool(args.include_foreign),
            min_notional=cfg.min_notional,
        )
        orders = assign_order_types(
            orders,
            whole_share_type="MKT",
            fractional_type="MKT",
        )
        payload = {
            "account": snap.account,
            "positions": snap.positions,
            "orders": [o.as_dict() for o in orders],
            "n_orders": len(orders),
        }
        if args.preview_only or not args.confirm_flatten:
            print(json.dumps({"submitted": False, "preview": payload}, indent=2, default=str))
            return 0
        if cfg.mode == "live":
            env_ok = os.environ.get("LIVE_TRADER_CONFIRM", "") == cfg.confirm_phrase
            if not (cfg.allow_live and args.arm_live and env_ok):
                print(json.dumps({"submitted": False, "reason": "live flatten not armed"}, indent=2))
                return 1
        if not orders:
            print(json.dumps({"submitted": False, "reason": "no_orders", **payload}, indent=2))
            return 0
        broker.cancel_open_book_orders()
        what = broker.what_if(orders, order_type="MKT", tif="DAY")
        if not what.get("ok", False):
            print(json.dumps({"submitted": False, "reason": "what_if_failed", "what_if": what}, indent=2, default=str))
            return 1
        fills = broker.submit(orders, order_type="MKT", tif="DAY")
        fills, fills_ok = broker.wait_for_fills(fills, timeout_s=cfg.fill_timeout_s)
        after = broker.snapshot()
        rec = {
            **payload,
            "submitted": True,
            "fills": fills,
            "fills_ok": fills_ok,
            "positions_after": after.positions,
        }
        _append_jsonl(
            _journal_path(),
            {
                "flatten": True,
                "account": snap.account,
                "fills": fills,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        print(json.dumps(rec, indent=2, default=str))
        return 0 if fills_ok else 1
    finally:
        broker.disconnect()


def _parse_date(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    return date.fromisoformat(s[:10])


def _add_refresh_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--refresh-data",
        action="store_true",
        help="Deprecated no-op; the live panel always refreshes",
    )
    p.add_argument(
        "--no-refresh-data",
        action="store_true",
        help="Use the on-disk IBKR/Yahoo cache instead of a fresh pull",
    )


def _add_submit_flags(p: argparse.ArgumentParser, *, live: bool = False) -> None:
    _add_refresh_flags(p)
    p.add_argument("--preview-only", action="store_true")
    p.add_argument("--force-mode", action="store_true")
    p.add_argument("--if-due", action="store_true", help="No-op unless week-end, month-end, or seed")
    if live:
        p.add_argument(
            "--arm-live",
            action="store_true",
            help="Required together with allow_live and LIVE_TRADER_CONFIRM=CORE",
        )


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ver = sub.add_parser("verify-data", help="Live IBKR (or Yahoo fallback) asof vs frozen pack bars.db")
    _add_refresh_flags(p_ver)

    p_plan = sub.add_parser("plan", help="Print target weights + would-be orders")
    p_plan.add_argument("--connect", action="store_true", help="Require TWS (default already tries IBKR data)")
    _add_refresh_flags(p_plan)
    p_plan.add_argument("--as-of", default="")

    p_pre = sub.add_parser("preflight", help="Data + IBKR account checks; no orders")
    p_pre.add_argument("--offline", action="store_true")
    _add_refresh_flags(p_pre)

    sub.add_parser("snapshot", help="Print IBKR account + positions")

    p_dry = sub.add_parser("dry-run", help="Compute orders, journal them, do not send")
    p_dry.add_argument("--connect", action="store_true")
    _add_refresh_flags(p_dry)
    p_dry.add_argument("--as-of", default="")
    p_dry.add_argument("--no-journal", action="store_true")

    p_paper = sub.add_parser("paper-submit", help="Send to IBKR paper (port 7497/4002)")
    _add_submit_flags(p_paper)

    p_live = sub.add_parser("live-submit", help="Send to live IBKR; refused unless armed")
    _add_submit_flags(p_live, live=True)

    p_due = sub.add_parser("run-if-due", help="Dry-run / paper / live only when a sleeve is due")
    _add_refresh_flags(p_due)
    p_due.add_argument("--preview-only", action="store_true")

    p_rec = sub.add_parser("reconcile", help="Compare IB positions to target weights")
    _add_refresh_flags(p_rec)

    sub.add_parser("cancel-open", help="Cancel working CoreEquity orders at IBKR")

    p_flat = sub.add_parser("flatten", help="Sell CoreEquity names (preview unless --confirm-flatten)")
    _add_refresh_flags(p_flat)
    p_flat.add_argument("--include-foreign", action="store_true")
    p_flat.add_argument("--preview-only", action="store_true")
    p_flat.add_argument("--confirm-flatten", action="store_true")
    p_flat.add_argument("--arm-live", action="store_true")

    args = parser.parse_args()
    cmds = {
        "verify-data": cmd_verify_data,
        "plan": cmd_plan,
        "preflight": cmd_preflight,
        "snapshot": cmd_snapshot,
        "dry-run": cmd_dry_run,
        "paper-submit": cmd_paper_submit,
        "live-submit": cmd_live_submit,
        "run-if-due": cmd_run_if_due,
        "reconcile": cmd_reconcile,
        "cancel-open": cmd_cancel_open,
        "flatten": cmd_flatten,
    }
    return int(cmds[args.cmd](args))


if __name__ == "__main__":
    raise SystemExit(main())
