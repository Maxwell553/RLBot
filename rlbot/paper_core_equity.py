"""Paper / forward loop for CoreEquity (locked pack, read-only).

Targets come from ``CoreEquity/`` via ``rlbot.pack_core_equity``. Yahoo daily
bars are used for signals + MTM / 5m live refresh. State under
``execution/paper_core_equity/``.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rlbot.forward_mark import (
    build_forward_mark_payload,
    set_active_forward_run,
)
from rlbot.pack_core_equity import (
    DEFAULT_INITIAL_CASH,
    PAPER_RUN_ID,
    STRATEGY_ID,
    apply_sleeve_a_to_targets,
    book_symbols,
    locked_params,
    panel_symbols,
    paper_plan as pack_paper_plan,
    sleeve_a_live_state,
    weights_from_targets,
)
from rlbot.prod_return_alpha import (
    fetch_daily_ohlc,
    session_rebalance_flags,
)
from rlbot.run_artifacts import PROJECT_ROOT

EXECUTION_DIR = PROJECT_ROOT / "execution"
PAPER_DIR = EXECUTION_DIR / "paper_core_equity"
STATE_PATH = PAPER_DIR / "state.json"
ORDERS_PATH = PAPER_DIR / "order_intents.jsonl"
_SLEEVE_B = frozenset({"GLD", "TLT", "BIL"})
_SLEEVE_A = frozenset(s for s in book_symbols() if s not in _SLEEVE_B)


def ledger_path(run_id: str = PAPER_RUN_ID) -> Path:
    return EXECUTION_DIR / f"shadow_ledger_{run_id}.jsonl"


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


def default_state(initial_cash: float = DEFAULT_INITIAL_CASH) -> dict[str, Any]:
    return {
        "strategy_id": STRATEGY_ID,
        "run_id": PAPER_RUN_ID,
        "equity": float(initial_cash),
        "cash": float(initial_cash),
        "peak_equity": float(initial_cash),
        "flat_a": False,
        "cool_remaining": 0,
        "positions": {},
        "target_weights": {"CASH": 1.0},
        "last_signal_date": None,
        "last_trade_date": None,
        "updated_at_utc": None,
    }


def et_today() -> date:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:  # noqa: BLE001
        return datetime.now().astimezone().date()


def resolve_paper_session_day(
    as_of: date | None,
    bar_day: date,
    *,
    today: date | None = None,
) -> date:
    """Calendar session for already-traded / week-end flags.

    Live ticks use ET today so a Friday morning seed is not blocked by a
    Thursday ``last_trade_date`` while Yahoo still tips Thursday's bar.
    Historical ``--as-of`` stays on that date.
    """
    del bar_day
    if as_of is not None:
        return as_of
    return today if today is not None else et_today()


def core_paper_already_traded(last_trade: str | None, session_day: date) -> bool:
    return str(last_trade or "") == str(session_day)


def paper_state_is_flat(state: dict[str, Any] | None) -> bool:
    pos = (state or {}).get("positions") if isinstance((state or {}).get("positions"), dict) else {}
    for val in pos.values():
        try:
            if abs(float(val)) > 1e-12:
                return False
        except (TypeError, ValueError):
            continue
    return True


def paper_book_needs_reopen(state: dict[str, Any] | None, *, session: str) -> bool:
    """Empty lots whose last trade is not this calendar session (cash-reset leftover)."""
    if not paper_state_is_flat(state):
        return False
    return str((state or {}).get("last_trade_date") or "") != str(session)


def load_state(initial_cash: float = DEFAULT_INITIAL_CASH) -> dict[str, Any]:
    st = default_state(initial_cash)
    st.update(_read_json(STATE_PATH))
    st.setdefault("positions", {})
    st.setdefault("target_weights", {"CASH": 1.0})
    st.setdefault("peak_equity", st.get("equity") or initial_cash)
    return st


def save_state(state: dict[str, Any]) -> None:
    trade = str(state.get("last_trade_date") or "").strip()
    today = datetime.now(timezone.utc).date().isoformat()
    if not state.get("updated_at_utc") or trade == today:
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(STATE_PATH, state)


def reset_paper_book(
    *,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    hold_until: date | None = None,
    flatten_mark: bool = True,
) -> dict[str, Any]:
    """Flat $100k cash book. ``hold_until`` (default: today ET) skips a same-day refill.

    Tomorrow's collector sees empty lots and a prior trade date, so it opens the
    book on the next session without touching GeneralEquity1.
    """
    cash = float(initial_cash)
    if hold_until is None:
        hold_until = datetime.now(timezone.utc).astimezone().date()
        try:
            from zoneinfo import ZoneInfo

            hold_until = datetime.now(ZoneInfo("America/New_York")).date()
        except Exception:  # noqa: BLE001
            pass
    state = default_state(cash)
    state["last_trade_date"] = str(hold_until)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_state(state)
    _append_jsonl(
        ledger_path(),
        {
            "run_id": PAPER_RUN_ID,
            "strategy_id": STRATEGY_ID,
            "target_weights": {"CASH": 1.0},
            "trade_date": str(hold_until),
            "recorded_at_utc": state["updated_at_utc"],
            "note": "Reset to 100k cash (flat paper book).",
        },
    )
    mark = flatten_core_equity_companion(initial_cash=cash) if flatten_mark else None
    return {
        "run_id": PAPER_RUN_ID,
        "equity": cash,
        "cash": cash,
        "positions": {},
        "hold_until": str(hold_until),
        "state_path": str(STATE_PATH),
        "mark_flattened": mark is not None,
        "n_bars": (mark or {}).get("n_bars") if isinstance(mark, dict) else None,
    }


def flatten_core_equity_companion(*, initial_cash: float = DEFAULT_INITIAL_CASH) -> dict[str, Any] | None:
    """Hold CoreEquity NAV at ``initial_cash`` on the live GE1 mark (GE1 untouched)."""
    from rlbot.forward_live import ALGO_LIVE_RUN_ID
    from rlbot.forward_loop import publish_public_forward
    from rlbot.forward_mark import load_forward_mark, write_forward_mark

    mark = load_forward_mark(ALGO_LIVE_RUN_ID)
    if not isinstance(mark, dict):
        return None
    cash = float(initial_cash)
    nav = dict(mark.get("nav") or {})
    model = nav.get("model") if isinstance(nav.get("model"), list) else []
    n = int(mark.get("n_bars") or len(model) or 0)
    if n < 1:
        return mark
    series = [cash] * n
    stamps = mark.get("timestamps") or mark.get("dates") or []
    nav["core_equity"] = series
    mark["nav"] = nav
    stats = dict(mark.get("stats") or {})
    stats["core_equity"] = {
        "total_return": 0.0,
        "sharpe": None,
        "max_drawdown": 0.0,
        "nav": cash,
    }
    mark["stats"] = stats
    candles = dict(mark.get("candles") or {}) if isinstance(mark.get("candles"), dict) else {}
    candles["core_equity"] = [
        {
            "t": str(stamps[i]) if i < len(stamps) else "",
            "o": cash,
            "h": cash,
            "l": cash,
            "c": cash,
        }
        for i in range(n)
    ]
    mark["candles"] = candles
    cash_pos = {
        "label": "Cash",
        "ticker": "CASH",
        "weight": 1.0,
        "value_usd": cash,
        "price": 1.0,
    }
    mark["core_equity_weights"] = {"CASH": 1.0}
    mark["core_equity_positions"] = [cash_pos]
    alloc = dict(mark.get("allocations") or {}) if isinstance(mark.get("allocations"), dict) else {}
    alloc["core_equity"] = {
        "key": "core_equity",
        "label": "CoreEquity",
        "run_id": PAPER_RUN_ID,
        "nav": cash,
        "as_of": (mark.get("live") or {}).get("as_of_utc") or mark.get("generated_at_utc"),
        "price_source": "yahoo",
        "positions": [cash_pos],
        "latest_weights": {"CASH": 1.0},
    }
    mark["allocations"] = alloc
    write_forward_mark(mark)
    publish_public_forward(ALGO_LIVE_RUN_ID, mark)
    return mark


def _marks_from_closes(closes: dict[str, np.ndarray], i: int) -> dict[str, float]:
    return {
        k: float(v[i])
        for k, v in closes.items()
        if np.isfinite(v[i]) and float(v[i]) > 0
    }


def _mark_equity(cash: float, positions: dict[str, float], marks: dict[str, float]) -> float:
    eq = float(cash)
    for sym, qty in positions.items():
        px = marks.get(str(sym).upper())
        if px is not None and px > 0:
            eq += float(qty) * float(px)
    return eq


def orders_to_targets(
    equity: float,
    positions: dict[str, float],
    marks: dict[str, float],
    targets: dict[str, float],
    *,
    allow_symbols: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Share deltas to hit target weights at current marks."""
    eq = max(float(equity), 1e-9)
    want_shares: dict[str, float] = {}
    for sym, w in targets.items():
        su = str(sym).upper()
        if su in ("CASH",) or float(w) <= 0:
            continue
        if allow_symbols is not None and su not in allow_symbols:
            continue
        px = marks.get(su)
        if px is None or px <= 0:
            continue
        want_shares[su] = (float(w) * eq) / px
    have = {str(k).upper(): float(v) for k, v in positions.items()}
    if allow_symbols is not None:
        have = {k: v for k, v in have.items() if k in allow_symbols}
    orders: list[dict[str, Any]] = []
    for sym in sorted(set(have) | set(want_shares)):
        delta = want_shares.get(sym, 0.0) - have.get(sym, 0.0)
        if abs(delta) * marks.get(sym, 0.0) < 1.0:
            continue
        side = "buy" if delta > 0 else "sell"
        orders.append(
            {
                "symbol": sym,
                "side": side,
                "qty": abs(delta),
                "target_weight": float(targets.get(sym, 0.0)),
            }
        )
    orders.sort(key=lambda o: (0 if o["side"] == "sell" else 1, o["symbol"]))
    return orders


def _apply_fills(
    state: dict[str, Any],
    intents: list[dict[str, Any]],
    marks: dict[str, float],
) -> list[dict[str, Any]]:
    """Apply paper fills. Cash may go negative (pack QQQ cash-finance ≤ 1.4×)."""
    cash = float(state.get("cash") or 0.0)
    positions = {str(k).upper(): float(v) for k, v in (state.get("positions") or {}).items()}
    fills: list[dict[str, Any]] = []
    for intent in intents:
        sym = str(intent["symbol"]).upper()
        qty = float(intent["qty"])
        px = marks.get(sym)
        if px is None or px <= 0 or qty <= 0:
            continue
        if intent["side"] == "sell":
            have = positions.get(sym, 0.0)
            sell = min(have, qty)
            if sell <= 0:
                continue
            cash += sell * px
            positions[sym] = have - sell
            if positions[sym] <= 1e-10:
                positions.pop(sym, None)
            fills.append({"symbol": sym, "side": "sell", "qty": sell, "price": px})
        else:
            cost = qty * px
            cash -= cost
            positions[sym] = positions.get(sym, 0.0) + qty
            fills.append({"symbol": sym, "side": "buy", "qty": qty, "price": px})
    state["cash"] = cash
    state["positions"] = positions
    return fills


def _append_shadow_ledger(
    *,
    run_id: str,
    decision_bar: date,
    trade_date: date,
    target_weights: dict[str, float],
    meta: dict[str, Any],
    orders: list[dict[str, Any]],
) -> None:
    rec = {
        "run_id": run_id,
        "strategy_id": STRATEGY_ID,
        "decision_bar": str(decision_bar),
        "trade_date": str(trade_date),
        "as_of": str(trade_date),
        "target_weights": {
            ("Cash" if k == "CASH" else k): float(v) for k, v in target_weights.items()
        },
        "sleeve_meta": meta,
        "n_orders": len(orders),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _append_jsonl(ledger_path(run_id), rec)


def _update_cool_state(state: dict[str, Any], sleeve: Any) -> bool:
    """Copy sleeve-A overlay diagnostics; cool is replayed, not ticked per call."""
    state["peak_equity"] = float(sleeve.peak)
    state["flat_a"] = bool(sleeve.flat)
    state["cool_remaining"] = int(sleeve.cool_remaining)
    state["sleeve_a_equity"] = float(sleeve.equity)
    return bool(sleeve.flat)


def build_daily_forward_mark(
    *,
    state: dict[str, Any],
    closes: dict[str, np.ndarray],
    dates: list[date],
    as_of: date,
    meta: dict[str, Any],
) -> dict[str, Any]:
    tw = state.get("target_weights") or {"CASH": 1.0}
    risky_labels = [
        str(k).upper()
        for k, v in tw.items()
        if str(k).upper() not in ("CASH",) and float(v) > 0
    ]
    asset_cols = [c for c in risky_labels if c in closes]
    labels = ["Cash", *asset_cols] if asset_cols else ["Cash"]
    initial = float(state.get("equity") or DEFAULT_INITIAL_CASH)
    keep = [i for i, d in enumerate(dates) if d <= as_of][-60:]
    if not keep or not asset_cols:
        dates_out: list[Any] = [as_of]
        nav = np.asarray([initial], dtype=np.float64)
        cash_w = float(tw.get("CASH", tw.get("BIL", 1.0)))
        w_mat = np.asarray([[cash_w]], dtype=np.float64)
        labels = ["Cash"]
        spy_nav = nav.copy()
        ew_nav = nav.copy()
    else:
        dates_out = [dates[i] for i in keep]
        mat = np.column_stack([closes[c][keep] for c in asset_cols])
        w_risky = np.asarray([float(tw.get(c, 0.0)) for c in asset_cols], dtype=np.float64)
        cash_w = float(tw.get("CASH", 0.0))
        tot = cash_w + float(w_risky.sum())
        if tot <= 1e-12:
            cash_w, w_risky = 1.0, np.zeros_like(w_risky)
        else:
            cash_w /= tot
            w_risky = w_risky / tot
        nav = np.empty(len(keep), dtype=np.float64)
        nav[0] = initial
        for t in range(len(keep) - 1):
            rets = mat[t + 1] / np.maximum(mat[t], 1e-12) - 1.0
            nav[t + 1] = nav[t] * (1.0 + float(np.dot(w_risky, rets)))
        w_mat = np.tile(np.concatenate([[cash_w], w_risky]), (len(keep), 1))
        spy = closes["SPY"][keep]
        spy_nav = spy / max(float(spy[0]), 1e-12) * initial
        ew = np.full(len(asset_cols), 1.0 / max(len(asset_cols), 1), dtype=np.float64)
        ew_nav = np.empty(len(keep), dtype=np.float64)
        ew_nav[0] = initial
        for t in range(len(keep) - 1):
            rets = mat[t + 1] / np.maximum(mat[t], 1e-12) - 1.0
            ew_nav[t + 1] = ew_nav[t] * (1.0 + float(np.dot(ew, rets)))

    payload = build_forward_mark_payload(
        run_id=PAPER_RUN_ID,
        checkpoint_label="locked",
        dates=dates_out,
        nav_model=nav,
        nav_spy=spy_nav,
        nav_ew=ew_nav,
        weights=w_mat,
        asset_labels=labels,
        initial_cash=initial,
        holdout_start=str(as_of),
        holdout_end=None,
        note=(
            f"{STRATEGY_ID} (CoreEquity pack): weekly QQQ close + month-end "
            "GLD/TLT dual. No 2x/3x ETFs. Live 5m MTM via /api/forward; "
            "EW-10 = research sleeve; RLModel companion."
        ),
    )
    eq = float(state.get("equity") or initial)
    i_last = keep[-1] if keep else 0
    marks = _marks_from_closes(closes, i_last) if closes else {}
    positions: list[dict[str, Any]] = [
        {
            "label": "Cash",
            "ticker": "CASH",
            "weight": float(tw.get("CASH", 0.0)),
            "value_usd": float(tw.get("CASH", 0.0)) * eq,
            "price": 1.0,
        }
    ]
    for lab in labels[1:]:
        w = float(tw.get(lab, 0.0))
        positions.append(
            {
                "label": lab,
                "ticker": lab,
                "weight": w,
                "value_usd": w * eq,
                "price": marks.get(lab),
            }
        )
    positions.sort(key=lambda r: (-1 if r["label"] == "Cash" else 0, -float(r["weight"])))
    payload["positions"] = positions
    payload["strategy_id"] = STRATEGY_ID
    payload["sleeve_meta"] = meta
    payload["latest_weights"] = {
        ("Cash" if str(k).upper() == "CASH" else str(k)): float(v)
        for k, v in tw.items()
    }
    return payload


def run_paper_day(
    *,
    as_of: date | None = None,
    force_refresh: bool = False,
    set_active: bool = False,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    dry_run: bool = False,
    params: Any | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    params = params if params is not None else locked_params()
    state = load_state(initial_cash)
    dates, closes, _ohlc = fetch_daily_ohlc(
        list(panel_symbols()),
        force_refresh=force_refresh,
        cache_dir=PAPER_DIR,
    )
    requested = as_of or dates[-1]
    keep = [i for i, d in enumerate(dates) if d <= requested]
    if not keep:
        raise ValueError(f"no bars on or before {requested}")
    i = keep[-1]
    bar_day = dates[i]
    session_day = resolve_paper_session_day(as_of, bar_day)
    day = bar_day
    marks = _marks_from_closes(closes, i)

    state["equity"] = _mark_equity(
        float(state.get("cash") or 0.0),
        {str(k).upper(): float(v) for k, v in (state.get("positions") or {}).items()},
        marks,
    )

    panel_dates = dates[: i + 1]
    panel_closes = {k: np.asarray(v[: i + 1], dtype=np.float64) for k, v in closes.items()}
    panel_ohlc = {
        k: tuple(np.asarray(x[: i + 1], dtype=np.float64) for x in tup)
        for k, tup in _ohlc.items()
    }
    hot = str(params.eq_sym).upper()
    sleeve = sleeve_a_live_state(panel_dates, panel_closes, panel_ohlc[hot], params)
    flat = _update_cool_state(state, sleeve)
    plan = pack_paper_plan(
        aum=float(state["equity"] or initial_cash),
        dates=panel_dates,
        closes=panel_closes,
        ohlc=panel_ohlc,
    )
    targets_raw = dict(plan.get("targets") or {})
    targets_raw = apply_sleeve_a_to_targets(targets_raw, sleeve, params)
    targets = weights_from_targets(targets_raw, params)
    if flat:
        dual = str(targets_raw.get("dual_asset") or "GLD").upper()
        parked: dict[str, float] = {}
        for k, v in targets.items():
            ku = str(k).upper()
            if ku in {dual, "BIL", "CASH"}:
                parked[ku] = parked.get(ku, 0.0) + float(v)
            else:
                parked["BIL"] = parked.get("BIL", 0.0) + float(v)
        targets = parked
        actions_note = "flat_a"
    else:
        actions_note = "pack_signal"
    wk_live, me_live = session_rebalance_flags(dates, i, calendar_today=session_day)
    pos = {str(k).upper(): float(v) for k, v in (state.get("positions") or {}).items()}
    es_park = bool(
        flat and any(abs(float(pos.get(s, 0.0) or 0.0)) > 1e-8 for s in _SLEEVE_A)
    )
    seed = not any(abs(float(v)) > 1e-8 for v in pos.values())
    allow: set[str] = set()
    if seed:
        allow = set(_SLEEVE_A | _SLEEVE_B)
    else:
        if wk_live:
            allow |= set(_SLEEVE_A)
            allow.add("BIL")
        if me_live:
            allow |= set(_SLEEVE_B)
        if es_park:
            allow |= set(_SLEEVE_A)
            allow.add("BIL")
    meta = {
        "source": "CoreEquity",
        "pack_asof": plan.get("asof"),
        "equity_rebalance_due": wk_live,
        "dual_rebalance_due": me_live,
        "portfolio_targets": plan.get("portfolio_targets"),
        "flat_a": flat,
        "es_park": es_park,
        "actions_note": actions_note,
        "data_source": plan.get("data_source"),
    }
    rebal = bool(wk_live or me_live or seed or es_park)
    actions: list[str] = [f"signal:{day}", actions_note]
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    already_traded = core_paper_already_traded(state.get("last_trade_date"), session_day)
    if rebal and already_traded:
        actions.append("already_traded")
        rebal = False

    if rebal:
        actions.append("rebalance")
        if wk_live:
            actions.append("week_end")
        if me_live:
            actions.append("month_end")
        if es_park:
            actions.append("es_park")
        orders = orders_to_targets(
            float(state["equity"]),
            {str(k).upper(): float(v) for k, v in (state.get("positions") or {}).items()},
            marks,
            targets,
            allow_symbols=frozenset(allow) if allow else None,
        )
        if not dry_run:
            fills = _apply_fills(state, orders, marks)
            for o in orders:
                _append_jsonl(
                    ORDERS_PATH,
                    {
                        **o,
                        "as_of": str(session_day),
                        "trade_date": str(session_day),
                        "signal_date": str(bar_day),
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                    },
                )
            _append_shadow_ledger(
                run_id=PAPER_RUN_ID,
                decision_bar=session_day,
                trade_date=session_day,
                target_weights=targets,
                meta=meta,
                orders=orders,
            )
        state["target_weights"] = {str(k).upper(): float(v) for k, v in targets.items()}
        state["last_signal_date"] = str(bar_day)
        state["last_trade_date"] = str(session_day)
    else:
        state["target_weights"] = {str(k).upper(): float(v) for k, v in targets.items()}
        state["last_signal_date"] = str(bar_day)
        actions.append("hold")

    state["equity"] = _mark_equity(
        float(state.get("cash") or 0.0),
        {str(k).upper(): float(v) for k, v in (state.get("positions") or {}).items()},
        marks,
    )
    _update_cool_state(state, sleeve)

    mark_payload = None
    if not dry_run:
        save_state(state)
        if set_active:
            set_active_forward_run(PAPER_RUN_ID)
        from rlbot.forward_live import ALGO_LIVE_RUN_ID, refresh_forward_mark_live
        from rlbot.forward_mark import load_forward_mark

        # CoreEquity is a companion sleeve. Never reset or steal the GE1 mark.
        try:
            mark_payload = refresh_forward_mark_live(
                ALGO_LIVE_RUN_ID,
                force_price_refresh=True,
                reset_book=False,
            )
        except Exception:  # noqa: BLE001
            mark_payload = load_forward_mark(ALGO_LIVE_RUN_ID)

    return {
        "as_of": str(session_day),
        "bar_date": str(bar_day),
        "actions": actions,
        "orders": orders,
        "fills": fills,
        "target_weights": state.get("target_weights"),
        "sleeve_meta": meta,
        "equity": state.get("equity"),
        "flat_a": state.get("flat_a"),
        "n_positions": len(state.get("positions") or {}),
        "elapsed_s": round(time.time() - t0, 2),
        "mark_written": mark_payload is not None,
        "run_id": PAPER_RUN_ID,
        "strategy_id": STRATEGY_ID,
    }
