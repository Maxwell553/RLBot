"""Paper / forward loop for GeneralEquity1 (prod_return_alpha_v3 pack).

Targets come from the locked ``GeneralEquity1/`` pack (read-only). Yahoo daily
bars are used only for MTM / 5m live refresh. State under
``execution/paper_general_equity1/``.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlbot.forward_mark import (
    build_forward_mark_payload,
    set_active_forward_run,
)
from rlbot.pack_general_equity1 import (
    DEFAULT_INITIAL_CASH,
    PAPER_RUN_ID,
    STRATEGY_ID,
    latest_portfolio_weights,
    locked_params,
    paper_plan as pack_paper_plan,
)
from rlbot.prod_return_alpha import (
    fetch_daily_ohlc,
    session_rebalance_flags,
    weights_with_cash,
)
from rlbot.run_artifacts import PROJECT_ROOT

EXECUTION_DIR = PROJECT_ROOT / "execution"
PAPER_DIR = EXECUTION_DIR / "paper_general_equity1"
STATE_PATH = PAPER_DIR / "state.json"
ORDERS_PATH = PAPER_DIR / "order_intents.jsonl"
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
    # Keep the fill timestamp on later hold-only marks so 5m lot MTM still
    # starts at the actual trade, not "now".
    if not state.get("updated_at_utc") or trade == today:
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(STATE_PATH, state)


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
) -> list[dict[str, Any]]:
    """Share deltas to hit target weights at current marks."""
    eq = max(float(equity), 1e-9)
    want_shares: dict[str, float] = {}
    for sym, w in targets.items():
        su = str(sym).upper()
        if su in ("CASH",) or float(w) <= 0:
            continue
        px = marks.get(su)
        if px is None or px <= 0:
            continue
        want_shares[su] = (float(w) * eq) / px
    have = {str(k).upper(): float(v) for k, v in positions.items()}
    orders: list[dict[str, Any]] = []
    for sym in sorted(set(have) | set(want_shares)):
        delta = want_shares.get(sym, 0.0) - have.get(sym, 0.0)
        if abs(delta) * marks.get(sym, 0.0) < 1.0:  # <$1 notional
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
    # Sells first for cash.
    orders.sort(key=lambda o: (0 if o["side"] == "sell" else 1, o["symbol"]))
    return orders


def _apply_fills(
    state: dict[str, Any],
    intents: list[dict[str, Any]],
    marks: dict[str, float],
) -> list[dict[str, Any]]:
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
            if cost > cash + 1e-6:
                qty = cash / px
                cost = qty * px
            if qty <= 0:
                continue
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
    tw = weights_with_cash(target_weights)
    rec = {
        "run_id": run_id,
        "strategy_id": STRATEGY_ID,
        "decision_bar": str(decision_bar),
        "trade_date": str(trade_date),
        "as_of": str(trade_date),
        "target_weights": {
            ("Cash" if k == "CASH" else k): float(v) for k, v in tw.items()
        },
        "sleeve_meta": meta,
        "n_orders": len(orders),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _append_jsonl(ledger_path(run_id), rec)


def _update_cool_state(state: dict[str, Any], equity: float, p: Any | None = None) -> bool:
    """Equity cool on sleeve A: −es from peak → flat for ``cool`` sessions."""
    p = p if p is not None else locked_params()
    peak = float(state.get("peak_equity") or equity)
    if equity > peak:
        peak = equity
        state["peak_equity"] = peak
    flat = bool(state.get("flat_a"))
    cool = int(state.get("cool_remaining") or 0)
    if (not flat) and peak > 0 and (equity / peak - 1.0) <= -float(p.es):
        flat = True
        cool = int(p.cool)
    elif flat:
        if cool > 0:
            cool -= 1
        else:
            # Re-arm when cool expires (trend filter still applied in signal).
            flat = False
            state["peak_equity"] = equity
    state["flat_a"] = flat
    state["cool_remaining"] = cool
    return flat


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
    # Prefer tradeable sleeve names present in the panel.
    asset_cols = [c for c in risky_labels if c in closes]
    labels = ["Cash", *asset_cols] if asset_cols else ["Cash"]
    initial = float(state.get("equity") or DEFAULT_INITIAL_CASH)
    # Last ~60 sessions for seed NAV.
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
        # BIL is a risky label here; do not double-count as cash.
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
            f"{STRATEGY_ID} (GeneralEquity1 pack): weekly TQQQ+QQQ hybrid + month-end "
            "dual momentum. Live 5m MTM via /api/forward; EW-10 = research sleeve; "
            "RLModel companion."
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
    set_active: bool = True,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    dry_run: bool = False,
    params: Any | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    params = params if params is not None else locked_params()
    state = load_state(initial_cash)
    dates, closes, _ohlc = fetch_daily_ohlc(force_refresh=force_refresh)
    day = as_of or dates[-1]
    keep = [i for i, d in enumerate(dates) if d <= day]
    if not keep:
        raise ValueError(f"no bars on or before {day}")
    i = keep[-1]
    day = dates[i]
    marks = _marks_from_closes(closes, i)

    # Mark equity + cool state before sizing.
    state["equity"] = _mark_equity(
        float(state.get("cash") or 0.0),
        {str(k).upper(): float(v) for k, v in (state.get("positions") or {}).items()},
        marks,
    )
    flat = _update_cool_state(state, float(state["equity"]), params)

    # Locked rules on the live Yahoo daily panel (not frozen pack bars.db).
    panel_dates = dates[: i + 1]
    panel_closes = {k: np.asarray(v[: i + 1], dtype=np.float64) for k, v in closes.items()}
    panel_ohlc = {
        k: tuple(np.asarray(x[: i + 1], dtype=np.float64) for x in tup)
        for k, tup in _ohlc.items()
    }
    plan = pack_paper_plan(
        aum=float(state["equity"] or initial_cash),
        dates=panel_dates,
        closes=panel_closes,
        ohlc=panel_ohlc,
    )
    targets = latest_portfolio_weights(
        aum=float(state["equity"] or initial_cash),
        dates=panel_dates,
        closes=panel_closes,
        ohlc=panel_ohlc,
    )
    if flat:
        # Cool-down: park sleeve A (risky ETFs except dual/GLD stay if already dual-only).
        dual_keys = {str(plan.get("targets", {}).get("dual_asset") or "GLD").upper()}
        parked: dict[str, float] = {}
        for k, v in targets.items():
            ku = str(k).upper()
            if ku in dual_keys or ku == "CASH":
                parked[ku] = float(v)
            else:
                parked["CASH"] = parked.get("CASH", 0.0) + float(v)
        targets = weights_with_cash(parked)
        actions_note = "flat_a"
    else:
        actions_note = "pack_signal"
    # Live calendar only. Pack week_end_mask always flags the series tip.
    wk_live, me_live = session_rebalance_flags(dates, i)
    meta = {
        "source": "GeneralEquity1",
        "pack_asof": plan.get("asof"),
        "equity_rebalance_due": wk_live,
        "dual_rebalance_due": me_live,
        "portfolio_targets": plan.get("portfolio_targets"),
        "flat_a": flat,
        "actions_note": actions_note,
        "data_source": plan.get("data_source"),
    }
    rebal = bool(
        wk_live or me_live or not (state.get("positions") or {})
    )
    actions: list[str] = [f"signal:{day}", actions_note]
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    already_traded = str(state.get("last_trade_date") or "") == str(day)
    if rebal and already_traded:
        actions.append("already_traded")
        rebal = False

    if rebal:
        actions.append("rebalance")
        if wk_live:
            actions.append("week_end")
        if me_live:
            actions.append("month_end")
        orders = orders_to_targets(
            float(state["equity"]),
            {str(k).upper(): float(v) for k, v in (state.get("positions") or {}).items()},
            marks,
            targets,
        )
        if not dry_run:
            fills = _apply_fills(state, orders, marks)
            for o in orders:
                _append_jsonl(
                    ORDERS_PATH,
                    {
                        **o,
                        "as_of": str(day),
                        "trade_date": str(day),
                        "signal_date": str(day),
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                    },
                )
            _append_shadow_ledger(
                run_id=PAPER_RUN_ID,
                decision_bar=day,
                trade_date=day,
                target_weights=targets,
                meta=meta,
                orders=orders,
            )
        # Pack weights already include CASH — do not run weights_with_cash (double-counts).
        state["target_weights"] = {str(k).upper(): float(v) for k, v in targets.items()}
        state["last_signal_date"] = str(day)
        state["last_trade_date"] = str(day)
    else:
        # Hold; still refresh displayed targets for the live mark.
        state["target_weights"] = {str(k).upper(): float(v) for k, v in targets.items()}
        state["last_signal_date"] = str(day)
        actions.append("hold")

    state["equity"] = _mark_equity(
        float(state.get("cash") or 0.0),
        {str(k).upper(): float(v) for k, v in (state.get("positions") or {}).items()},
        marks,
    )
    _update_cool_state(state, float(state["equity"]), params)

    mark_payload = None
    if not dry_run:
        save_state(state)
        if set_active:
            set_active_forward_run(PAPER_RUN_ID)
        # Never publish a multi-month daily NAV replay as the live forward book.
        # Keep ledger/weights; start (or continue) today's 5m MTM at $initial_cash.
        from rlbot.forward_live import refresh_forward_mark_live, seed_flat_forward_baseline
        from rlbot.forward_mark import load_forward_mark

        existing = load_forward_mark(PAPER_RUN_ID)
        has_live_5m = (
            isinstance(existing, dict) and str(existing.get("bar_interval") or "") == "5m"
        )
        try:
            mark_payload = refresh_forward_mark_live(
                PAPER_RUN_ID,
                force_price_refresh=True,
                reset_book=not has_live_5m,
            )
        except Exception:  # noqa: BLE001
            mark_payload = None
        if not isinstance(mark_payload, dict):
            mark_payload = seed_flat_forward_baseline(
                PAPER_RUN_ID,
                initial_cash=float(initial_cash),
                include_live_model=True,
            )

    return {
        "as_of": str(day),
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
