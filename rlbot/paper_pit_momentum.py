"""Paper broker loop for the locked PIT momentum strategy.

Wires ``rlbot.pit_momentum`` into the existing forward/shadow stack:

- Computes month-end signals from daily adjusted closes + PIT membership
- Places order intents on the next session (exec lag = 1)
- Appends ``execution/shadow_ledger_FINALMODEL.jsonl`` so ``/ops/forward``
  mark-to-markets and shows the stock book

No margin, no shorts. State lives under ``execution/paper_pit_momentum/``.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlbot.forward_mark import (
    build_forward_mark_payload,
    set_active_forward_run,
    write_forward_mark,
)
from rlbot.pit_momentum import (
    DEFAULT_LOCKED_CONFIG,
    DEFAULT_PIT_CSV,
    PAPER_RUN_ID,
    STRATEGY_ID,
    Params,
    compute_target_weights,
    execution_date,
    last_actionable_signal,
    load_pit_snapshots,
    membership_asof,
    month_end_signals,
    orders_to_targets,
    to_yahoo_symbol,
    weights_with_cash,
)
from rlbot.run_artifacts import PROJECT_ROOT

EXECUTION_DIR = PROJECT_ROOT / "execution"
PAPER_DIR = EXECUTION_DIR / "paper_pit_momentum"
STATE_PATH = PAPER_DIR / "state.json"
ORDERS_PATH = PAPER_DIR / "order_intents.jsonl"
PRICE_CACHE_PATH = PAPER_DIR / "daily_prices.csv"
DEFAULT_INITIAL_CASH = 100_000.0
# Lookback buffer beyond formation window for calendar construction.
PRICE_LOOKBACK_CALENDAR_DAYS = 400


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def default_state(initial_cash: float = DEFAULT_INITIAL_CASH) -> dict[str, Any]:
    return {
        "strategy_id": STRATEGY_ID,
        "run_id": PAPER_RUN_ID,
        "equity": float(initial_cash),
        "cash": float(initial_cash),
        "positions": {},  # symbol -> shares
        "target_weights": {"CASH": 1.0},
        "pending_signal": None,
        "last_signal_date": None,
        "last_trade_date": None,
        "updated_at_utc": None,
    }


def load_state(initial_cash: float = DEFAULT_INITIAL_CASH) -> dict[str, Any]:
    st = default_state(initial_cash)
    st.update(_read_json(STATE_PATH))
    st.setdefault("positions", {})
    st.setdefault("target_weights", {"CASH": 1.0})
    return st


def save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(STATE_PATH, state)


def _prices_dict_from_frame(frame: pd.DataFrame) -> dict[str, list[tuple[date, float]]]:
    """Wide adj-close frame (DatetimeIndex × ticker) → signal-engine price dict."""
    out: dict[str, list[tuple[date, float]]] = {}
    if frame is None or frame.empty:
        return out
    idx = pd.DatetimeIndex(frame.index)
    for col in frame.columns:
        sym = str(col).upper()
        series = frame[col].dropna()
        if series.empty:
            continue
        rows: list[tuple[date, float]] = []
        for ts, px in series.items():
            try:
                rows.append((pd.Timestamp(ts).date(), float(px)))
            except (TypeError, ValueError):
                continue
        if rows:
            out[sym] = rows
    return out


def _trading_calendar_from_prices(
    prices: dict[str, list[tuple[date, float]]],
) -> list[date]:
    if "SPY" in prices and prices["SPY"]:
        return [d for d, _ in prices["SPY"]]
    return sorted({d for series in prices.values() for d, _ in series})


def fetch_daily_adj_closes(
    symbols: list[str],
    *,
    start: date,
    end: date | None = None,
    cache_path: Path = PRICE_CACHE_PATH,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download (or refresh) daily adjusted closes; cache under execution/.

    Columns are uppercase PIT-style symbols (dots). Yahoo uses dashes.
    """
    end_d = end or date.today()
    want = sorted({s.upper() for s in symbols if s})
    if "SPY" not in want:
        want = ["SPY", *want]

    cached = pd.DataFrame()
    parquet_alt = cache_path.with_suffix(".parquet")
    if not force_refresh:
        for path in (cache_path, parquet_alt):
            if not path.is_file():
                continue
            try:
                if path.suffix == ".parquet":
                    cached = pd.read_parquet(path)
                else:
                    cached = pd.read_csv(path, index_col=0, parse_dates=True)
                if not isinstance(cached.index, pd.DatetimeIndex):
                    cached.index = pd.to_datetime(cached.index)
                cached.columns = [str(c).upper() for c in cached.columns]
                break
            except Exception:  # noqa: BLE001
                cached = pd.DataFrame()

    need = [s for s in want if s not in cached.columns]
    tip_stale = True
    if not cached.empty:
        tip = pd.Timestamp(cached.index.max()).date()
        tip_stale = tip < (end_d - timedelta(days=3))

    if need or tip_stale or force_refresh:
        import yfinance as yf

        yahoo_map = {to_yahoo_symbol(s): s for s in want}
        yf_syms = list(yahoo_map.keys())
        # Batch in chunks — Yahoo flaky on 500-ticker single calls.
        pieces: dict[str, pd.Series] = {}
        chunk_size = 50
        n_chunks = (len(yf_syms) + chunk_size - 1) // chunk_size
        print(
            f"[paper_pit] fetching daily closes for {len(yf_syms)} symbols "
            f"in {n_chunks} chunk(s) ({start} → {end_d})",
            flush=True,
        )
        for i in range(0, len(yf_syms), chunk_size):
            chunk = yf_syms[i : i + chunk_size]
            print(
                f"[paper_pit]   chunk {i // chunk_size + 1}/{n_chunks} "
                f"({len(chunk)} tickers)…",
                flush=True,
            )
            raw = yf.download(
                chunk,
                start=str(start),
                end=str(end_d + timedelta(days=1)),
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = set(raw.columns.get_level_values(0))
                for ysym in chunk:
                    pit_sym = yahoo_map[ysym]
                    if ysym not in level0:
                        continue
                    try:
                        close = raw[ysym]["Close"]
                    except (KeyError, TypeError):
                        continue
                    s = close.dropna()
                    if not s.empty:
                        pieces[pit_sym] = s
            elif "Close" in raw.columns and len(chunk) == 1:
                pieces[yahoo_map[chunk[0]]] = raw["Close"].dropna()

        if pieces:
            fresh = pd.DataFrame(pieces)
            fresh.index = pd.to_datetime(fresh.index)
            if cached.empty:
                cached = fresh
            else:
                for col in fresh.columns:
                    if col in cached.columns:
                        cached[col] = fresh[col].combine_first(cached[col])
                    else:
                        cached[col] = fresh[col]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached.sort_index().to_csv(cache_path)

    if cached.empty:
        return cached
    cols = [c for c in want if c in cached.columns]
    out = cached.loc[:, cols].sort_index()
    out = out.loc[(out.index.date >= start) & (out.index.date <= end_d)]
    return out


def _marks_from_frame(frame: pd.DataFrame, as_of: date) -> dict[str, float]:
    if frame.empty:
        return {}
    sub = frame.loc[frame.index.date <= as_of]
    if sub.empty:
        return {}
    row = sub.iloc[-1]
    return {str(c).upper(): float(v) for c, v in row.items() if pd.notna(v) and float(v) > 0}


def _mark_equity(
    cash: float,
    positions: dict[str, float],
    marks: dict[str, float],
) -> float:
    eq = float(cash)
    for sym, qty in positions.items():
        px = marks.get(sym.upper())
        if px is None or px <= 0:
            # Spec: never silently drop shares — keep last notion if no quote.
            continue
        eq += float(qty) * float(px)
    return eq


def _apply_fills_at_marks(
    state: dict[str, Any],
    intents: list[dict],
    marks: dict[str, float],
) -> list[dict]:
    """Paper fills at last mark (MOC approximation). Sells first, then buys."""
    cash = float(state.get("cash") or 0.0)
    positions = {str(k).upper(): float(v) for k, v in (state.get("positions") or {}).items()}
    fills: list[dict] = []

    for intent in intents:
        if intent.get("side") != "sell":
            continue
        sym = str(intent["symbol"]).upper()
        qty = float(intent["qty"])
        px = marks.get(sym)
        if px is None or px <= 0 or qty <= 0:
            continue
        have = positions.get(sym, 0.0)
        sell = min(have, qty)
        if sell <= 0:
            continue
        cash += sell * px
        positions[sym] = have - sell
        if positions[sym] <= 1e-10:
            positions.pop(sym, None)
        fills.append({"symbol": sym, "side": "sell", "qty": sell, "price": px})

    for intent in intents:
        if intent.get("side") != "buy":
            continue
        sym = str(intent["symbol"]).upper()
        qty = float(intent["qty"])
        px = marks.get(sym)
        if px is None or px <= 0 or qty <= 0:
            continue
        cost = qty * px
        if cost > cash + 1e-6:
            qty = cash / px
            cost = qty * px
        if qty <= 0 or cost < 1.0:
            continue
        cash -= cost
        positions[sym] = positions.get(sym, 0.0) + qty
        fills.append({"symbol": sym, "side": "buy", "qty": qty, "price": px})

    state["cash"] = float(cash)
    state["positions"] = positions
    state["equity"] = _mark_equity(cash, positions, marks)
    return fills


def _append_shadow_ledger(
    *,
    run_id: str,
    decision_bar: date,
    trade_date: date,
    target_weights: dict[str, float],
    signal_date: date,
    orders: list[dict],
) -> None:
    path = ledger_path(run_id)
    existing = {
        (str(r.get("decision_bar") or r.get("as_of")), str(r.get("checkpoint")))
        for r in _read_jsonl(path)
    }
    # decision_bar = signal date (weights computed after that close)
    key = (str(decision_bar), "locked")
    if key in existing:
        return
    with_cash = weights_with_cash(target_weights)
    rec = {
        "run_id": run_id,
        "strategy_id": STRATEGY_ID,
        "as_of": str(trade_date),
        "decision_bar": str(decision_bar),
        "signal_date": str(signal_date),
        "trade_date": str(trade_date),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": "locked",
        "target_weights": with_cash,
        "cash_weight": float(with_cash.get("CASH", 0.0)),
        "orders": orders,
        "provenance": {
            "locked_config": str(DEFAULT_LOCKED_CONFIG.relative_to(PROJECT_ROOT)),
            "pit_csv": str(DEFAULT_PIT_CSV.relative_to(PROJECT_ROOT)),
            "params_locked": True,
        },
        "realized": None,
    }
    _append_jsonl(path, rec)


def prepare_prices(
    as_of: date | None = None,
    *,
    force_refresh: bool = False,
    params: Params | None = None,
) -> tuple[pd.DataFrame, dict[str, list[tuple[date, float]]], list[date], Params]:
    p = params or Params.from_locked_json()
    print("[paper_pit] loading PIT membership…", flush=True)
    snaps = load_pit_snapshots()
    print(f"[paper_pit] PIT snapshots: {len(snaps)} (through {snaps[-1][0]})", flush=True)
    end = as_of or date.today()
    # PIT file may end before today — use last snapshot date as membership as-of.
    pit_end = snaps[-1][0] if snaps else end
    membership_day = min(end, pit_end)
    # Current PIT members + SPY. Names that left mid-formation window are simply
    # ineligible (missing price) rather than forcing a wider download.
    members_now = membership_asof(snaps, membership_day)
    symbols = sorted(members_now | {"SPY"})
    print(f"[paper_pit] universe size: {len(symbols)}", flush=True)
    start = end - timedelta(days=PRICE_LOOKBACK_CALENDAR_DAYS)
    frame = fetch_daily_adj_closes(
        symbols, start=start, end=end, force_refresh=force_refresh
    )
    prices = _prices_dict_from_frame(frame)
    cal = _trading_calendar_from_prices(prices)
    return frame, prices, cal, p


def run_paper_day(
    *,
    as_of: date | None = None,
    force_refresh: bool = False,
    set_active: bool = True,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One paper session: signal / trade / hold, then refresh forward mark."""
    t0 = time.time()
    print(f"[paper_pit] run-day starting (as_of={as_of or 'last bar'})", flush=True)
    frame, prices, cal, params = prepare_prices(as_of, force_refresh=force_refresh)
    print(
        f"[paper_pit] prices ready: {frame.shape[1] if not frame.empty else 0} symbols, "
        f"{len(cal)} sessions",
        flush=True,
    )
    if not cal:
        raise RuntimeError(
            "No trading calendar from price fetch — check network / yfinance."
        )

    day = as_of or cal[-1]
    if day not in cal:
        # Snap to last available session on/before as_of.
        prior = [d for d in cal if d <= day]
        if not prior:
            raise RuntimeError(f"No price bars on/before {day}")
        day = prior[-1]

    state = load_state(initial_cash)
    marks = _marks_from_frame(frame, day)
    # Mark open positions to latest closes before sizing.
    state["equity"] = _mark_equity(
        float(state.get("cash") or 0.0),
        {str(k).upper(): float(v) for k, v in (state.get("positions") or {}).items()},
        marks,
    )

    signals = set(month_end_signals(cal))
    actions: list[str] = []
    orders: list[dict] = []
    fills: list[dict] = []

    # --- Bootstrap first: establish the last fully-traded book if empty ---
    if not state.get("last_trade_date") and not (state.get("positions") or {}):
        actionable = last_actionable_signal(
            cal, day, lag=params.execution_lag_trading_days
        )
        if actionable is not None:
            sig_d, trade_d = actionable
            targets = compute_target_weights(
                sig_d, prices, DEFAULT_PIT_CSV, params=params
            )
            equity = float(state.get("equity") or initial_cash)
            state["cash"] = equity
            state["positions"] = {}
            orders = orders_to_targets(equity, {}, marks, targets)
            if not dry_run:
                fills = _apply_fills_at_marks(state, orders, marks)
                for o in orders:
                    _append_jsonl(
                        ORDERS_PATH,
                        {
                            **o,
                            "as_of": str(day),
                            "trade_date": str(trade_d),
                            "signal_date": str(sig_d),
                            "bootstrap": True,
                            "recorded_at_utc": datetime.now(timezone.utc).isoformat(
                                timespec="seconds"
                            ),
                        },
                    )
                _append_shadow_ledger(
                    run_id=PAPER_RUN_ID,
                    decision_bar=sig_d,
                    trade_date=trade_d,
                    target_weights=targets,
                    signal_date=sig_d,
                    orders=orders,
                )
            state["target_weights"] = weights_with_cash(targets)
            state["last_signal_date"] = str(sig_d)
            state["last_trade_date"] = str(trade_d)
            state["pending_signal"] = None
            actions.append(f"bootstrap:{sig_d}->{trade_d}")
            # Refresh marks/equity after fills.
            state["equity"] = _mark_equity(
                float(state.get("cash") or 0.0),
                {
                    str(k).upper(): float(v)
                    for k, v in (state.get("positions") or {}).items()
                },
                marks,
            )

    # --- Signal day: compute targets, no orders ---
    if day in signals:
        targets = compute_target_weights(
            day, prices, DEFAULT_PIT_CSV, params=params
        )
        trade = execution_date(
            day, cal, lag=params.execution_lag_trading_days
        )
        state["pending_signal"] = {
            "signal_date": str(day),
            "trade_date": str(trade) if trade else None,
            "targets": targets,
        }
        state["last_signal_date"] = str(day)
        actions.append(f"signal:{day}")

    pending = state.get("pending_signal")
    # If signal was computed on the calendar tip, trade_date is unknown until a
    # later session appears in the price calendar — resolve it here.
    if pending and not pending.get("trade_date") and pending.get("signal_date"):
        sig_pending = date.fromisoformat(str(pending["signal_date"])[:10])
        resolved = execution_date(
            sig_pending, cal, lag=params.execution_lag_trading_days
        )
        if resolved is not None:
            pending = {**pending, "trade_date": str(resolved)}
            state["pending_signal"] = pending
            actions.append(f"resolve_trade_date:{resolved}")

    # --- Trade day: sells first, then buys; record ledger ---
    if pending and pending.get("trade_date"):
        trade_d = date.fromisoformat(str(pending["trade_date"])[:10])
        if day >= trade_d and str(state.get("last_trade_date")) != str(trade_d):
            targets = {
                str(k).upper(): float(v)
                for k, v in (pending.get("targets") or {}).items()
            }
            for sym, qty in list((state.get("positions") or {}).items()):
                su = str(sym).upper()
                if su not in marks and qty:
                    actions.append(f"missing_quote:{su}")
            equity = float(state["equity"])
            orders = orders_to_targets(
                equity,
                {
                    str(k).upper(): float(v)
                    for k, v in (state.get("positions") or {}).items()
                },
                marks,
                targets,
            )
            if not dry_run:
                fills = _apply_fills_at_marks(state, orders, marks)
                for o in orders:
                    _append_jsonl(
                        ORDERS_PATH,
                        {
                            **o,
                            "as_of": str(day),
                            "trade_date": str(trade_d),
                            "signal_date": pending.get("signal_date"),
                            "recorded_at_utc": datetime.now(timezone.utc).isoformat(
                                timespec="seconds"
                            ),
                        },
                    )
                sig_d = date.fromisoformat(str(pending["signal_date"])[:10])
                _append_shadow_ledger(
                    run_id=PAPER_RUN_ID,
                    decision_bar=sig_d,
                    trade_date=trade_d,
                    target_weights=targets,
                    signal_date=sig_d,
                    orders=orders,
                )
            state["target_weights"] = weights_with_cash(targets)
            state["last_trade_date"] = str(trade_d)
            state["pending_signal"] = None
            actions.append(f"trade:{trade_d}")

    # Re-mark equity after any fills.
    state["equity"] = _mark_equity(
        float(state.get("cash") or 0.0),
        {str(k).upper(): float(v) for k, v in (state.get("positions") or {}).items()},
        marks,
    )

    if not dry_run:
        save_state(state)
        if set_active:
            set_active_forward_run(PAPER_RUN_ID)
        mark_payload = build_daily_forward_mark(
            state=state,
            frame=frame,
            as_of=day,
            params=params,
        )
        write_forward_mark(mark_payload)
    else:
        mark_payload = None

    return {
        "as_of": str(day),
        "actions": actions,
        "orders": orders,
        "fills": fills,
        "target_weights": state.get("target_weights"),
        "equity": state.get("equity"),
        "n_positions": len(state.get("positions") or {}),
        "elapsed_s": round(time.time() - t0, 2),
        "mark_written": mark_payload is not None,
        "run_id": PAPER_RUN_ID,
    }


def build_daily_forward_mark(
    *,
    state: dict[str, Any],
    frame: pd.DataFrame,
    as_of: date,
    params: Params,
) -> dict[str, Any]:
    """Daily (1d) forward-mark seed so live 5m refresh has weights + labels."""
    tw = state.get("target_weights") or {"CASH": 1.0}
    risky_labels = [
        str(k).upper()
        for k, v in tw.items()
        if str(k).upper() != "CASH" and float(v) > 0
    ]
    labels = ["Cash", *risky_labels]
    sub = frame.loc[frame.index.date <= as_of].tail(60) if not frame.empty else frame
    asset_cols = [c for c in risky_labels if c in getattr(sub, "columns", [])]
    labels = ["Cash", *asset_cols] if asset_cols else labels
    initial = float(state.get("equity") or DEFAULT_INITIAL_CASH)

    if sub.empty or not asset_cols:
        dates: list[Any] = [as_of]
        nav = np.asarray([initial], dtype=np.float64)
        cash_w = float(tw.get("CASH", 1.0))
        w_mat = np.asarray([[cash_w]], dtype=np.float64)
        labels = ["Cash"]
        spy_nav = nav.copy()
        ew_nav = nav.copy()
    else:
        closes = sub[asset_cols].astype(float).ffill().bfill().to_numpy(dtype=np.float64)
        dates = [pd.Timestamp(t).date() for t in sub.index]
        w_risky = np.asarray(
            [float(tw.get(c, 0.0)) for c in asset_cols], dtype=np.float64
        )
        cash_w = float(tw.get("CASH", max(0.0, 1.0 - float(w_risky.sum()))))
        tot = cash_w + float(w_risky.sum())
        if tot <= 1e-12:
            cash_w, w_risky = 1.0, np.zeros_like(w_risky)
        else:
            cash_w /= tot
            w_risky = w_risky / tot
        nav = np.empty(len(dates), dtype=np.float64)
        nav[0] = initial
        for t in range(len(dates) - 1):
            rets = closes[t + 1] / np.maximum(closes[t], 1e-12) - 1.0
            nav[t + 1] = nav[t] * (1.0 + float(np.dot(w_risky, rets)))
        w_mat = np.tile(np.concatenate([[cash_w], w_risky]), (len(dates), 1))
        if "SPY" in sub.columns:
            spy = sub["SPY"].astype(float).ffill().to_numpy(dtype=np.float64)
            spy_nav = spy / max(float(spy[0]), 1e-12) * initial
        else:
            spy_nav = nav.copy()
        ew = np.full(len(asset_cols), 1.0 / max(len(asset_cols), 1), dtype=np.float64)
        ew_nav = np.empty(len(dates), dtype=np.float64)
        ew_nav[0] = initial
        for t in range(len(dates) - 1):
            rets = closes[t + 1] / np.maximum(closes[t], 1e-12) - 1.0
            ew_nav[t + 1] = ew_nav[t] * (1.0 + float(np.dot(ew, rets)))

    # Seed mark starts at the latest session; live refresh rebuilds today's 5m grid.
    seed_start = str(as_of)
    payload = build_forward_mark_payload(
        run_id=PAPER_RUN_ID,
        checkpoint_label="locked",
        dates=dates,
        nav_model=nav,
        nav_spy=spy_nav,
        nav_ew=ew_nav,
        weights=w_mat,
        asset_labels=labels,
        initial_cash=initial,
        holdout_start=seed_start,
        holdout_end=None,
        note=(
            f"{STRATEGY_ID}: cash long-only PIT momentum "
            f"(top_n={params.top_n}, lookback={params.lookback_trading_days}, "
            f"skip={params.skip_trading_days}, gross={params.portfolio_gross_weight}). "
            "Orders on session after month-end; live 5m MTM via /api/forward "
            "(EW-10 = research sleeve, not stock picks; LIVE_MODEL companion series)."
        ),
    )
    eq = float(state.get("equity") or initial)
    last_marks = _marks_from_frame(frame, as_of) if not frame.empty else {}
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
                "price": last_marks.get(lab),
            }
        )
    positions.sort(
        key=lambda r: (-1 if r["label"] == "Cash" else 0, -float(r["weight"]))
    )
    payload["positions"] = positions
    payload["strategy_id"] = STRATEGY_ID
    payload["latest_weights"] = {
        ("Cash" if str(k).upper() == "CASH" else str(k)): float(v)
        for k, v in tw.items()
    }
    return payload
