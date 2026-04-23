#!/usr/bin/env python3
"""
IBKR (TWS) paper: yfinance + same obs as training -> 65M model -> MKT orders toward target weights.

  cd to repo root, TWS (paper) running with API on 7497, then:
    .venv/bin/python ibkr_paper/run_ibkr_paper.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (REPO, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import contract_map
from execution_sizing_map import SIZING_YF
from trading_env import portfolio_weights_from_action

from ibkr_rebalance import (
    DEFAULT_PORT,
    connect_ib,
    dry_run_deltas,
    mkt_dollar_by_logical,
    nlv_usd,
    qualify_legs,
    rebalance_toward_targets,
    RebalanceConfig,
)

_STATE_FILE = HERE / "state.json"
_JOURNAL = HERE / "trade_journal.txt"


def _load_paper_trade():
    p = REPO / "paper_trade" / "paper_trade.py"
    spec = importlib.util.spec_from_file_location("paper_trade_rl", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_state() -> dict:
    if not _STATE_FILE.is_file():
        return {"last_bar_date": None, "peak_nav": 0.0}
    return json.loads(_STATE_FILE.read_text(encoding="utf-8"))


def _save_state(st: dict) -> None:
    _STATE_FILE.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")


def _synthetic_cash_units(
    pt,
    mkt: dict[str, float],
    ohlcv: np.ndarray,
    t: int,
    nlv: float,
) -> tuple[float, np.ndarray, float, float]:
    """Virtual cash/units/peak for build_obs: units[i]=mkt_i/yf_close so weights match account."""
    TICK = pt.TICKERS
    close = ohlcv[t, :, 3]
    units = np.array([mkt.get(TICK[i], 0.0) / max(float(close[i]), 1e-12) for i in range(10)], dtype=np.float64)
    cash = float(nlv) - float(sum(mkt.get(x, 0.0) for x in TICK))
    nav = cash + float(np.dot(units, close))
    return cash, units, float(nav), max(nav, 1e-6)


def main() -> None:
    p = argparse.ArgumentParser(description="IBKR paper: RL portfolio rebalance (TWS + ib_insync)")
    p.add_argument("--run-id", default="65M_4_20_26_a", help="models/<id>/ best_model + vec_normalize")
    p.add_argument("--model", default="", help="Override policy .zip path")
    p.add_argument("--vec-normalize", default="", help="Override vec_normalize.pkl")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Inference + print targets / deltas; no TWS, no state write, no orders",
    )
    p.add_argument(
        "--no-orders",
        action="store_true",
        help="Connect to TWS, read NLV/positions, run inference, print; do not place orders",
    )
    p.add_argument(
        "--use-fixed-nlv",
        type=float,
        default=0.0,
        metavar="USD",
        help="Size targets from this notional (e.g. 5e6) instead of account NetLiquidation",
    )
    p.add_argument("--ib-host", default=os.environ.get("IB_HOST", "127.0.0.1"))
    p.add_argument(
        "--ib-port",
        type=int,
        default=int(os.environ.get("IB_PORT", str(DEFAULT_PORT))),
    )
    p.add_argument("--ib-client-id", type=int, default=int(os.environ.get("IB_CLIENT_ID", "7")))
    p.add_argument("--min-leg-dollars", type=float, default=100.0)
    p.add_argument("--forex-lot", type=float, default=1000.0)
    p.add_argument("--force", action="store_true", help="Ignore one-run-per-bar guard in state")
    p.add_argument(
        "--contracts",
        type=str,
        default="",
        help="Path to custom contracts.json (default: ibkr_paper/contracts.json)",
    )
    args = p.parse_args()

    pt = _load_paper_trade()
    mzip, vnp = pt.resolve_trade_artifacts(args.run_id, args.model, args.vec_normalize)

    cpath = Path(args.contracts).resolve() if args.contracts else None
    logical_order, contracts = contract_map.load_contracts(cpath)
    quals: list | None = None
    ib = None

    if logical_order != tuple(pt.TICKERS):
        print("ERROR: contract_map order != paper_trade.TICKERS")
        sys.exit(1)

    idx, ohlcv, rsi, macd, macro, fd, fdm = pt.fetch_recent(days=90)
    min_bars = pt.LOOKBACK + pt.OBS_LAG + 1
    if len(idx) < min_bars:
        print(f"ERROR: need >= {min_bars} bars, got {len(idx)}")
        sys.exit(1)
    t = len(idx) - 1
    latest = str(idx[t].date())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    st = _load_state()
    if st.get("last_bar_date") == latest and not args.force and not args.dry_run and not args.no_orders:
        print(f"Already recorded run for yfinance last bar {latest}. Use --force to go again.")
        return

    print(f"IBKR paper  run_id={args.run_id}  last_bar={latest}  utc={now}")
    print(f"  model: {mzip}\n  vec:   {vnp}")

    mkt: dict[str, float] = {k: 0.0 for k in pt.TICKERS}
    nlv: float
    if args.dry_run:
        nlv = float(args.use_fixed_nlv) if args.use_fixed_nlv > 0 else 5_000_000.0
        if args.use_fixed_nlv > 0:
            print(f"  dry-run: using fixed NLV ${nlv:,.0f}")
        else:
            print(f"  dry-run: using default NLV ${nlv:,.0f} (set --use-fixed-nlv to override)")
    else:
        ib = connect_ib(args.ib_host, args.ib_port, args.ib_client_id)
        quals = qualify_legs(ib, contracts)
        mkt = mkt_dollar_by_logical(ib, quals, list(logical_order))
        if args.use_fixed_nlv and args.use_fixed_nlv > 0:
            nlv = float(args.use_fixed_nlv)
            print(
                f"  using --use-fixed-nlv ${nlv:,.0f} (override; IB also used for positions in obs.)"
            )
        else:
            nlv, nlv_src = nlv_usd(ib)
            print(f"  Account equity (NLV) ${nlv:,.2f}  — {nlv_src}")
        if nlv < 1.0 and not (args.use_fixed_nlv and args.use_fixed_nlv > 0):
            print(
                "\n  WARNING: NLV/estimate is ~$0, so target dollar lines will be $0. Check:\n"
                "  - TWS: API → Settings: Read-Only API *off*; reconnect after connect.\n"
                "  - Fund / reset **paper** cash (Client Portal) if the balance should be non-zero.\n"
                "  - Optional override:  --use-fixed-nlv 5000000  (for sizing only, not the obs portfolio slice).\n"
            )
        for k, v in mkt.items():
            if abs(v) > 1.0:
                print(f"  {k} mkt ~ ${v:,.2f}")

    cash, units, nav, _ = _synthetic_cash_units(pt, mkt, ohlcv, t, nlv)
    peak_nav = max(float(st.get("peak_nav", 0.0) or 0.0), nav, 1.0)
    obs = pt.build_obs(
        ohlcv, rsi, macd, macro, fd, fdm,
        t, cash, units, peak_nav, progress=0.5, obs_lag=pt.OBS_LAG,
    )
    if obs.shape != (98,):
        print(f"ERROR: obs shape {obs.shape}, expected 98")
        sys.exit(1)
    m_o, m_v = pt.load_obs_normalizer(vnp)
    obs_n = pt.normalize_obs(obs, m_o, m_v)
    model = RecurrentPPO.load(str(mzip), device="cpu")
    action, _ = model.predict(
        obs_n.reshape(1, -1),
        state=None,
        episode_start=np.ones((1,), dtype=bool),
        deterministic=True,
    )
    act = np.asarray(action).reshape(-1)
    target_w = portfolio_weights_from_action(act.astype(np.float64))
    print("  target weights: CASH " + f"{100*target_w[0]:.1f}%  " + "  ".join(
        f"{pt.TICKERS[i]} {100*target_w[1+i]:.1f}%" for i in range(10)
    ))

    cfg = RebalanceConfig(
        min_leg_dollars=args.min_leg_dollars,
        forex_lot=args.forex_lot,
    )
    drows = dry_run_deltas(nlv, target_w, mkt, list(logical_order), min_leg=cfg.min_leg_dollars)
    for row in drows:
        s = "SKIP" if row["skip"] else "TRADE"
        print(
            f"  [{s}] {row['logical']}: target ${row['target_usd']:,.0f}  "
            f"cur ${row['current_usd']:,.0f}  d ${row['delta_usd']:+,.0f}"
        )

    if args.dry_run:
        print("\n(dry-run: no TWS orders, no state update)")
        return
    if args.no_orders:
        if ib is not None and ib.isConnected():
            ib.disconnect()
        print("\n(--no-orders: connected but did not place orders)")
        return

    assert ib is not None and quals is not None
    try:
        placed = rebalance_toward_targets(
            ib,
            target_w,
            nlv,
            list(logical_order),
            quals,
            config=cfg,
            yf_ticker_by_logical=SIZING_YF,
        )
        for row in placed:
            print("  order:", row)
        st["last_bar_date"] = latest
        st["peak_nav"] = max(peak_nav, nav)
        _save_state(st)
        block = f"=== {latest}  {now}  NLV ~ ${nlv:,.0f} ===\n" + "\n".join(
            str(x) for x in placed
        )
        with open(_JOURNAL, "a", encoding="utf-8") as f:
            f.write(block + "\n\n")
        print(f"State: {_STATE_FILE}  journal: {_JOURNAL}")
    finally:
        if ib is not None and ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
