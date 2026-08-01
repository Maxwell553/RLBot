#!/usr/bin/env python3
"""
prod_return_alpha_v1 — return-alpha under locked knobs (1360pctAlgo pack).

Mandate: beat costed SPY with Sharpe ≥ 1.0 under retail costs.
Book: TQQQ close-to-close (weekly) + GLD/TLT dual momentum (month-end).

Live / paper path (MarketTrainer forward test)::

    python scripts/paper_prod_return_alpha.py run-day --refresh-data

Research CLI (this file)::

    python 1360pctAlgo/strategy.py --targets
    python 1360pctAlgo/strategy.py --backtest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlbot.prod_return_alpha import (  # noqa: E402
    P,
    PAPER_RUN_ID,
    STRATEGY_ID,
    _atrp,
    _rets,
    _roll_vol,
    _sma_ok,
    compute_target_weights,
    fetch_daily_ohlc,
    month_end_mask,
    portfolio_weights_from_sleeves,
    week_end_mask,
)


def _gross_sanity_backtest(force_refresh: bool = False) -> None:
    """Frictionless lag-1 path (not the retail+impact research numbers)."""
    dates, closes, ohlc = fetch_daily_ohlc(force_refresh=force_refresh)
    n = len(dates)
    _o, h, l, c = ohlc["TQQQ"]
    vol = _roll_vol(_rets(c), P.vol_lb)
    atr = _atrp(h, l, c)
    ok = _sma_ok(closes["QQQ"], P.ma)
    v_gld = _roll_vol(_rets(closes["GLD"]), P.vol_lb)
    v_alt = _roll_vol(_rets(closes[P.dual_b]), P.vol_lb)
    wk = week_end_mask(dates)
    me = month_end_mask(dates)

    tqqq_w = 0.0
    dual_sym, dual_w = "BIL", 0.0
    flat = False
    cool = 0
    eq = 100_000.0
    peak = eq
    nav = np.empty(n, dtype=np.float64)
    nav[0] = eq

    for i in range(1, n):
        j = i - 1
        # Rebalance sleeves on schedule using lag-1 info at j.
        if (not flat) and (wk[j] or i == 1):
            tqqq_w = 0.0
            if ok[j] and (P.atr_max >= 9 or not np.isfinite(atr[j]) or atr[j] <= P.atr_max):
                if np.isfinite(vol[j]) and vol[j] > 1e-8:
                    tqqq_w = float(min(1.0, P.vt / vol[j]))
        if flat:
            tqqq_w = 0.0
        if me[j] or i == 1:
            dual_sym, dual_w = "BIL", 0.0
            gld, alt = closes["GLD"], closes[P.dual_b]
            if j >= P.dual_lb and gld[j] > 0 and gld[j - P.dual_lb] > 0 and alt[j] > 0 and alt[j - P.dual_lb] > 0:
                tr1 = gld[j] / gld[j - P.dual_lb] - 1.0
                tr2 = alt[j] / alt[j - P.dual_lb] - 1.0
                if tr1 > 0 or tr2 > 0:
                    if tr1 >= tr2 and np.isfinite(v_gld[j]) and v_gld[j] > 1e-8:
                        dual_sym, dual_w = "GLD", float(min(1.0, P.dual_vt / v_gld[j]))
                    elif np.isfinite(v_alt[j]) and v_alt[j] > 1e-8:
                        dual_sym, dual_w = P.dual_b, float(min(1.0, P.dual_vt / v_alt[j]))

        w = portfolio_weights_from_sleeves(
            tqqq_w=tqqq_w, dual_asset=dual_sym, dual_w=dual_w, p=P
        )
        r = 0.0
        for sym, wt in w.items():
            px = closes[sym]
            if px[i - 1] > 0 and np.isfinite(px[i]) and np.isfinite(px[i - 1]):
                r += float(wt) * (px[i] / px[i - 1] - 1.0)
        eq *= 1.0 + r
        if eq > peak:
            peak = eq
        if (not flat) and peak > 0 and (eq / peak - 1.0) <= -P.es:
            flat = True
            cool = P.cool
            tqqq_w = 0.0
        elif flat:
            cool -= 1
            if cool <= 0:
                flat = False
                peak = eq
        nav[i] = eq

    rets = np.diff(nav) / nav[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) and rets.std() > 0 else 0.0
    dd = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
    print(f"\n=== {STRATEGY_ID} gross sanity backtest (yfinance, lag-1) ===")
    print(
        f"  bars={n}  total_return={nav[-1] / nav[0] - 1:+.1%}  "
        f"sharpe={sharpe:.2f}  max_dd={dd:.1%}"
    )
    print("  note: nested retail+impact metrics are from the locked research pack;")
    print("  this path is a frictionless sanity check for the live signal.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", action="store_true", help="Print next-session sleeve + portfolio weights")
    ap.add_argument("--backtest", action="store_true", help="Gross sanity NAV path (no retail impact)")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()
    do_tg = args.targets or not args.backtest
    do_bt = args.backtest

    if do_tg:
        w, meta = compute_target_weights(force_refresh=bool(args.refresh_data))
        print(f"=== {STRATEGY_ID} targets (run_id={PAPER_RUN_ID}) ===")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print("  portfolio_weights:")
        for k, v in sorted(w.items(), key=lambda kv: -float(kv[1])):
            print(f"    {k:6s} {float(v):6.2%}")

    if do_bt:
        _gross_sanity_backtest(force_refresh=bool(args.refresh_data))


if __name__ == "__main__":
    main()
