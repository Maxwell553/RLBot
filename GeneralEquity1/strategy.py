#!/usr/bin/env python3
"""
prod_return_alpha_v3 — stronger mid cushion + plateau under frozen prod_alpha_env.

Env (unchanged): retail + ADV impact, nested one-shot holdout, turnover ≤20×, stress×1.5.
Selection (raised): mid Sharpe ≥ 1.05, full+mid DD ≤ 15%, stress×2, plateau mid≥1.04 ≥25%.

Book vs v2 (1070pct):
  - Same w_a=0.58, dual GLD/TLT 231d VT 14%
  - Sleeve A hybrid: 78% TQQQ + 22% QQQ (was 80/20) — more QQQ share for mid
  - ATR hysteresis 5% (exit ATR = atr_max*1.05) — binds churn without trail-stop mid destruction
  - Objective: mid_cushion × plateau × capacity (not return%)

Usage:
  python strategy.py --backtest
  python strategy.py --targets
  python strategy.py --paper-plan
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parent
sys.path.insert(0, str(PACK))

from prod_alpha_env import (  # noqa: E402
    HOLD_END,
    HOLD_START,
    IMPACT_PER_UNIT_ADV,
    MID_END,
    MID_START,
    PROD_FRICTION,
    START_EQ,
    STRESS2_FRICTION,
    STRESS_FRICTION,
    TRAIN_END,
    ProdParams,
    load_adv_usd,
    nested_metrics,
    passes_raised_holdout_gates,
    passes_raised_selection_gates,
    passes_selection_gates,
    passes_stress_gates,
    run_prod,
    week_end_mask,
)
from real_alpha_env import (  # noqa: E402
    _atrp,
    _roll_vol,
    _sma_ok,
    load_ohlc,
    load_panel,
    month_end_mask,
    rets,
    spy_bh_costed,
)

# Locked 2026-08-01 (train+mid search; holdout one-shot)
P = ProdParams(
    ma=151,
    vt=0.27,
    atr_max=0.10,
    es=0.278,
    cool=15,
    w_a=0.58,
    dual_lb=231,
    dual_vt=0.14,
    dual_b="TLT",
    rebal="wk",
    vol_lb=21,
    w_tqqq=0.78,
    q_vt=0.08,
    q_cap=1.5,
    q_atr_max=0.10,
    vol_spike=9.0,
    atr_hyst=0.05,
)


def _sleeve_weight(ok, vol, atr, vt, atr_max, w_cap, i, atr_hyst=0.0):
    """Point-in-time weight. With atr_hyst>0 use exit ATR (= atr_max*(1+hyst)) so
    targets match the research hold band (no false flat when ATR is between on/off)."""
    atr_gate = atr_max * (1.0 + atr_hyst) if atr_hyst > 0.0 else atr_max
    if ok[i] and (atr_gate >= 9 or not np.isfinite(atr[i]) or atr[i] <= atr_gate):
        if np.isfinite(vol[i]) and vol[i] > 1e-8:
            return float(min(w_cap, max(0.0, vt / vol[i])))
    return 0.0


def latest_targets(dates, px, tqqq_ohlc, qqq_ohlc, p: ProdParams = P):
    i = len(dates) - 1
    _o, h, l, c = tqqq_ohlc
    vol = _roll_vol(rets(c), p.vol_lb)
    atr = _atrp(h, l, c)
    ok = _sma_ok(px["QQQ"], p.ma)
    w_t = _sleeve_weight(ok, vol, atr, p.vt, p.atr_max, 1.0, i, p.atr_hyst)

    qo, qh, ql, qc = qqq_ohlc
    qvol = _roll_vol(rets(qc), p.vol_lb)
    qatr = _atrp(qh, ql, qc)
    w_q = _sleeve_weight(ok, qvol, qatr, p.q_vt, p.q_atr_max, p.q_cap, i, p.atr_hyst)

    gld, alt = px["GLD"], px[p.dual_b]
    v1 = _roll_vol(rets(gld), p.vol_lb)
    v2 = _roll_vol(rets(alt), p.vol_lb)
    dual_sym, dual_w = "BIL", 0.0
    if i >= p.dual_lb and gld[i] > 0 and gld[i - p.dual_lb] > 0 and alt[i] > 0 and alt[i - p.dual_lb] > 0:
        tr1 = gld[i] / gld[i - p.dual_lb] - 1.0
        tr2 = alt[i] / alt[i - p.dual_lb] - 1.0
        if tr1 > 0 or tr2 > 0:
            if tr1 >= tr2 and np.isfinite(v1[i]) and v1[i] > 1e-8:
                dual_sym, dual_w = "GLD", float(min(1.0, p.dual_vt / v1[i]))
            elif np.isfinite(v2[i]) and v2[i] > 1e-8:
                dual_sym, dual_w = p.dual_b, float(min(1.0, p.dual_vt / v2[i]))

    return {
        "asof": str(dates[i]),
        "rebalance_mode": "weekly close-to-close (TQQQ+QQQ hybrid) + month-end (dual)",
        "sleeve_A_capital": p.w_a,
        "sleeve_A_TQQQ_share": p.w_tqqq,
        "sleeve_A_QQQ_share": 1.0 - p.w_tqqq,
        "TQQQ_cc_weight": w_t,
        "QQQ_cc_weight": w_q,
        "portfolio_TQQQ": p.w_a * p.w_tqqq * w_t,
        "portfolio_QQQ": p.w_a * (1.0 - p.w_tqqq) * w_q,
        "sleeve_B_capital": 1.0 - p.w_a,
        "dual_asset": dual_sym,
        "dual_weight": dual_w,
        "portfolio_dual": (1.0 - p.w_a) * dual_w,
        "friction": "retail + 3bps impact per 1% ADV",
        "params": asdict(p),
    }


def paper_plan(dates, px, tqqq_ohlc, qqq_ohlc, p: ProdParams = P, aum: float = START_EQ):
    """Exact executable plan: weekly equity CC + month-end dual, with fee/slippage log rates."""
    t = latest_targets(dates, px, tqqq_ohlc, qqq_ohlc, p)
    wk = week_end_mask(dates)
    me = month_end_mask(dates)
    i = len(dates) - 1
    # next action flags from last completed session
    equity_rebal_today = bool(wk[i])
    dual_rebal_today = bool(me[i])
    fr = PROD_FRICTION
    plan = {
        "asof": t["asof"],
        "aum": aum,
        "equity_rebalance_due": equity_rebal_today,
        "dual_rebalance_due": dual_rebal_today,
        "orders": [],
        "fee_schedule_one_way": {
            "TQQQ_bps": fr.tqqq * 1e4,
            "QQQ_GLD_TLT_bps": fr.liquid * 1e4,
            "BIL_bps": fr.bil * 1e4,
            "impact_bps_per_1pct_ADV": IMPACT_PER_UNIT_ADV * 1e4 / 100.0,
        },
        "targets": t,
    }
    # Portfolio target weights (QQQ may be cash-financed above its sleeve cash slot).
    w_tqqq = float(t["portfolio_TQQQ"])
    w_qqq = float(t["portfolio_QQQ"])
    w_dual = float(t["portfolio_dual"]) if t["dual_asset"] != "BIL" else 0.0
    w_bil = max(0.0, 1.0 - w_tqqq - min(w_qqq, p.w_a * (1.0 - p.w_tqqq)) - w_dual)
    if equity_rebal_today:
        plan["orders"].append(
            {
                "sleeve": "A",
                "when": "weekly close",
                "legs": [
                    {"symbol": "TQQQ", "target_weight": w_tqqq},
                    {"symbol": "QQQ", "target_weight": w_qqq},
                ],
            }
        )
    if dual_rebal_today:
        plan["orders"].append(
            {
                "sleeve": "B",
                "when": "month-end close",
                "legs": [
                    {"symbol": t["dual_asset"] if t["dual_asset"] != "BIL" else "BIL", "target_weight": w_dual},
                ],
            }
        )
    plan["portfolio_targets"] = {
        "TQQQ": w_tqqq,
        "QQQ": w_qqq,
        t["dual_asset"] if t["dual_asset"] != "BIL" else "dual_BIL": w_dual,
        "BIL_residual_approx": w_bil,
    }
    if not plan["orders"]:
        plan["orders"].append({"note": "hold prior targets; no weekly/month-end rebalance today"})
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--paper-plan", action="store_true")
    ap.add_argument("--aum", type=float, default=START_EQ)
    args = ap.parse_args()
    do_bt = args.backtest or (not args.targets and not args.paper_plan)
    do_tg = args.targets or (not args.backtest and not args.paper_plan)
    do_pp = args.paper_plan

    dates, px = load_panel(["SPY", "QQQ", "TQQQ", "BIL", "GLD", P.dual_b])
    tqqq_ohlc = load_ohlc("TQQQ", dates)
    qqq_ohlc = load_ohlc("QQQ", dates)
    adv_tqqq = load_adv_usd("TQQQ", dates)
    adv_qqq = load_adv_usd("QQQ", dates)
    adv_gld = load_adv_usd("GLD", dates)
    adv_alt = load_adv_usd(P.dual_b, dates)
    adv_liquid = np.minimum(
        np.where(np.isfinite(adv_gld), adv_gld, 1e18),
        np.where(np.isfinite(adv_alt), adv_alt, 1e18),
    )
    spy_eq = spy_bh_costed(px["SPY"].astype(np.float64), dates, PROD_FRICTION.liquid * 2)

    if do_bt:
        eq, meta = run_prod(
            dates,
            px,
            tqqq_ohlc,
            adv_tqqq,
            adv_liquid,
            P,
            fr=PROD_FRICTION,
            aum=args.aum,
            qqq_ohlc=qqq_ohlc,
            adv_qqq=adv_qqq,
        )
        m = nested_metrics(eq, dates, spy_eq)
        eq_s, _ = run_prod(
            dates,
            px,
            tqqq_ohlc,
            adv_tqqq,
            adv_liquid,
            P,
            fr=STRESS_FRICTION,
            aum=args.aum,
            qqq_ohlc=qqq_ohlc,
            adv_qqq=adv_qqq,
        )
        m_s = nested_metrics(eq_s, dates, spy_eq)
        eq_2, _ = run_prod(
            dates,
            px,
            tqqq_ohlc,
            adv_tqqq,
            adv_liquid,
            P,
            fr=STRESS2_FRICTION,
            aum=args.aum,
            qqq_ohlc=qqq_ohlc,
            adv_qqq=adv_qqq,
        )
        m_2 = nested_metrics(eq_2, dates, spy_eq)

        print("=== prod_return_alpha_v3 (retail + ADV impact, weekly CC hybrid) ===")
        print(f"params: {json.dumps(asdict(P))}")
        print(
            f"turnover ann one-way: {meta['ann_one_way_turnover']:.2f}×  "
            f"(sleeve-A {meta['turn_a']:.2f}×)"
        )
        for lab, mm in [("retail", m), ("stress×1.5", m_s), ("stress×2.0", m_2)]:
            print(f"\n-- {lab} --")
            for k in ("train", "mid", "hold", "full"):
                x = mm[k]
                spy = mm[f"spy_{k}"]
                print(
                    f"  {k:5s} ret={x['total_return']:+.1%} sh={x['sharpe_ratio']:.3f} "
                    f"dd={x['max_drawdown']:.1%} | SPY {spy['total_return']:+.1%} "
                    f"beat={x['total_return']>spy['total_return']}"
                )
        print(
            f"\ngates: frozen_select={passes_selection_gates(m, meta['ann_one_way_turnover'])} "
            f"raised_select={passes_raised_selection_gates(m, meta['ann_one_way_turnover'])} "
            f"raised_holdout={passes_raised_holdout_gates(m)} "
            f"stress15={passes_stress_gates(m_s)} stress2={passes_stress_gates(m_2)}"
        )
        print(
            f"windows: train≤{TRAIN_END} mid {MID_START}→{MID_END} "
            f"holdout {HOLD_START}→{HOLD_END}"
        )

    if do_tg:
        t = latest_targets(dates, px, tqqq_ohlc, qqq_ohlc, P)
        print("\n=== next-session targets (lag-1) ===")
        for k, v in t.items():
            if k == "params":
                continue
            print(f"  {k}: {v}")

    if do_pp:
        plan = paper_plan(dates, px, tqqq_ohlc, qqq_ohlc, P, aum=args.aum)
        print("\n=== paper plan (exact book) ===")
        print(json.dumps(plan, indent=2, default=str))


if __name__ == "__main__":
    main()
