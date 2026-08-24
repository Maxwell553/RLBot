#!/usr/bin/env python3
"""
CoreEquity — no 2x/3x ETFs; QQQ calm sleeve may cash-finance up to 1.5×.

Weekly close-to-close equity hybrid + month-end dual momentum.

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

from core_equity_env import (  # noqa: E402
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
    Q_CAP_MAX,
    W_CAP,
    ProdParams,
    load_adv_usd,
    nested_metrics,
    passes_raised_holdout_gates,
    passes_raised_selection_gates,
    passes_selection_gates,
    passes_stress_gates,
    run_prod,
    trend_ok,
    week_end_mask,
)
from real_alpha_env import (  # noqa: E402
    _atrp,
    _roll_vol,
    load_ohlc,
    load_panel,
    month_end_mask,
    rets,
    spy_bh_costed,
)

# Locked 2026-08-20 (train+mid search; holdout one-shot).
# QQQ 12m-abs OR SMA(42) re-entry; q_cap 1.4. Raised GE gates cleared.
P = ProdParams(
    ma=252,
    vt=0.30,
    atr_max=0.12,
    es=0.18,
    cool=15,
    w_a=0.58,
    dual_lb=231,
    dual_vt=0.14,
    dual_b="TLT",
    rebal="wk",
    vol_lb=21,
    eq_sym="QQQ",
    w_hot=1.0,
    q_vt=0.30,
    q_atr_max=0.12,
    q_cap=1.4,
    vol_spike=9.0,
    atr_hyst=0.05,
    trend_sym="QQQ",
    trend_mode="abs_or_sma",
    crash_lb=42,
)


def _sleeve_weight(ok, vol, atr, vt, atr_max, w_cap, i, atr_hyst=0.0):
    atr_gate = atr_max * (1.0 + atr_hyst) if atr_hyst > 0.0 else atr_max
    if ok[i] and (atr_gate >= 9 or not np.isfinite(atr[i]) or atr[i] <= atr_gate):
        if np.isfinite(vol[i]) and vol[i] > 1e-8:
            return float(min(w_cap, max(0.0, vt / vol[i])))
    return 0.0


def target_features(px, hot_ohlc, qqq_ohlc, p: ProdParams = P):
    """Causal indicator arrays for latest_targets; compute once for a fill journal."""
    _o, h, l, c = hot_ohlc
    qo, qh, ql, qc = qqq_ohlc
    return {
        "ok": trend_ok(px, p),
        "vol": _roll_vol(rets(c), p.vol_lb),
        "atr": _atrp(h, l, c),
        "qvol": _roll_vol(rets(qc), p.vol_lb),
        "qatr": _atrp(qh, ql, qc),
        "v1": _roll_vol(rets(px["GLD"]), p.vol_lb),
        "v2": _roll_vol(rets(px[p.dual_b]), p.vol_lb),
    }


def latest_targets(
    dates, px, hot_ohlc, qqq_ohlc, p: ProdParams = P, i: int | None = None, features=None
):
    if i is None:
        i = len(dates) - 1
    feat = features if features is not None else target_features(px, hot_ohlc, qqq_ohlc, p)
    ok, vol, atr = feat["ok"], feat["vol"], feat["atr"]
    qvol, qatr = feat["qvol"], feat["qatr"]
    v1, v2 = feat["v1"], feat["v2"]
    hot_is_qqq = str(p.eq_sym).upper() == "QQQ"
    q_cap = float(min(Q_CAP_MAX, max(0.0, getattr(p, "q_cap", 1.0))))
    hot_cap = q_cap if hot_is_qqq else W_CAP
    w_h = _sleeve_weight(ok, vol, atr, p.vt, p.atr_max, hot_cap, i, p.atr_hyst)
    w_q = _sleeve_weight(ok, qvol, qatr, p.q_vt, p.q_atr_max, q_cap, i, p.atr_hyst)

    gld, alt = px["GLD"], px[p.dual_b]
    dual_sym, dual_w = "BIL", 0.0
    if i >= p.dual_lb and gld[i] > 0 and gld[i - p.dual_lb] > 0 and alt[i] > 0 and alt[i - p.dual_lb] > 0:
        tr1 = gld[i] / gld[i - p.dual_lb] - 1.0
        tr2 = alt[i] / alt[i - p.dual_lb] - 1.0
        if tr1 > 0 or tr2 > 0:
            if tr1 >= tr2 and np.isfinite(v1[i]) and v1[i] > 1e-8:
                dual_sym, dual_w = "GLD", float(min(W_CAP, p.dual_vt / v1[i]))
            elif np.isfinite(v2[i]) and v2[i] > 1e-8:
                dual_sym, dual_w = p.dual_b, float(min(W_CAP, p.dual_vt / v2[i]))

    w_hot = float(min(1.0, max(0.0, p.w_hot)))
    if hot_is_qqq:
        w_hot = 1.0
        port_hot = p.w_a * w_h
        port_qqq = port_hot
    else:
        port_hot = p.w_a * w_hot * w_h
        port_qqq = p.w_a * (1.0 - w_hot) * w_q
    return {
        "asof": str(dates[i]),
        "rebalance_mode": f"weekly close-to-close ({p.eq_sym}+QQQ hybrid) + month-end (dual)",
        "leverage": f"no 2x/3x ETFs; QQQ cap {q_cap:.2f}x cash-financed; hot/dual cap 1.0",
        "sleeve_A_capital": p.w_a,
        "sleeve_A_hot_symbol": p.eq_sym,
        "sleeve_A_hot_share": w_hot,
        "sleeve_A_QQQ_share": 0.0 if hot_is_qqq else 1.0 - w_hot,
        "hot_cc_weight": w_h,
        "QQQ_cc_weight": w_h if hot_is_qqq else w_q,
        f"portfolio_{p.eq_sym}": port_hot,
        "portfolio_QQQ": port_qqq,
        "sleeve_B_capital": 1.0 - p.w_a,
        "dual_asset": dual_sym,
        "dual_weight": dual_w,
        "portfolio_dual": (1.0 - p.w_a) * dual_w,
        "friction": "retail + 3bps impact per 1% ADV",
        "params": asdict(p),
    }


def paper_plan(dates, px, hot_ohlc, qqq_ohlc, p: ProdParams = P, aum: float = START_EQ):
    t = latest_targets(dates, px, hot_ohlc, qqq_ohlc, p)
    wk = week_end_mask(dates)
    me = month_end_mask(dates)
    i = len(dates) - 1
    equity_rebal_today = bool(wk[i])
    dual_rebal_today = bool(me[i])
    fr = PROD_FRICTION
    hot = p.eq_sym
    w_hot_port = float(t[f"portfolio_{hot}"])
    w_qqq = float(t["portfolio_QQQ"])
    w_dual = float(t["portfolio_dual"]) if t["dual_asset"] != "BIL" else 0.0
    qqq_cash = 0.0 if str(hot).upper() == "QQQ" else min(w_qqq, p.w_a * (1.0 - p.w_hot))
    w_bil = max(0.0, 1.0 - w_hot_port - qqq_cash - w_dual)
    plan = {
        "asof": t["asof"],
        "aum": aum,
        "equity_rebalance_due": equity_rebal_today,
        "dual_rebalance_due": dual_rebal_today,
        "orders": [],
        "fee_schedule_one_way": {
            "liquid_bps": fr.liquid * 1e4,
            "BIL_bps": fr.bil * 1e4,
            "impact_bps_per_1pct_ADV": IMPACT_PER_UNIT_ADV * 1e4 / 100.0,
        },
        "targets": t,
    }
    if equity_rebal_today:
        legs = [{"symbol": hot, "target_weight": w_hot_port}]
        if str(hot).upper() != "QQQ":
            legs.append({"symbol": "QQQ", "target_weight": w_qqq})
        plan["orders"].append({"sleeve": "A", "when": "weekly close", "legs": legs})
    if dual_rebal_today:
        plan["orders"].append(
            {
                "sleeve": "B",
                "when": "month-end close",
                "legs": [
                    {
                        "symbol": t["dual_asset"] if t["dual_asset"] != "BIL" else "BIL",
                        "target_weight": w_dual,
                    }
                ],
            }
        )
    plan["portfolio_targets"] = {
        hot: w_hot_port,
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

    dates, px = load_panel(["SPY", "QQQ", "BIL", "GLD", P.dual_b, P.eq_sym])
    hot_ohlc = load_ohlc(P.eq_sym, dates)
    qqq_ohlc = load_ohlc("QQQ", dates)
    adv_hot = load_adv_usd(P.eq_sym, dates)
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
            dates, px, hot_ohlc, adv_hot, adv_liquid, P,
            fr=PROD_FRICTION, aum=args.aum, qqq_ohlc=qqq_ohlc, adv_qqq=adv_qqq,
        )
        m = nested_metrics(eq, dates, spy_eq)
        eq_s, _ = run_prod(
            dates, px, hot_ohlc, adv_hot, adv_liquid, P,
            fr=STRESS_FRICTION, aum=args.aum, qqq_ohlc=qqq_ohlc, adv_qqq=adv_qqq,
        )
        m_s = nested_metrics(eq_s, dates, spy_eq)
        eq_2, _ = run_prod(
            dates, px, hot_ohlc, adv_hot, adv_liquid, P,
            fr=STRESS2_FRICTION, aum=args.aum, qqq_ohlc=qqq_ohlc, adv_qqq=adv_qqq,
        )
        m_2 = nested_metrics(eq_2, dates, spy_eq)

        print("=== CoreEquity (no 3x ETFs; QQQ cap ≤1.5; abs OR SMA42; retail + ADV impact) ===")
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
        t = latest_targets(dates, px, hot_ohlc, qqq_ohlc, P)
        print("\n=== next-session targets (lag-1) ===")
        for k, v in t.items():
            if k == "params":
                continue
            print(f"  {k}: {v}")

    if do_pp:
        plan = paper_plan(dates, px, hot_ohlc, qqq_ohlc, P, aum=args.aum)
        print("\n=== paper plan (exact book) ===")
        print(json.dumps(plan, indent=2, default=str))


if __name__ == "__main__":
    main()
