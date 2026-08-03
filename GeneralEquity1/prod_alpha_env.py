#!/usr/bin/env python3
"""
Production research environment (frozen).

Mandate: return-alpha — beat costed SPY under retail (default) / stress×1.5.
Book: close-to-close only (no overnight/day split reshape).
Costs: RETAIL as default. Size-aware impact on top.
Nested windows: tune on train, validate on mid, one-shot holdout.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numba import njit

import sys

PACK = Path(__file__).resolve().parent
sys.path.insert(0, str(PACK))

from real_alpha_env import (  # noqa: E402
    Friction,
    RETAIL_FRICTION,
    START_EQ,
    blend_eq,
    dual_mom_daily,
    friction_scaled,
    load_ohlc,
    load_panel,
    metrics_uncapped,
    month_end_mask,
    rets,
    spy_bh_costed,
    tqqq_cc_daily,
    trend_sleeve_daily,
    window_slice,
)
from real_alpha_env import _atrp, _roll_vol, _sma_ok  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen prod gates
# ---------------------------------------------------------------------------
PROD_FRICTION = RETAIL_FRICTION  # default bar
STRESS_FRICTION = friction_scaled(RETAIL_FRICTION, 1.5)

BT_START = date(2010, 1, 4)
TRAIN_END = date(2017, 12, 29)
MID_START = date(2018, 1, 2)
MID_END = date(2023, 12, 29)
HOLD_START = date(2024, 1, 2)  # never in objective / selection
HOLD_END = date(2026, 7, 29)
BT_END = HOLD_END

# Impact: extra slip = participation * IMPACT_PER_UNIT_ADV
# with participation = trade$ / ADV$; 0.03 ⇒ 3 bps per 1% ADV.
IMPACT_PER_UNIT_ADV = 0.03
MAX_ANN_ONE_WAY_TURNOVER = 20.0  # hard gate
TARGET_ANN_ONE_WAY_TURNOVER = 10.0


@dataclass(frozen=True)
class ProdParams:
    ma: int = 151
    vt: float = 0.27
    atr_max: float = 0.20
    es: float = 0.278
    cool: int = 15
    w_a: float = 0.57
    dual_lb: int = 231
    dual_vt: float = 0.14
    dual_b: str = "TLT"
    rebal: str = "wk"  # 'wk' or 'me' — single rebalance mode
    vol_lb: int = 21
    # Hybrid sleeve A (defaults preserve v1 TQQQ-only book):
    # fraction of sleeve-A capital in TQQQ; remainder = QQQ vol-target (capped).
    w_tqqq: float = 1.0
    q_vt: float = 0.12
    q_cap: float = 1.5
    q_atr_max: float = 0.10
    # Risk gates (9.0 / 0.0 = disabled; keep es wide unless intentionally binding):
    vol_spike: float = 9.0  # flat if vol > vol_spike * 63d avg vol
    atr_hyst: float = 0.0  # exit ATR = atr_max*(1+atr_hyst); reduces on/off churn


def week_end_mask(dates) -> np.ndarray:
    n = len(dates)
    m = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if dates[i].isocalendar()[1] != dates[i - 1].isocalendar()[1]:
            m[i - 1] = True
    m[-1] = True
    return m


def load_adv_usd(sym: str, dates, db: Path | None = None) -> np.ndarray:
    """21-day ADV in USD (volume * close), aligned to dates."""
    if db is None:
        db = PACK / "data" / "bars.db"
    idx = {d: i for i, d in enumerate(dates)}
    n = len(dates)
    vol = np.full(n, np.nan)
    close = np.full(n, np.nan)
    con = sqlite3.connect(db)
    for ts, v, c in con.execute(
        "SELECT timestamp, volume, close FROM bars WHERE symbol=?", (sym,)
    ):
        d = date.fromisoformat(ts[:10])
        if d in idx:
            i = idx[d]
            vol[i] = float(v) if v is not None else np.nan
            close[i] = float(c) if c is not None else np.nan
    con.close()
    dollar = vol * close
    adv = np.full(n, np.nan)
    for i in range(20, n):
        w = dollar[i - 20 : i + 1]
        if np.all(np.isfinite(w)) and np.all(w > 0):
            adv[i] = float(np.mean(w))
    # forward-fill early NaNs with first valid
    first = next((i for i in range(n) if np.isfinite(adv[i])), None)
    if first is not None:
        adv[:first] = adv[first]
        for i in range(first + 1, n):
            if not np.isfinite(adv[i]):
                adv[i] = adv[i - 1]
    return adv


@njit
def _apply_impact_on_turnover(gross_r, turn_w, adv, equity, aum0, cost_base, impact_coef):
    """
    gross_r: sleeve daily gross return before costs
    turn_w: one-way weight turnover that day (|Δw| of the traded sleeve book)
    Charge: turn_w * (cost_base + impact), impact = impact_coef * (turn_w * equity / adv)
    """
    n = len(gross_r)
    out = np.zeros(n)
    eq = aum0
    for i in range(n):
        if not np.isfinite(gross_r[i]):
            out[i] = 0.0
            continue
        tw = turn_w[i] if np.isfinite(turn_w[i]) else 0.0
        extra = 0.0
        if tw > 0.0 and np.isfinite(adv[i]) and adv[i] > 1.0 and eq > 0.0:
            participation = (tw * eq) / adv[i]
            extra = impact_coef * participation
        cost = tw * (cost_base + extra)
        r = gross_r[i] - cost
        if not np.isfinite(r):
            r = 0.0
        out[i] = r
        eq *= 1.0 + r
    return out


@njit
def tqqq_cc_with_turnover(
    ok,
    vol21,
    atrp,
    asset_r,
    bil,
    i0,
    i1,
    vt,
    atr_max,
    es,
    cool,
    rebal_mask,
    w_cap=1.0,
    vol_spike=9.0,
    atr_hyst=0.0,
):
    """Close-to-close sleeve; returns (daily_gross_before_asset_cost, one_way_turn_w, weight).

    w_cap defaults to 1.0 (v1 TQQQ). Values >1 allow capped QQQ leverage via
    cash-financed excess: r = r_asset + (w-1)*(r_asset - r_bil).
    vol_spike < 8: skip entry if vol > vol_spike * 63d average vol.
    atr_hyst > 0: once long, allow ATR up to atr_max*(1+atr_hyst) before exit.
    """
    n = len(bil)
    daily = np.zeros(n)
    turn = np.zeros(n)
    w_path = np.zeros(n)
    # causal 63d average vol for spike gate
    vol_avg = np.empty(n)
    vol_avg[:] = np.nan
    s = 0.0
    cnt = 0
    for i in range(n):
        if np.isfinite(vol21[i]):
            s += vol21[i]
            cnt += 1
            if i >= 63 and np.isfinite(vol21[i - 63]):
                s -= vol21[i - 63]
                cnt -= 1
            if cnt > 0 and i >= 62:
                vol_avg[i] = s / cnt
    eq = 1.0
    peak = 1.0
    flat = False
    cd = 0
    prev_w = 0.0
    w = 0.0
    atr_exit = atr_max * (1.0 + atr_hyst) if atr_hyst > 0.0 else atr_max
    for i in range(i0 + 1, i1 + 1):
        j = i - 1
        do_rebal = rebal_mask[j] or (i == i0 + 1)
        if flat:
            w = 0.0
        elif do_rebal:
            # hysteresis: if already long, use looser ATR exit; else atr_max entry
            was_long = prev_w > 1e-8
            atr_lim = atr_exit if was_long else atr_max
            w = 0.0
            spike = False
            if (
                vol_spike < 8.0
                and np.isfinite(vol21[j])
                and np.isfinite(vol_avg[j])
                and vol_avg[j] > 1e-8
            ):
                if vol21[j] > vol_spike * vol_avg[j]:
                    spike = True
            if (
                (not spike)
                and ok[j]
                and (atr_max >= 9.0 or (not np.isfinite(atrp[j])) or atrp[j] <= atr_lim)
            ):
                if np.isfinite(vol21[j]) and vol21[j] > 1e-8:
                    ww = vt / vol21[j]
                    if ww > w_cap:
                        ww = w_cap
                    if ww < 0.0:
                        ww = 0.0
                    w = ww
        if w <= 1.0:
            g = w * asset_r[i] + (1.0 - w) * bil[i]
        else:
            g = asset_r[i] + (w - 1.0) * (asset_r[i] - bil[i])
        daily[i] = g
        turn[i] = abs(w - prev_w)
        w_path[i] = w
        if not np.isfinite(g):
            g = 0.0
        eq *= 1.0 + g
        if eq > peak:
            peak = eq
        if (not flat) and eq / peak - 1.0 <= -es:
            flat = True
            cd = cool
            w = 0.0
        elif flat:
            if cd > 0:
                cd -= 1
            elif ok[i]:
                flat = False
                peak = eq
        prev_w = w
    return daily, turn, w_path


@njit
def dual_mom_with_turnover(c1, r1, c2, r2, bil, i0, i1, lb, vt, rebal_mask):
    n = len(bil)
    daily = np.zeros(n)
    turn = np.zeros(n)  # liquid-asset one-way (|Δw1|+|Δw2|)
    turn_bil = np.zeros(n)
    vol1 = _roll_vol(r1, 21)
    vol2 = _roll_vol(r2, 21)
    w1 = 0.0
    w2 = 0.0
    wb = 1.0
    prev_w1 = 0.0
    prev_w2 = 0.0
    prev_wb = 1.0
    for i in range(i0 + 1, i1 + 1):
        j = i - 1
        if rebal_mask[j] or (i == i0 + 1):
            w1 = 0.0
            w2 = 0.0
            if j >= lb and c1[j] > 0 and c1[j - lb] > 0 and c2[j] > 0 and c2[j - lb] > 0:
                tr1 = c1[j] / c1[j - lb] - 1.0
                tr2 = c2[j] / c2[j - lb] - 1.0
                if tr1 > 0.0 or tr2 > 0.0:
                    if tr1 >= tr2 and np.isfinite(vol1[j]) and vol1[j] > 1e-8:
                        w1 = min(1.0, vt / vol1[j])
                    elif np.isfinite(vol2[j]) and vol2[j] > 1e-8:
                        w2 = min(1.0, vt / vol2[j])
            wb = 1.0 - w1 - w2
        g = w1 * r1[i] + w2 * r2[i] + wb * bil[i]
        daily[i] = g
        turn[i] = abs(w1 - prev_w1) + abs(w2 - prev_w2)
        turn_bil[i] = abs(wb - prev_wb)
        prev_w1, prev_w2, prev_wb = w1, w2, wb
    return daily, turn, turn_bil


def ann_turnover(turn_w: np.ndarray, i0: int, i1: int) -> float:
    """Annualized one-way turnover from daily |Δw| series."""
    years = max((i1 - i0) / 252.0, 1e-9)
    return float(np.nansum(turn_w[i0 + 1 : i1 + 1]) / years)


def run_prod(
    dates,
    px,
    tqqq_ohlc,
    adv_tqqq,
    adv_liquid,
    p: ProdParams,
    fr: Friction = PROD_FRICTION,
    aum: float = START_EQ,
    impact_coef: float = IMPACT_PER_UNIT_ADV,
    qqq_ohlc=None,
    adv_qqq=None,
):
    """
    Tradeable book: sleeve A = TQQQ CC (optionally hybrid with QQQ VT);
    sleeve B = GLD/dual_b dual mom month-end.
    Single rebalance mode (wk or me) for sleeve A; dual always month-end.
    Costs = retail (or passed fr) + size-aware impact.

    Friction / windows / turnover cap remain frozen. Hybrid fields on ProdParams
    default to v1 TQQQ-only (w_tqqq=1.0).
    """
    n = len(dates)
    i0, i1 = window_slice(dates, BT_START, BT_END)
    bil = rets(px["BIL"])
    me = month_end_mask(dates)
    mask = week_end_mask(dates) if p.rebal == "wk" else me
    ok = _sma_ok(px["QQQ"], p.ma)

    o, h, l, c = tqqq_ohlc
    vol = _roll_vol(rets(c), p.vol_lb)
    atr = _atrp(h, l, c)
    tr = rets(c)
    g_t, turn_t, _ = tqqq_cc_with_turnover(
        ok,
        vol,
        atr,
        tr,
        bil,
        i0,
        i1,
        p.vt,
        p.atr_max,
        p.es,
        p.cool,
        mask,
        1.0,
        float(p.vol_spike),
        float(p.atr_hyst),
    )

    w_t = float(min(1.0, max(0.0, p.w_tqqq)))
    use_qqq = w_t < 1.0 - 1e-12
    if use_qqq:
        if qqq_ohlc is None:
            qqq_ohlc = load_ohlc("QQQ", dates)
        if adv_qqq is None:
            adv_qqq = load_adv_usd("QQQ", dates)
        qo, qh, ql, qc = qqq_ohlc
        qvol = _roll_vol(rets(qc), p.vol_lb)
        qatr = _atrp(qh, ql, qc)
        qr = rets(qc)
        g_q, turn_q, _ = tqqq_cc_with_turnover(
            ok,
            qvol,
            qatr,
            qr,
            bil,
            i0,
            i1,
            p.q_vt,
            p.q_atr_max,
            p.es,
            p.cool,
            mask,
            float(p.q_cap),
            float(p.vol_spike),
            float(p.atr_hyst),
        )
    else:
        g_q = np.zeros(n)
        turn_q = np.zeros(n)
        adv_qqq = adv_tqqq

    # cost each A sub-sleeve on its own capital share, then blend inside A
    net_t = np.zeros(n)
    net_q = np.zeros(n)
    for i in range(n):
        tw = turn_t[i]
        extra = 0.0
        if tw > 0 and np.isfinite(adv_tqqq[i]) and adv_tqqq[i] > 1 and aum > 0:
            participation = (tw * (aum * p.w_a * w_t)) / adv_tqqq[i]
            extra = impact_coef * participation
        cost = tw * (fr.tqqq + extra) + tw * fr.bil
        r = g_t[i] - cost
        net_t[i] = 0.0 if not np.isfinite(r) else r

        if use_qqq:
            twq = turn_q[i]
            extra_q = 0.0
            if twq > 0 and np.isfinite(adv_qqq[i]) and adv_qqq[i] > 1 and aum > 0:
                participation = (twq * (aum * p.w_a * (1.0 - w_t))) / adv_qqq[i]
                extra_q = impact_coef * participation
            cost_q = twq * (fr.liquid + extra_q) + twq * fr.bil
            rq = g_q[i] - cost_q
            net_q[i] = 0.0 if not np.isfinite(rq) else rq

    if use_qqq:
        net_a = w_t * net_t + (1.0 - w_t) * net_q
        turn_a = w_t * turn_t + (1.0 - w_t) * turn_q
    else:
        net_a = net_t
        turn_a = turn_t

    gld, gldr = px["GLD"], rets(px["GLD"])
    c2, r2 = px[p.dual_b], rets(px[p.dual_b])
    g_d, turn_d, turn_db = dual_mom_with_turnover(
        gld, gldr, c2, r2, bil, i0, i1, p.dual_lb, p.dual_vt, me
    )
    net_d = np.zeros(n)
    for i in range(n):
        tw = turn_d[i]
        extra = 0.0
        if tw > 0 and np.isfinite(adv_liquid[i]) and adv_liquid[i] > 1:
            participation = (tw * (aum * (1.0 - p.w_a))) / adv_liquid[i]
            extra = impact_coef * participation
        cost = tw * (fr.liquid + extra) + turn_db[i] * fr.bil
        r = g_d[i] - cost
        if not np.isfinite(r):
            r = 0.0
        net_d[i] = r

    eq = blend_eq(np.array([p.w_a, 1.0 - p.w_a]), np.vstack([net_a, net_d]), i0, i1, aum)

    turn_port = p.w_a * turn_a + (1.0 - p.w_a) * (turn_d + turn_db)
    to = ann_turnover(turn_port, i0, i1)
    return eq, {"ann_one_way_turnover": to, "turn_a": ann_turnover(turn_a, i0, i1)}


def nested_metrics(eq, dates, spy_eq):
    return {
        "train": metrics_uncapped(eq, dates, BT_START, TRAIN_END),
        "mid": metrics_uncapped(eq, dates, MID_START, MID_END),
        "hold": metrics_uncapped(eq, dates, HOLD_START, HOLD_END),
        "full": metrics_uncapped(eq, dates, BT_START, BT_END),
        "spy_train": metrics_uncapped(spy_eq, dates, BT_START, TRAIN_END),
        "spy_mid": metrics_uncapped(spy_eq, dates, MID_START, MID_END),
        "spy_hold": metrics_uncapped(spy_eq, dates, HOLD_START, HOLD_END),
        "spy_full": metrics_uncapped(spy_eq, dates, BT_START, BT_END),
    }


def passes_selection_gates(m, to_ann: float) -> bool:
    """Train+mid only. Holdout is one-shot later."""
    if to_ann > MAX_ANN_ONE_WAY_TURNOVER:
        return False
    if m["train"]["sharpe_ratio"] < 1.0:
        return False
    if m["train"]["total_return"] <= m["spy_train"]["total_return"]:
        return False
    if m["mid"]["sharpe_ratio"] < 1.0:
        return False
    if m["mid"]["total_return"] <= m["spy_mid"]["total_return"]:
        return False
    return True


def passes_holdout_gates(m) -> bool:
    if m["hold"]["sharpe_ratio"] < 1.0:
        return False
    if m["hold"]["total_return"] <= m["spy_hold"]["total_return"]:
        return False
    if m["full"]["sharpe_ratio"] < 1.0:
        return False
    if m["full"]["total_return"] <= m["spy_full"]["total_return"]:
        return False
    return True


# Raised search/selection bar (harder for researchers; does not soften frozen env).
RAISED_MID_SHARPE = 1.05
RAISED_MAX_DD = -0.15
RAISED_HOLD_SHARPE = 1.2
STRESS2_FRICTION = friction_scaled(RETAIL_FRICTION, 2.0)


def passes_raised_selection_gates(m, to_ann: float) -> bool:
    """Train+mid raised bar. Holdout still one-shot via passes_holdout_gates."""
    if not passes_selection_gates(m, to_ann):
        return False
    if m["mid"]["sharpe_ratio"] < RAISED_MID_SHARPE:
        return False
    if m["full"]["max_drawdown"] < RAISED_MAX_DD:
        return False
    if m["mid"]["max_drawdown"] < RAISED_MAX_DD:
        return False
    return True


def passes_raised_holdout_gates(m) -> bool:
    if not passes_holdout_gates(m):
        return False
    if m["hold"]["sharpe_ratio"] < RAISED_HOLD_SHARPE:
        return False
    return True


def passes_stress_gates(m, min_full_sharpe: float = 1.0) -> bool:
    if m["full"]["sharpe_ratio"] < min_full_sharpe:
        return False
    for k in ("train", "mid", "hold", "full"):
        if m[k]["total_return"] <= m[f"spy_{k}"]["total_return"]:
            return False
    return True
