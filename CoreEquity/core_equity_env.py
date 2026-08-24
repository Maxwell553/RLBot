#!/usr/bin/env python3
"""
CoreEquity production environment (frozen).

No 2x/3x ETFs. Hot equity and dual stay cap 1.0.
QQQ calm sleeve may use GeneralEquity-style cash-financed leverage, cap ≤ 1.5.
Weekly close-to-close equity sleeve + month-end dual mom.
Retail friction + ADV impact, nested train/mid/holdout.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parent
sys.path.insert(0, str(PACK))

from real_alpha_env import (  # noqa: E402
    RETAIL_FRICTION,
    START_EQ,
    blend_eq,
    friction_scaled,
    load_ohlc,
    metrics_uncapped,
    month_end_mask,
    rets,
    window_slice,
)
from real_alpha_env import _atrp, _roll_vol, _sma_ok  # noqa: E402

PROD_FRICTION = RETAIL_FRICTION
STRESS_FRICTION = friction_scaled(RETAIL_FRICTION, 1.5)
STRESS2_FRICTION = friction_scaled(RETAIL_FRICTION, 2.0)

BT_START = date(2010, 1, 4)
TRAIN_END = date(2017, 12, 29)
MID_START = date(2018, 1, 2)
MID_END = date(2023, 12, 29)
HOLD_START = date(2024, 1, 2)
HOLD_END = date(2026, 7, 29)
BT_END = HOLD_END

IMPACT_PER_UNIT_ADV = 0.03
MAX_ANN_ONE_WAY_TURNOVER = 20.0
W_CAP = 1.0  # hot equity + dual
Q_CAP_MAX = 1.5  # QQQ calm sleeve only (cash-financed, same as GeneralEquity)

LEVERED_ETFS = frozenset(
    {
        "TQQQ",
        "SQQQ",
        "QLD",
        "QID",
        "UPRO",
        "SPXU",
        "SSO",
        "SDS",
        "SPXL",
        "SPXS",
        "TNA",
        "TZA",
        "UDOW",
        "SDOW",
        "TECL",
        "TECS",
        "SOXL",
        "SOXS",
        "FAS",
        "FAZ",
        "TMF",
        "TMV",
        "UBT",
        "TBT",
    }
)


@dataclass(frozen=True)
class ProdParams:
    ma: int = 151
    vt: float = 0.22
    atr_max: float = 0.06
    es: float = 0.278
    cool: int = 15
    w_a: float = 0.70
    dual_lb: int = 231
    dual_vt: float = 0.14
    dual_b: str = "TLT"
    rebal: str = "wk"
    vol_lb: int = 21
    # Sleeve A hybrid: fraction in eq_sym (1x); remainder = QQQ vol-target.
    eq_sym: str = "QQQ"
    w_hot: float = 1.0
    q_vt: float = 0.16
    q_atr_max: float = 0.06
    q_cap: float = 1.5  # QQQ sleeve only; clamped to Q_CAP_MAX
    vol_spike: float = 9.0
    atr_hyst: float = 0.05
    trend_sym: str = "QQQ"
    # sma: price > SMA(ma)
    # abs: 12m-style absolute momentum
    # abs_crash: abs(ma) AND abs(crash_lb) — long 2010s bull, flat in 2022-style bears
    # abs_sma: abs(ma) AND SMA(crash_lb) — same idea with a short SMA crash overlay
    # abs_or_sma: abs(ma) OR SMA(crash_lb) — stay long in 2010s bull; re-enter 2023 on short SMA
    trend_mode: str = "sma"
    crash_lb: int = 0  # overlay lookback; 0 = disabled


def _abs_ok(c, lb):
    n = len(c)
    ok = np.zeros(n, dtype=np.bool_)
    if lb <= 0:
        return ok
    for i in range(lb, n):
        if c[i] > 0 and c[i - lb] > 0 and c[i] >= c[i - lb]:
            ok[i] = True
    return ok


def trend_ok(px, p: ProdParams):
    series = px.get(p.trend_sym, px["QQQ"])
    mode = str(p.trend_mode)
    if mode == "sma":
        return _sma_ok(series, p.ma)
    if mode == "abs_crash":
        primary = _abs_ok(series, p.ma)
        crash = _abs_ok(series, int(p.crash_lb) if p.crash_lb else 126)
        return primary & crash
    if mode == "abs_sma":
        primary = _abs_ok(series, p.ma)
        crash = _sma_ok(series, int(p.crash_lb) if p.crash_lb else 63)
        return primary & crash
    if mode == "abs_or_sma":
        primary = _abs_ok(series, p.ma)
        fast = _sma_ok(series, int(p.crash_lb) if p.crash_lb else 42)
        return primary | fast
    return _abs_ok(series, p.ma)


def params_from_dict(d: dict) -> ProdParams:
    keys = ProdParams.__dataclass_fields__
    return ProdParams(**{k: d[k] for k in keys if k in d})


def week_end_mask(dates) -> np.ndarray:
    n = len(dates)
    m = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if dates[i].isocalendar()[1] != dates[i - 1].isocalendar()[1]:
            m[i - 1] = True
    m[-1] = True
    return m


def load_adv_usd(sym: str, dates, db: Path | None = None) -> np.ndarray:
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
    first = next((i for i in range(n) if np.isfinite(adv[i])), None)
    if first is not None:
        adv[:first] = adv[first]
        for i in range(first + 1, n):
            if not np.isfinite(adv[i]):
                adv[i] = adv[i - 1]
    return adv


def equity_cc_with_turnover(
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
    vol_spike=9.0,
    atr_hyst=0.0,
    w_cap=1.0,
):
    """Close-to-close sleeve. w_cap>1 uses cash-financed excess vs BIL (QQQ only)."""
    n = len(bil)
    daily = np.zeros(n)
    turn = np.zeros(n)
    w_path = np.zeros(n)
    vol_avg = np.full(n, np.nan)
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
        do_rebal = bool(rebal_mask[j]) or (i == i0 + 1)
        if flat:
            w = 0.0
        elif do_rebal:
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
                    cap = w_cap if w_cap > 0.0 else W_CAP
                    if ww > cap:
                        ww = cap
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


def dual_mom_with_turnover(c1, r1, c2, r2, bil, i0, i1, lb, vt, rebal_mask):
    n = len(bil)
    daily = np.zeros(n)
    turn = np.zeros(n)
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
                        w1 = min(W_CAP, vt / vol1[j])
                    elif np.isfinite(vol2[j]) and vol2[j] > 1e-8:
                        w2 = min(W_CAP, vt / vol2[j])
            wb = 1.0 - w1 - w2
        g = w1 * r1[i] + w2 * r2[i] + wb * bil[i]
        daily[i] = g
        turn[i] = abs(w1 - prev_w1) + abs(w2 - prev_w2)
        turn_bil[i] = abs(wb - prev_wb)
        prev_w1, prev_w2, prev_wb = w1, w2, wb
    return daily, turn, turn_bil


def ann_turnover(turn_w: np.ndarray, i0: int, i1: int) -> float:
    years = max((i1 - i0) / 252.0, 1e-9)
    return float(np.nansum(turn_w[i0 + 1 : i1 + 1]) / years)


def _assert_unlevered(p: ProdParams) -> None:
    for sym in (p.eq_sym, p.trend_sym, p.dual_b, "QQQ", "GLD", "BIL"):
        if str(sym).upper() in LEVERED_ETFS:
            raise ValueError(f"levered ETF not allowed: {sym}")
    q_cap = float(getattr(p, "q_cap", 1.0))
    if q_cap > Q_CAP_MAX + 1e-12:
        raise ValueError(f"q_cap {q_cap} exceeds Q_CAP_MAX {Q_CAP_MAX}")
    if q_cap < 0.0:
        raise ValueError("q_cap must be >= 0")


def _cost_sleeve(gross, turn, adv, aum_sleeve, cost_asset, cost_bil, impact_coef):
    n = len(gross)
    out = np.zeros(n)
    for i in range(n):
        tw = turn[i]
        extra = 0.0
        if tw > 0.0 and np.isfinite(adv[i]) and adv[i] > 1.0 and aum_sleeve > 0.0:
            extra = impact_coef * ((tw * aum_sleeve) / adv[i])
        r = gross[i] - tw * (cost_asset + extra) - tw * cost_bil
        out[i] = 0.0 if not np.isfinite(r) else r
    return out


def run_prod(
    dates,
    px,
    hot_ohlc,
    adv_hot,
    adv_liquid,
    p: ProdParams,
    fr=PROD_FRICTION,
    aum: float = START_EQ,
    impact_coef: float = IMPACT_PER_UNIT_ADV,
    qqq_ohlc=None,
    adv_qqq=None,
):
    """
    Sleeve A = 1x eq_sym CC (cap 1.0) optionally hybrid with QQQ VT (cap ≤ 1.5).
    If eq_sym is QQQ, the single equity sleeve uses q_cap (≤ 1.5).
    Sleeve B = GLD/dual_b dual mom month-end (cap 1.0).
    """
    _assert_unlevered(p)
    n = len(dates)
    i0, i1 = window_slice(dates, BT_START, BT_END)
    bil = rets(px["BIL"])
    me = month_end_mask(dates)
    mask = week_end_mask(dates) if p.rebal == "wk" else me
    ok = trend_ok(px, p)
    q_cap = float(min(Q_CAP_MAX, max(0.0, getattr(p, "q_cap", 1.0))))

    o, h, l, c = hot_ohlc
    vol = _roll_vol(rets(c), p.vol_lb)
    atr = _atrp(h, l, c)
    tr = rets(c)
    hot_is_qqq = str(p.eq_sym).upper() == "QQQ"
    hot_cap = q_cap if hot_is_qqq else W_CAP
    g_h, turn_h, _ = equity_cc_with_turnover(
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
        float(p.vol_spike),
        float(p.atr_hyst),
        hot_cap,
    )

    w_h = float(min(1.0, max(0.0, p.w_hot)))
    use_qqq = (w_h < 1.0 - 1e-12) and (not hot_is_qqq)
    if hot_is_qqq:
        use_qqq = False
        w_h = 1.0

    if use_qqq:
        if qqq_ohlc is None:
            qqq_ohlc = load_ohlc("QQQ", dates)
        if adv_qqq is None:
            adv_qqq = load_adv_usd("QQQ", dates)
        qo, qh, ql, qc = qqq_ohlc
        qvol = _roll_vol(rets(qc), p.vol_lb)
        qatr = _atrp(qh, ql, qc)
        qr = rets(qc)
        g_q, turn_q, _ = equity_cc_with_turnover(
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
            float(p.vol_spike),
            float(p.atr_hyst),
            q_cap,
        )
    else:
        g_q = np.zeros(n)
        turn_q = np.zeros(n)
        adv_qqq = adv_hot

    net_h = _cost_sleeve(
        g_h, turn_h, adv_hot, aum * p.w_a * w_h, fr.liquid, fr.bil, impact_coef
    )
    if use_qqq:
        net_q = _cost_sleeve(
            g_q,
            turn_q,
            adv_qqq,
            aum * p.w_a * (1.0 - w_h),
            fr.liquid,
            fr.bil,
            impact_coef,
        )
        net_a = w_h * net_h + (1.0 - w_h) * net_q
        turn_a = w_h * turn_h + (1.0 - w_h) * turn_q
    else:
        net_a = net_h
        turn_a = turn_h

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
            extra = impact_coef * ((tw * (aum * (1.0 - p.w_a))) / adv_liquid[i])
        r = g_d[i] - tw * (fr.liquid + extra) - turn_db[i] * fr.bil
        net_d[i] = 0.0 if not np.isfinite(r) else r

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


RAISED_MID_SHARPE = 1.05
RAISED_MAX_DD = -0.15
RAISED_HOLD_SHARPE = 1.2


def passes_raised_selection_gates(m, to_ann: float) -> bool:
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
