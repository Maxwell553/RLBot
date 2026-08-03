#!/usr/bin/env python3
"""
Realistic cash long-only research environment.

Addresses paper-only inflation in beat_spy_s20_v1:
  - one-way turnover costs by instrument (spread + slip)
  - stop fills at the session low when the stop is tagged (not magically at stop)
  - open-entry slip on overnight sleeve
  - SPY buy-hold benchmark with entry friction
  - stress multiplier on all frictions

Lag-1 only. No shorts / margin.

Stop-fill note
--------------
Daily OHLC cannot simulate a live stop. Filling at the stop price when the low
merely tags it is optimistic (especially on 3x ETFs). Default is fill at the
low (+ adverse bps). Pass fill_mode=0 for the legacy optimistic stop-price fill
only when explicitly stress-testing that artifact.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numba import njit

PACK = Path(__file__).resolve().parent
DB = PACK / "data" / "bars.db"


@dataclass(frozen=True)
class Friction:
    """Per-side trading friction in fraction of notional (not bps)."""

    tqqq: float = 0.0003  # ~3 bps one-way (liquid 3x; institutional/broker tier)
    liquid: float = 0.0002  # ~2 bps (aligns with CostModel half-spread+slip on ETFs)
    bil: float = 0.00005
    stop_adverse: float = 0.0005  # extra 5 bps on stop fills
    open_slip: float = 0.0001  # entry slip on weight increases only


# Default = institutional tier used by beat_spy_real_v1.
# Stress / retail cross-check: friction_scaled(FRICTION, 1.5) or
# Friction(tqqq=0.0006, liquid=0.00045, bil=0.00005, stop_adverse=0.0008, open_slip=0.0002)
FRICTION = Friction()
STRESS_MULT = 1.5
RETAIL_FRICTION = Friction(
    tqqq=0.0006, liquid=0.00045, bil=0.00005, stop_adverse=0.0008, open_slip=0.0002
)

BT_START = date(2010, 1, 4)
TRAIN_END = date(2017, 12, 29)
OOS_START = date(2018, 1, 2)
BT_END = date(2026, 7, 29)
START_EQ = 100_000.0


def window_slice(dates, start, end):
    idx = [i for i, d in enumerate(dates) if start <= d <= end]
    return idx[0], idx[-1]


def metrics_uncapped(eq, dates, start, end):
    i0, i1 = window_slice(dates, start, end)
    s = eq[i0 : i1 + 1]
    if len(s) < 5 or s[0] <= 0:
        return {"total_return": -1.0, "cagr": -1.0, "sharpe_ratio": -9.0, "max_drawdown": -1.0}
    total = float(s[-1] / s[0] - 1.0)
    if not math.isfinite(total):
        return {"total_return": -1.0, "cagr": -1.0, "sharpe_ratio": -9.0, "max_drawdown": -1.0}
    years = max((end - start).days / 365.25, 1e-6)
    cagr = float((s[-1] / s[0]) ** (1 / years) - 1)
    r = np.diff(s) / s[:-1]
    r = r[np.isfinite(r)]
    sharpe = float(r.mean() / r.std() * math.sqrt(252)) if len(r) and r.std() > 0 else 0.0
    peak = np.maximum.accumulate(s)
    dd = float(np.min(s / peak - 1.0))
    return {"total_return": total, "cagr": cagr, "sharpe_ratio": sharpe, "max_drawdown": dd}


def load_panel(symbols: list[str]):
    """Load close panel from pack-local bars.db (bundled subset)."""
    con = sqlite3.connect(DB)
    spy = con.execute(
        "SELECT timestamp, close FROM bars WHERE symbol='SPY' AND timestamp>='2008-01-01' ORDER BY timestamp"
    ).fetchall()
    dates = [date.fromisoformat(t[:10]) for t, _ in spy]
    idx = {d: i for i, d in enumerate(dates)}
    n = len(dates)
    px: dict[str, np.ndarray] = {}
    for s in symbols:
        arr = np.full(n, np.nan)
        for t, c in con.execute(
            "SELECT timestamp, close FROM bars WHERE symbol=? AND timestamp>='2008-01-01' ORDER BY timestamp",
            (s,),
        ):
            d = date.fromisoformat(t[:10])
            if d in idx:
                arr[idx[d]] = float(c)
        if np.isfinite(arr).sum() >= 500:
            px[s] = arr
    con.close()
    return dates, px


def load_ohlc(sym: str, dates):
    idx = {d: i for i, d in enumerate(dates)}
    n = len(dates)
    o = np.full(n, np.nan)
    h = np.full(n, np.nan)
    l = np.full(n, np.nan)
    c = np.full(n, np.nan)
    con = sqlite3.connect(DB)
    for t, oo, hh, ll, cc in con.execute(
        "SELECT timestamp,open,high,low,close FROM bars WHERE symbol=?", (sym,)
    ):
        d = date.fromisoformat(t[:10])
        if d in idx:
            i = idx[d]
            o[i], h[i], l[i], c[i] = oo, hh, ll, cc
    con.close()
    return (
        o.astype(np.float64),
        h.astype(np.float64),
        l.astype(np.float64),
        c.astype(np.float64),
    )


def rets(arr):
    r = np.zeros(len(arr), dtype=np.float64)
    for i in range(1, len(arr)):
        if arr[i - 1] > 0 and np.isfinite(arr[i]) and np.isfinite(arr[i - 1]):
            r[i] = arr[i] / arr[i - 1] - 1.0
    return r


def spy_bh_costed(spy_c, dates, entry_cost=0.0009):
    """SPY buy-hold with one-time entry friction (realistic BH)."""
    i0, i1 = window_slice(dates, BT_START, BT_END)
    eq = np.zeros(len(dates))
    eq[i0] = START_EQ * (1.0 - entry_cost)
    for i in range(i0 + 1, i1 + 1):
        if spy_c[i - 1] > 0 and np.isfinite(spy_c[i]):
            eq[i] = eq[i - 1] * (spy_c[i] / spy_c[i - 1])
        else:
            eq[i] = eq[i - 1]
    return eq


@njit
def _roll_vol(rr, lb):
    n = len(rr)
    v = np.full(n, np.nan)
    for i in range(lb - 1, n):
        m = 0.0
        ok = True
        for k in range(i - lb + 1, i + 1):
            if not np.isfinite(rr[k]):
                ok = False
                break
            m += rr[k]
        if not ok:
            continue
        m /= lb
        var = 0.0
        for k in range(i - lb + 1, i + 1):
            d = rr[k] - m
            var += d * d
        v[i] = math.sqrt(var / (lb - 1)) * math.sqrt(252.0)
    return v


@njit
def _sma_ok(q, ma):
    n = len(q)
    ok = np.zeros(n, dtype=np.bool_)
    for i in range(ma - 1, n):
        m = 0.0
        good = True
        for k in range(i - ma + 1, i + 1):
            if not np.isfinite(q[k]):
                good = False
                break
            m += q[k]
        if good and q[i] >= m / ma:
            ok[i] = True
    return ok


@njit
def _atrp(h, l, c, lb=14):
    n = len(c)
    out = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        if np.isfinite(h[i]) and np.isfinite(l[i]) and np.isfinite(c[i - 1]):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    for i in range(lb - 1, n):
        if c[i] > 0:
            m = 0.0
            for k in range(i - lb + 1, i + 1):
                m += tr[k]
            out[i] = (m / lb) / c[i]
    return out


@njit
def otc_day_ret(o, h, l, c, ds, stop_adverse, fill_mode=1):
    """
    Open-to-close with stop.

    fill_mode:
      0 = legacy optimistic: fill at -(ds + stop_adverse) when low tags stop
      1 = clean default: fill at low/open - 1 - stop_adverse when low tags stop
    """
    n = len(c)
    dr = np.zeros(n)
    hit = np.zeros(n)
    for i in range(n):
        if not (np.isfinite(o[i]) and o[i] > 0 and np.isfinite(c[i])):
            continue
        stop = o[i] * (1.0 - ds)
        if np.isfinite(l[i]) and l[i] <= stop:
            hit[i] = 1.0
            if fill_mode == 0:
                dr[i] = -(ds + stop_adverse)
            else:
                dr[i] = l[i] / o[i] - 1.0 - stop_adverse
        else:
            dr[i] = c[i] / o[i] - 1.0
    return dr, hit


@njit
def overnight_ret(o, c, open_slip=0.0):
    """
    Prior close → open gross return.
    open_slip kept for API compat; prefer charging slip via turnover on trades,
    not as a daily tax on every overnight mark (holding ≠ trading).
    """
    n = len(c)
    r = np.zeros(n)
    for i in range(1, n):
        if c[i - 1] > 0 and np.isfinite(o[i]):
            r[i] = o[i] / c[i - 1] - 1.0 - open_slip
    return r


@njit
def dual_mom_daily(c1, r1, c2, r2, bil, i0, i1, lb, vt, cost_asset, cost_bil, rebal_mask):
    """
    Dual momentum sleeve with turnover costs vs prior weight.
    Refresh target weights only when rebal_mask[j] is True (pass all-True for daily).
    """
    n = len(bil)
    daily = np.zeros(n)
    vol1 = _roll_vol(r1, 21)
    vol2 = _roll_vol(r2, 21)
    prev_w1 = 0.0
    prev_w2 = 0.0
    prev_wb = 1.0
    w1 = 0.0
    w2 = 0.0
    wb = 1.0
    for i in range(i0 + 1, i1 + 1):
        j = i - 1
        do_rebal = rebal_mask[j] or (i == i0 + 1)
        if do_rebal:
            w1 = 0.0
            w2 = 0.0
            if j >= lb and c1[j] > 0 and c1[j - lb] > 0 and c2[j] > 0 and c2[j - lb] > 0:
                tr1 = c1[j] / c1[j - lb] - 1.0
                tr2 = c2[j] / c2[j - lb] - 1.0
                if tr1 > 0 or tr2 > 0:
                    if tr1 >= tr2:
                        if np.isfinite(vol1[j]) and vol1[j] > 1e-8:
                            w1 = min(1.0, vt / vol1[j])
                    else:
                        if np.isfinite(vol2[j]) and vol2[j] > 1e-8:
                            w2 = min(1.0, vt / vol2[j])
            wb = 1.0 - w1 - w2
        g = w1 * r1[i] + w2 * r2[i] + wb * bil[i]
        turn = abs(w1 - prev_w1) * cost_asset + abs(w2 - prev_w2) * cost_asset + abs(wb - prev_wb) * cost_bil
        daily[i] = g - turn
        prev_w1, prev_w2, prev_wb = w1, w2, wb
    return daily


@njit
def _target_w(ok_flag, vol, atrp, atr_max, vt):
    if (not ok_flag) or (atr_max < 9.0 and np.isfinite(atrp) and atrp > atr_max):
        return 0.0
    if (not np.isfinite(vol)) or vol <= 1e-8:
        return 0.0
    w = vt / vol
    if w > 1.0:
        w = 1.0
    if w < 0.0:
        w = 0.0
    return w


@njit
def tqqq_split_daily(
    ok,
    vol21,
    atrp,
    on_r,
    day_r,
    bil,
    i0,
    i1,
    vt_on,
    vt_day,
    atr_max,
    es,
    cool,
    cost_tqqq,
    cost_bil,
    open_entry_slip=0.0,
):
    """
    Sequential overnight+day TQQQ sleeve with equity stop and realistic turnover.

    Timing (no leakage):
      - overnight into day i uses signal at close i-1
      - daytime on day i uses signal at close i-1
      - overnight into day i+1 is set at close i (signal at i)

    Costs (same-symbol continuous book):
      - open: trade |wd - held_overnight|
      - close: trade |wo_next - wd|
      - open_entry_slip on increases at the open only
    """
    n = len(bil)
    daily = np.zeros(n)
    eq = 1.0
    peak = 1.0
    flat = False
    cd = 0
    # warm-start overnight into first bar from signal at i0
    held = _target_w(ok[i0], vol21[i0], atrp[i0], atr_max, vt_on)
    for i in range(i0 + 1, i1 + 1):
        j = i - 1
        if flat:
            wd = 0.0
            wo_next = 0.0
            g = bil[i]
            turn = abs(0.0 - held) * cost_tqqq
            if held > 0.0:
                turn += held * cost_bil  # into bil
            if cd > 0:
                cd -= 1
            elif ok[i]:
                flat = False
                peak = eq
            held = 0.0
        else:
            # overnight already held was set yesterday; PnL uses actual held
            # day target from lag-1
            wd = _target_w(ok[j], vol21[j], atrp[j], atr_max, vt_day)
            g = (1.0 + held * on_r[i]) * (1.0 + wd * day_r[i]) - 1.0
            idle = max(0.0, 1.0 - max(held, wd))
            g = g + idle * bil[i]
            open_turn = abs(wd - held)
            turn = open_turn * cost_tqqq
            if wd > held:
                turn += (wd - held) * open_entry_slip
            # set next overnight from today's close signal (ok[i])
            wo_next = _target_w(ok[i], vol21[i], atrp[i], atr_max, vt_on)
            close_turn = abs(wo_next - wd)
            turn += close_turn * cost_tqqq
            # bil residual approx
            wb = idle
            prev_wb = max(0.0, 1.0 - held)
            turn += abs(wb - prev_wb) * cost_bil
            held = wo_next

        r = g - turn
        if not np.isfinite(r):
            r = 0.0
        daily[i] = r
        eq *= 1.0 + r
        if eq > peak:
            peak = eq
        if (not flat) and eq / peak - 1.0 <= -es:
            flat = True
            cd = cool
            held = 0.0
    return daily


@njit
def trend_sleeve_daily(c, r, bil, i0, i1, ma, vt, cost_asset, cost_bil, rebal_mask):
    n = len(c)
    daily = np.zeros(n)
    vol = _roll_vol(r, 21)
    prev_w = 0.0
    prev_wb = 1.0
    w = 0.0
    wb = 1.0
    for i in range(i0 + 1, i1 + 1):
        j = i - 1
        do_rebal = rebal_mask[j] or (i == i0 + 1)
        if do_rebal:
            w = 0.0
            if j >= ma - 1:
                m = 0.0
                good = True
                for k in range(j - ma + 1, j + 1):
                    if not np.isfinite(c[k]):
                        good = False
                        break
                    m += c[k]
                if good and c[j] >= m / ma and np.isfinite(vol[j]) and vol[j] > 1e-8:
                    w = min(1.0, vt / vol[j])
            wb = 1.0 - w
        g = w * r[i] + wb * bil[i]
        turn = abs(w - prev_w) * cost_asset + abs(wb - prev_wb) * cost_bil
        daily[i] = g - turn
        prev_w, prev_wb = w, wb
    return daily


@njit
def tqqq_cc_daily(
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
    cost_asset,
    cost_bil,
    rebal_mask,
):
    """
    Close-to-close levered sleeve (hold through session).
    Costs only on weight changes — realistic for a continuous long book.
    """
    n = len(bil)
    daily = np.zeros(n)
    eq = 1.0
    peak = 1.0
    flat = False
    cd = 0
    prev_w = 0.0
    prev_wb = 1.0
    w = 0.0
    for i in range(i0 + 1, i1 + 1):
        j = i - 1
        do_rebal = rebal_mask[j] or (i == i0 + 1)
        if flat:
            w = 0.0
        elif do_rebal:
            w = _target_w(ok[j], vol21[j], atrp[j], atr_max, vt)
        wb = 1.0 - w
        g = w * asset_r[i] + wb * bil[i]
        turn = abs(w - prev_w) * cost_asset + abs(wb - prev_wb) * cost_bil
        r = g - turn
        if not np.isfinite(r):
            r = 0.0
        daily[i] = r
        eq *= 1.0 + r
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
        prev_w, prev_wb = w, wb
    return daily


@njit
def overnight_only_daily(
    ok,
    vol21,
    atrp,
    on_r,
    bil,
    i0,
    i1,
    vt,
    atr_max,
    es,
    cool,
    cost_asset,
    cost_bil,
    open_entry_slip,
    rebal_mask,
):
    """Overnight-only levered sleeve; flat daytime residual in BIL."""
    n = len(bil)
    daily = np.zeros(n)
    eq = 1.0
    peak = 1.0
    flat = False
    cd = 0
    prev_w = 0.0
    prev_wb = 1.0
    w = 0.0
    for i in range(i0 + 1, i1 + 1):
        j = i - 1
        do_rebal = rebal_mask[j] or (i == i0 + 1)
        if flat:
            w = 0.0
        elif do_rebal:
            w = 0.0
            if ok[j] and (atr_max >= 9.0 or (not np.isfinite(atrp[j])) or atrp[j] <= atr_max):
                if np.isfinite(vol21[j]) and vol21[j] > 1e-8:
                    w = min(1.0, vt / vol21[j])
        wb = 1.0 - w
        g = w * on_r[i] + wb * bil[i]
        # Overnight-only is NOT a continuous hold: sell at open, re-buy at close.
        # Charge a full round-trip on the overnight notional each day in position,
        # plus slip on size increases, plus Δ costs when the target changes.
        turn = abs(w - prev_w) * cost_asset + abs(wb - prev_wb) * cost_bil
        if w > 0.0:
            turn += 2.0 * w * cost_asset
        if w > prev_w:
            turn += (w - prev_w) * open_entry_slip
        r = g - turn
        if not np.isfinite(r):
            r = 0.0
        daily[i] = r
        eq *= 1.0 + r
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
        prev_w, prev_wb = w, wb
    return daily


@njit
def blend_eq(parts_w, parts, i0, i1, start_eq):
    n = parts.shape[1]
    eq = np.zeros(n)
    eq[i0] = start_eq
    nparts = parts.shape[0]
    for i in range(i0 + 1, i1 + 1):
        r = 0.0
        for p in range(nparts):
            r += parts_w[p] * parts[p, i]
        eq[i] = eq[i - 1] * (1.0 + r)
    return eq


@njit
def quick_stats(eq, i0, i1):
    nobs = i1 - i0
    rets_ = np.empty(nobs)
    peak = eq[i0]
    max_dd = 0.0
    for i in range(i0 + 1, i1 + 1):
        rets_[i - i0 - 1] = eq[i] / eq[i - 1] - 1.0
        if eq[i] > peak:
            peak = eq[i]
        dd = eq[i] / peak - 1.0
        if dd < max_dd:
            max_dd = dd
    m = 0.0
    for i in range(nobs):
        m += rets_[i]
    m /= nobs
    var = 0.0
    for i in range(nobs):
        d = rets_[i] - m
        var += d * d
    sd = math.sqrt(var / (nobs - 1))
    sh = (m / sd) * math.sqrt(252.0) if sd > 1e-12 else -9.0
    return sh, eq[i1] / eq[i0] - 1.0, max_dd


def friction_scaled(fric: Friction, mult: float) -> Friction:
    return Friction(
        tqqq=fric.tqqq * mult,
        liquid=fric.liquid * mult,
        bil=fric.bil * mult,
        stop_adverse=fric.stop_adverse * mult,
        open_slip=fric.open_slip * mult,
    )


def month_end_mask(dates) -> np.ndarray:
    n = len(dates)
    m = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if dates[i].month != dates[i - 1].month:
            m[i - 1] = True
    m[-1] = True
    return m


def daily_rebal_mask(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.bool_)
