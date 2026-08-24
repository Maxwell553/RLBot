#!/usr/bin/env python3
"""
Cash long-only research helpers (no numba).

Lag-1 only. No shorts. No 2x/3x ETFs. QQQ may be cash-financed up to 1.5x in CoreEquity.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parent
DB = PACK / "data" / "bars.db"


@dataclass(frozen=True)
class Friction:
    """Per-side trading friction in fraction of notional (not bps)."""

    liquid: float = 0.0002
    bil: float = 0.00005
    stop_adverse: float = 0.0005
    open_slip: float = 0.0001


FRICTION = Friction()
RETAIL_FRICTION = Friction(
    liquid=0.00045, bil=0.00005, stop_adverse=0.0008, open_slip=0.0002
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
    """Load close panel from pack-local bars.db."""
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
    prev = np.asarray(arr, dtype=np.float64)
    for i in range(1, len(arr)):
        if prev[i - 1] > 0 and np.isfinite(prev[i]) and np.isfinite(prev[i - 1]):
            r[i] = prev[i] / prev[i - 1] - 1.0
    return r


def spy_bh_costed(spy_c, dates, entry_cost=0.0009):
    i0, i1 = window_slice(dates, BT_START, BT_END)
    eq = np.zeros(len(dates))
    eq[i0] = START_EQ * (1.0 - entry_cost)
    for i in range(i0 + 1, i1 + 1):
        if spy_c[i - 1] > 0 and np.isfinite(spy_c[i]):
            eq[i] = eq[i - 1] * (spy_c[i] / spy_c[i - 1])
        else:
            eq[i] = eq[i - 1]
    return eq


def _roll_vol(rr, lb):
    n = len(rr)
    v = np.full(n, np.nan)
    r = np.asarray(rr, dtype=np.float64)
    for i in range(lb - 1, n):
        w = r[i - lb + 1 : i + 1]
        if not np.all(np.isfinite(w)):
            continue
        v[i] = float(np.std(w, ddof=1) * math.sqrt(252.0))
    return v


def _sma_ok(q, ma):
    n = len(q)
    ok = np.zeros(n, dtype=np.bool_)
    x = np.asarray(q, dtype=np.float64)
    csum = np.cumsum(np.where(np.isfinite(x), x, 0.0))
    finite = np.isfinite(x).astype(np.int32)
    fsum = np.cumsum(finite)
    for i in range(ma - 1, n):
        nfin = int(fsum[i] - (fsum[i - ma] if i >= ma else 0))
        if nfin != ma:
            continue
        total = csum[i] - (csum[i - ma] if i >= ma else 0.0)
        if x[i] >= total / ma:
            ok[i] = True
    return ok


def _atrp(h, l, c, lb=14):
    n = len(c)
    out = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        if np.isfinite(h[i]) and np.isfinite(l[i]) and np.isfinite(c[i - 1]):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    for i in range(lb - 1, n):
        if c[i] > 0:
            out[i] = (tr[i - lb + 1 : i + 1].sum() / lb) / c[i]
    return out


def blend_eq(parts_w, parts, i0, i1, start_eq):
    n = parts.shape[1]
    eq = np.zeros(n)
    eq[i0] = start_eq
    w = np.asarray(parts_w, dtype=np.float64)
    for i in range(i0 + 1, i1 + 1):
        eq[i] = eq[i - 1] * (1.0 + float(np.dot(w, parts[:, i])))
    return eq


def friction_scaled(fric: Friction, mult: float) -> Friction:
    return Friction(
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
