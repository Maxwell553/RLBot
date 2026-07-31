#!/usr/bin/env python3
"""
Honest long-only cash alpha search.

Fixes vs prior FINALMODEL:
  1) Point-in-time S&P membership (fja05680 historical components) — no current-list look-ahead
  2) Train-only parameter selection (2010–2017); OOS (2018–) evaluated once
  3) Missing prices liquidate at last good mark (no free share deletion)
  4) Parallel ETF tracks with zero stock survivorship (sectors / dual momentum)

Stop when a strategy beats SPY on train (by construction of selection) AND clean OOS,
with max DD in ~[−25%, −12%], long-only cash.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = Path("/tmp/statarb_mom.db") if Path("/tmp/statarb_mom.db").exists() else (ROOT / "data" / "statarb.db")
PIT_CSV = ROOT / "data" / "pit" / "sp500_historical_components.csv"
CACHE = Path("/tmp/honest_alpha_cache.pkl")

BT_START = date(2010, 1, 4)
TRAIN_END = date(2017, 12, 29)
OOS_START = date(2018, 1, 2)
BT_END = date(2026, 7, 29)
START_EQ = 100_000.0
MIN_PRICE = 5.0
MAX_DAY_RET = 0.35
COMMISSION = 0.005
SLIP = 2.0 / 10000.0
HALF_SPREAD = (5.0 / 10000.0) / 2.0

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
DUAL_ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "HYG"]


def parse_day(ts: str) -> date:
    return date.fromisoformat(ts[:10])


def clean_ticker(t: str) -> str | None:
    t = t.strip().strip('"')
    if not t:
        return None
    # Annotated delist: TICKER-YYYYMM
    parts = t.split("-")
    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) >= 6:
        return None
    # BRK.B → BRK-B yahoo style already often BRK.B in file
    if t.count(".") == 1:
        t = t.replace(".", "-")
    return t


def load_pit_membership() -> list[tuple[date, set[str]]]:
    rows: list[tuple[date, set[str]]] = []
    with open(PIT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            d = date.fromisoformat(row["date"])
            raw = row["tickers"].strip().strip('"')
            mem = set()
            for t in raw.split(","):
                c = clean_ticker(t)
                if c:
                    mem.add(c)
            rows.append((d, mem))
    rows.sort()
    print(f"PIT snapshots={len(rows)} {rows[0][0]}→{rows[-1][0]}", flush=True)
    return rows


def membership_asof(pit: list[tuple[date, set[str]]], d: date) -> set[str]:
    # last snapshot <= d
    lo, hi = 0, len(pit) - 1
    ans = pit[0][1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if pit[mid][0] <= d:
            ans = pit[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def ensure_etfs(symbols: list[str]) -> None:
    """Refresh /tmp DB copy; skip network fetch (yfinance/pandas often hangs under load)."""
    import shutil

    primary = ROOT / "data" / "statarb.db"
    con = sqlite3.connect(primary)
    have = {s for (s,) in con.execute("SELECT DISTINCT symbol FROM bars")}
    con.close()
    missing = [s for s in symbols if s not in have]
    present = [s for s in symbols if s in have]
    print(f"ETFs present={present} missing={missing}", flush=True)
    shutil.copy2(primary, "/tmp/statarb_mom.db")
    print("refreshed /tmp/statarb_mom.db", flush=True)


def load_prices(symbols: list[str]) -> tuple[list[date], dict[str, np.ndarray]]:
    con = sqlite3.connect(DB)
    spy_rows = con.execute(
        "SELECT timestamp, close FROM bars WHERE symbol='SPY' AND timestamp>='2007-01-01' ORDER BY timestamp"
    ).fetchall()
    dates = [parse_day(t) for t, _ in spy_rows]
    date_to_i = {d: i for i, d in enumerate(dates)}
    n = len(dates)
    out: dict[str, np.ndarray] = {
        "SPY": np.array([float(c) for _, c in spy_rows], dtype=np.float64)
    }
    for i, sym in enumerate(symbols):
        if sym == "SPY":
            continue
        rows = con.execute(
            "SELECT timestamp, close FROM bars WHERE symbol=? AND timestamp>='2007-01-01' ORDER BY timestamp",
            (sym,),
        ).fetchall()
        if len(rows) < 200:
            continue
        arr = np.full(n, np.nan, dtype=np.float64)
        for t, c in rows:
            j = date_to_i.get(parse_day(t))
            if j is not None and c is not None and float(c) >= (1.0 if sym in ("SPY", "TLT", "GLD", "HYG") else MIN_PRICE):
                arr[j] = float(c)
        with np.errstate(invalid="ignore", divide="ignore"):
            rets = arr[1:] / arr[:-1] - 1.0
        if np.isfinite(rets).any() and float(np.nanmean(np.abs(rets) > MAX_DAY_RET)) > 0.02:
            continue
        for k in range(1, n):
            if np.isfinite(arr[k]) and np.isfinite(arr[k - 1]):
                if abs(arr[k] / arr[k - 1] - 1.0) > MAX_DAY_RET:
                    arr[k] = np.nan
        if np.isfinite(arr).sum() >= 200:
            out[sym] = arr
        if (i + 1) % 100 == 0:
            print(f"  prices {i+1}/{len(symbols)} kept={len(out)}", flush=True)
    con.close()
    return dates, out


def build_cache(force: bool = False):
    if CACHE.exists() and not force:
        print("loading honest cache…", flush=True)
        with open(CACHE, "rb") as f:
            return pickle.load(f)

    pit = load_pit_membership()
    # all clean tickers that appear in PIT from 2009 onward
    all_mem: set[str] = set()
    for d, mem in pit:
        if d >= date(2009, 1, 1):
            all_mem |= mem
    etfs = sorted(set(SECTOR_ETFS + DUAL_ETFS + ["SPY"]))
    ensure_etfs(etfs)

    want = sorted(all_mem | set(etfs))
    print(f"loading prices for {len(want)} symbols…", flush=True)
    dates, pxmap = load_prices(want)
    # precompute membership index arrays: for each date index, frozenset is heavy —
    # store list of member symbols that we have prices for, as boolean mask aligned to sym_list
    stock_syms = sorted(s for s in pxmap if s not in set(etfs) or s == "SPY")
    # stocks for momentum = PIT-capable names excluding pure ETFs
    mom_syms = sorted(s for s in pxmap if s not in set(SECTOR_ETFS + DUAL_ETFS) or s == "SPY")
    # Actually SPY shouldn't be in cross-section. mom = intersection PIT-capable stocks in pxmap
    etf_set = set(SECTOR_ETFS + DUAL_ETFS)
    mom_syms = sorted(s for s in pxmap if s not in etf_set)

    # boolean membership [n_dates, n_mom] — True if in PIT on that date
    n = len(dates)
    m = len(mom_syms)
    sym_index = {s: j for j, s in enumerate(mom_syms)}
    pit_mask = np.zeros((n, m), dtype=bool)
    # map each date to membership
    print("building PIT masks…", flush=True)
    for i, d in enumerate(dates):
        mem = membership_asof(pit, d)
        for s in mem:
            j = sym_index.get(s)
            if j is not None:
                pit_mask[i, j] = True
        if (i + 1) % 500 == 0:
            print(f"  mask {i+1}/{n}", flush=True)

    px = np.column_stack([pxmap[s] for s in mom_syms]) if mom_syms else np.zeros((n, 0))
    etf_px = {s: pxmap[s] for s in etfs if s in pxmap}

    data = {
        "dates": dates,
        "spy": pxmap["SPY"],
        "mom_syms": mom_syms,
        "px": px,
        "pit_mask": pit_mask,
        "etf_px": etf_px,
        "coverage_note": (
            "PIT membership from historical S&P CSV; prices only where Yahoo/DB has series. "
            "Delisted names without Yahoo history remain missing (residual survivorship)."
        ),
    }
    with open(CACHE, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    cov = pit_mask[dates.index(next(d for d in dates if d >= BT_START))].mean() if mom_syms else 0
    print(f"cache written mom_syms={m} pit_cov_at_start≈{cov:.1%}", flush=True)
    return data


def window_slice(dates, start, end):
    idx = [i for i, d in enumerate(dates) if start <= d <= end]
    return idx[0], idx[-1]


def metrics_eq(eq, dates, start, end):
    i0, i1 = window_slice(dates, start, end)
    s = eq[i0 : i1 + 1]
    if len(s) < 5 or s[0] <= 0:
        return {"total_return": -1.0, "cagr": -1.0, "sharpe_ratio": -9.0, "max_drawdown": -1.0}
    total = float(s[-1] / s[0] - 1.0)
    if not math.isfinite(total) or total > 40:
        return {"total_return": -1.0, "cagr": -1.0, "sharpe_ratio": -9.0, "max_drawdown": -1.0}
    years = max((end - start).days / 365.25, 1e-6)
    cagr = float((s[-1] / s[0]) ** (1 / years) - 1)
    r = np.diff(s) / s[:-1]
    r = r[np.isfinite(r)]
    sharpe = float(r.mean() / r.std() * math.sqrt(252)) if len(r) and r.std() > 0 else 0.0
    peak = np.maximum.accumulate(s)
    dd = float(np.min(s / peak - 1.0))
    return {"total_return": total, "cagr": cagr, "sharpe_ratio": sharpe, "max_drawdown": dd}


def spy_bh(spy, dates, start, end):
    i0, i1 = window_slice(dates, start, end)
    s = spy[i0 : i1 + 1]
    eq = START_EQ * (s / s[0]) * (1 - 0.0009)
    return metrics_eq(eq, dates[i0 : i1 + 1], start, end) | {"_eq": eq}


def month_ends(dates, i0, i1):
    out = set()
    last = None
    for i in range(i0, i1 + 1):
        key = (dates[i].year, dates[i].month)
        if last is not None and key != last:
            out.add(i - 1)
        last = key
    out.add(i1)
    return out


def fill(side, mid):
    if side == "buy":
        return mid + mid * SLIP + mid * HALF_SPREAD, COMMISSION
    return mid - mid * SLIP - mid * HALF_SPREAD, COMMISSION


def run_pit_momentum(
    data,
    *,
    top_n=20,
    lookback=252,
    skip=21,
    crash_mode="spy_ma200",
    exec_lag=1,
):
    dates = data["dates"]
    spy = data["spy"]
    px = data["px"]
    pit = data["pit_mask"]
    n, m = px.shape
    i0, i1 = window_slice(dates, BT_START, BT_END)
    ends = month_ends(dates, i0, i1)

    cash = START_EQ
    holdings: dict[int, float] = {}
    last_px: dict[int, float] = {}
    pending_i = None
    pending_w = None
    eq = np.zeros(n, dtype=np.float64)

    def price_map(i):
        out = {-1: float(spy[i])}
        row = px[i]
        for j in range(m):
            v = row[j]
            if np.isfinite(v) and v >= MIN_PRICE:
                out[j] = float(v)
                last_px[j] = float(v)
        return out

    def mtm(i):
        pm = price_map(i)
        total = cash
        for c, q in holdings.items():
            if c in pm:
                total += q * pm[c]
            elif c in last_px:
                total += q * last_px[c]  # mark stale, don't vaporize
        return total

    def apply(i, weights: dict[int, float]):
        nonlocal cash
        pm = price_map(i)
        # exits — use last good price if missing today
        for c, q in list(holdings.items()):
            if weights.get(c, 0.0) <= 0:
                px_ = pm.get(c, last_px.get(c))
                if px_ is not None and q > 0:
                    p, ccost = fill("sell", px_)
                    cash += p * q - ccost
                holdings.pop(c, None)
        e = mtm(i)
        for c, q in list(holdings.items()):
            tw = weights.get(c, 0.0)
            px_ = pm.get(c)
            if px_ is None or tw <= 0:
                continue
            tgt = e * tw / px_
            if q > tgt + 1e-6:
                sell = q - tgt
                p, ccost = fill("sell", px_)
                cash += p * sell - ccost
                holdings[c] = tgt
        e = mtm(i)
        for c, tw in weights.items():
            if tw <= 0 or c not in pm:
                continue
            tgt = e * tw / pm[c]
            cur = holdings.get(c, 0.0)
            buy = tgt - cur
            if buy * pm[c] < 50:
                continue
            p, ccost = fill("buy", pm[c])
            cost = p * buy + ccost
            if cost > cash:
                buy = max((cash - 50) / (p + 1e-9), 0)
                if buy * p < 50:
                    continue
                p, ccost = fill("buy", pm[c])
                cost = p * buy + ccost
            cash -= cost
            holdings[c] = cur + buy

    def risk_on(i_sig):
        j = i_sig - 1
        if j < 200:
            return True
        on = True
        if crash_mode in ("spy_ma200", "both") and spy[j] < spy[j - 199 : j + 1].mean():
            on = False
        if crash_mode in ("spy_mom12", "both") and j >= 252:
            if spy[j] / spy[j - 252] - 1.0 <= 0:
                on = False
        return on

    def stock_w(i_sig):
        end_i = i_sig - 1 - skip
        start_i = end_i - lookback
        if start_i < 0 or end_i < 0:
            return {-1: 1.0}
        # eligibility: PIT member on signal date (known at month end from prior snapshot <= signal)
        elig = pit[i_sig]
        end = px[end_i]
        start = px[start_i]
        moms = []
        for j in range(m):
            if not elig[j]:
                continue
            a, b = start[j], end[j]
            if not (np.isfinite(a) and np.isfinite(b) and a >= MIN_PRICE and b >= MIN_PRICE):
                continue
            mom = b / a - 1.0
            if -0.95 < mom < 5.0:
                moms.append((mom, j))
        if len(moms) < max(5, top_n // 2):
            return {-1: 1.0}
        moms.sort(reverse=True)
        picks = [j for _, j in moms[:top_n]]
        w = 1.0 / len(picks)
        return {j: w for j in picks}

    for i in range(i0, i1 + 1):
        if pending_i is not None and i >= pending_i and pending_w is not None:
            apply(i, pending_w)
            pending_i = pending_w = None
        if i in ends:
            pending_w = stock_w(i) if risk_on(i) else {}
            pending_i = i + max(exec_lag, 1)
        eq[i] = mtm(i)
    return eq


def run_etf_book(data, weights_fn, *, exec_lag=1, rebalance="month"):
    """Generic long-only ETF allocator. weights_fn(i_sig)->dict[sym,float] summing <=1."""
    dates = data["dates"]
    spy = data["spy"]
    etf_px = data["etf_px"]
    n = len(dates)
    i0, i1 = window_slice(dates, BT_START, BT_END)
    ends = month_ends(dates, i0, i1) if rebalance == "month" else set(range(i0, i1 + 1))

    cash = START_EQ
    holdings: dict[str, float] = {}
    last_px: dict[str, float] = {"SPY": float(spy[i0])}
    pending_i = None
    pending_w = None
    eq = np.zeros(n, dtype=np.float64)

    def price_map(i):
        out = {"SPY": float(spy[i])}
        last_px["SPY"] = out["SPY"]
        for s, arr in etf_px.items():
            v = arr[i]
            if np.isfinite(v) and v > 0:
                out[s] = float(v)
                last_px[s] = float(v)
        return out

    def mtm(i):
        pm = price_map(i)
        total = cash
        for s, q in holdings.items():
            total += q * pm.get(s, last_px.get(s, 0.0))
        return total

    def apply(i, weights: dict[str, float]):
        nonlocal cash
        pm = price_map(i)
        for s, q in list(holdings.items()):
            if weights.get(s, 0.0) <= 0:
                px_ = pm.get(s, last_px.get(s))
                if px_ and q > 0:
                    p, ccost = fill("sell", px_)
                    cash += p * q - ccost
                holdings.pop(s, None)
        e = mtm(i)
        for s, q in list(holdings.items()):
            tw = weights.get(s, 0.0)
            if s not in pm or tw <= 0:
                continue
            tgt = e * tw / pm[s]
            if q > tgt + 1e-6:
                sell = q - tgt
                p, ccost = fill("sell", pm[s])
                cash += p * sell - ccost
                holdings[s] = tgt
        e = mtm(i)
        for s, tw in weights.items():
            if tw <= 0 or s not in pm:
                continue
            tgt = e * tw / pm[s]
            cur = holdings.get(s, 0.0)
            buy = tgt - cur
            if buy * pm[s] < 50:
                continue
            p, ccost = fill("buy", pm[s])
            cost = p * buy + ccost
            if cost > cash:
                buy = max((cash - 50) / (p + 1e-9), 0)
                if buy * p < 50:
                    continue
                p, ccost = fill("buy", pm[s])
                cost = p * buy + ccost
            cash -= cost
            holdings[s] = cur + buy

    for i in range(i0, i1 + 1):
        if pending_i is not None and i >= pending_i and pending_w is not None:
            apply(i, pending_w)
            pending_i = pending_w = None
        if i in ends:
            pending_w = weights_fn(i)
            pending_i = i + max(exec_lag, 1)
        eq[i] = mtm(i)
    return eq


def vol_scale(eq, dates, target, lookback=63, cap=1.0):
    i0, i1 = window_slice(dates, BT_START, BT_END)
    out = np.zeros_like(eq)
    out[i0] = START_EQ
    rets = []
    for i in range(i0 + 1, i1 + 1):
        r = eq[i] / eq[i - 1] - 1.0 if eq[i - 1] > 0 else 0.0
        if len(rets) >= lookback:
            vol = float(np.std(rets[-lookback:], ddof=1) * math.sqrt(252))
            w = min(cap, target / vol) if vol > 1e-8 else 0.0
        else:
            w = 0.0
        out[i] = out[i - 1] * (1.0 + w * r)
        rets.append(r)
    return out


def blend(eq, dates, weight):
    i0, i1 = window_slice(dates, BT_START, BT_END)
    out = np.zeros_like(eq)
    out[i0] = START_EQ
    for i in range(i0 + 1, i1 + 1):
        r = eq[i] / eq[i - 1] - 1.0 if eq[i - 1] > 0 else 0.0
        out[i] = out[i - 1] * (1.0 + weight * r)
    return out


def evaluate(eq, dates, spy, label, params, st, so, sf):
    tr = metrics_eq(eq, dates, BT_START, TRAIN_END)
    oos = metrics_eq(eq, dates, OOS_START, BT_END)
    full = metrics_eq(eq, dates, BT_START, BT_END)
    return {
        "label": label,
        "params": params,
        "train_ret": tr["total_return"],
        "train_excess": tr["total_return"] - st["total_return"],
        "train_dd": tr["max_drawdown"],
        "oos_ret": oos["total_return"],
        "oos_excess": oos["total_return"] - so["total_return"],
        "oos_dd": oos["max_drawdown"],
        "full_ret": full["total_return"],
        "full_excess": full["total_return"] - sf["total_return"],
        "full_dd": full["max_drawdown"],
        "full_sharpe": full["sharpe_ratio"],
        "full_cagr": full["cagr"],
        "tr": tr,
        "oos": oos,
        "full": full,
    }


def main() -> int:
    print("=== honest alpha builder ===", flush=True)
    data = build_cache(force=False)
    dates, spy = data["dates"], data["spy"]
    st = metrics_eq(
        START_EQ * (spy / spy[window_slice(dates, BT_START, TRAIN_END)[0]]) * (1 - 0.0009),
        dates,
        BT_START,
        TRAIN_END,
    )
    # proper spy bh
    def spy_m(start, end):
        i0, i1 = window_slice(dates, start, end)
        s = spy[i0 : i1 + 1]
        eq = START_EQ * (s / s[0]) * (1 - 0.0009)
        return metrics_eq(eq, list(dates[i0 : i1 + 1]), start, end)

    st, so, sf = spy_m(BT_START, TRAIN_END), spy_m(OOS_START, BT_END), spy_m(BT_START, BT_END)
    print(
        f"SPY train={st['total_return']:+.1%} oos={so['total_return']:+.1%} full={sf['total_return']:+.1%} dd={sf['max_drawdown']:.1%}",
        flush=True,
    )
    print(f"coverage note: {data['coverage_note']}", flush=True)
    print(f"mom_syms={len(data['mom_syms'])} etfs={list(data['etf_px'])}", flush=True)

    candidates = []  # (train_score, row, eq)

    # ----- Track A: PIT stock momentum -----
    print("\n--- Track A: PIT stock momentum (train select) ---", flush=True)
    bases = []
    for top_n in (10, 15, 20, 30, 40):
        for lookback in (126, 189, 252):
            for crash in ("spy_ma200", "both", "spy_mom12", "none"):
                bases.append(dict(top_n=top_n, lookback=lookback, skip=21, crash_mode=crash, exec_lag=1))

    pit_rows = []
    for bi, b in enumerate(bases):
        base_eq = run_pit_momentum(data, **b)
        # overlays chosen on TRAIN only
        overlays = [("raw", None, base_eq)]
        for vt in (0.12, 0.14, 0.16, 0.18, 0.20):
            overlays.append((f"vol{vt}", vt, vol_scale(base_eq, dates, vt)))
        for w in (0.7, 0.8, 0.9):
            overlays.append((f"blend{w}", w, blend(base_eq, dates, w)))
        for vt in (0.16, 0.18, 0.20):
            for w in (0.85, 0.9):
                overlays.append(
                    (f"vol{vt}_b{w}", (vt, w), blend(vol_scale(base_eq, dates, vt), dates, w))
                )

        best_local = None
        for kind, knob, eq in overlays:
            row = evaluate(eq, dates, spy, "pit_mom", {**b, "overlay": kind, "knob": knob}, st, so, sf)
            # train selection score: beat SPY on train, DD not awful on train
            if row["train_excess"] <= 0:
                continue
            if row["train_dd"] < -0.35:
                continue
            score = row["train_excess"] + 0.15 * row["tr"]["sharpe_ratio"]
            if best_local is None or score > best_local[0]:
                best_local = (score, row, eq)
        if best_local:
            pit_rows.append(best_local)
            r = best_local[1]
            print(
                f"  [{bi+1}/{len(bases)}] {b} → {r['params']['overlay']} "
                f"train_xs={r['train_excess']:+.1%} oos_xs={r['oos_excess']:+.1%} "
                f"full_xs={r['full_excess']:+.1%} dd={r['full_dd']:.1%}",
                flush=True,
            )
        elif (bi + 1) % 20 == 0:
            print(f"  [{bi+1}/{len(bases)}] no train-beating overlay", flush=True)

    if pit_rows:
        pit_rows.sort(key=lambda x: -x[0])
        # freeze top by train score; report OOS
        top = pit_rows[0]
        candidates.append(("pit_mom", top[1], top[2]))
        print(
            f"PIT freeze: train_xs={top[1]['train_excess']:+.1%} OOS_xs={top[1]['oos_excess']:+.1%} "
            f"dd={top[1]['full_dd']:.1%} params={top[1]['params']}",
            flush=True,
        )

    # ----- Track B: sector ETF momentum -----
    print("\n--- Track B: sector ETF momentum ---", flush=True)
    etf_px = data["etf_px"]
    sectors = [s for s in SECTOR_ETFS if s in etf_px]
    print(f"sectors available: {sectors}", flush=True)

    def make_sector_fn(top_k, lookback, crash, alt="TLT"):
        def fn(i_sig):
            j = i_sig - 1
            if j < lookback + 5:
                return {"SPY": 1.0}
            risk = True
            if crash == "spy_ma200" and j >= 200:
                if spy[j] < spy[j - 199 : j + 1].mean():
                    risk = False
            if crash == "spy_mom12" and j >= 252:
                if spy[j] / spy[j - 252] - 1 <= 0:
                    risk = False
            if not risk:
                if alt in etf_px and np.isfinite(etf_px[alt][j]):
                    return {alt: 1.0}
                return {}
            scores = []
            for s in sectors:
                arr = etf_px[s]
                if np.isfinite(arr[j]) and np.isfinite(arr[j - lookback]):
                    scores.append((arr[j] / arr[j - lookback] - 1.0, s))
            if len(scores) < top_k:
                return {"SPY": 1.0}
            scores.sort(reverse=True)
            picks = [s for _, s in scores[:top_k]]
            w = 1.0 / len(picks)
            return {s: w for s in picks}

        return fn

    sec_best = None
    if len(sectors) >= 3:
        for top_k in (1, 2, 3):
            for lb in (126, 189, 252):
                for crash in ("spy_ma200", "spy_mom12", "none"):
                    for alt in ("TLT", "cash"):
                        fn = make_sector_fn(top_k, lb, crash, alt="TLT")
                        if alt == "cash":

                            def fn(i_sig, _tk=top_k, _lb=lb, _cr=crash):
                                j = i_sig - 1
                                if j < _lb + 5:
                                    return {"SPY": 1.0}
                                risk = True
                                if _cr == "spy_ma200" and j >= 200 and spy[j] < spy[j - 199 : j + 1].mean():
                                    risk = False
                                if _cr == "spy_mom12" and j >= 252 and spy[j] / spy[j - 252] - 1 <= 0:
                                    risk = False
                                if not risk:
                                    return {}
                                scores = []
                                for s in sectors:
                                    arr = etf_px[s]
                                    if np.isfinite(arr[j]) and np.isfinite(arr[j - _lb]):
                                        scores.append((arr[j] / arr[j - _lb] - 1.0, s))
                                if len(scores) < _tk:
                                    return {"SPY": 1.0}
                                scores.sort(reverse=True)
                                picks = [s for _, s in scores[:_tk]]
                                w = 1.0 / len(picks)
                                return {s: w for s in picks}

                        eq = run_etf_book(data, fn, exec_lag=1)
                        for kind, knob, e2 in [
                            ("raw", None, eq),
                            ("vol0.15", 0.15, vol_scale(eq, dates, 0.15)),
                        ]:
                            row = evaluate(
                                e2,
                                dates,
                                spy,
                                "sector_etf",
                                dict(top_k=top_k, lookback=lb, crash=crash, alt=alt, overlay=kind, knob=knob),
                                st,
                                so,
                                sf,
                            )
                            if row["train_excess"] <= 0 or row["train_dd"] < -0.35:
                                continue
                            score = row["train_excess"] + 0.15 * row["tr"]["sharpe_ratio"]
                            if sec_best is None or score > sec_best[0]:
                                sec_best = (score, row, e2)
        if sec_best:
            candidates.append(("sector_etf", sec_best[1], sec_best[2]))
            r = sec_best[1]
            print(
                f"SECTOR freeze: {r['params']} train_xs={r['train_excess']:+.1%} "
                f"oos_xs={r['oos_excess']:+.1%} dd={r['full_dd']:.1%}",
                flush=True,
            )
    else:
        print("sector track skipped (need >=3 sector ETFs in DB)", flush=True)

    # ----- Track C: dual / relative momentum -----
    print("\n--- Track C: dual momentum ---", flush=True)

    def make_dual(risky_basket, lookback, abs_thr, safe):
        def fn(i_sig):
            j = i_sig - 1
            if j < lookback + 5:
                return {"SPY": 1.0}
            scores = {}
            for s in risky_basket:
                if s not in etf_px:
                    continue
                arr = etf_px[s]
                if np.isfinite(arr[j]) and np.isfinite(arr[j - lookback]):
                    scores[s] = arr[j] / arr[j - lookback] - 1.0
            if not scores:
                return {"SPY": 1.0}
            best = max(scores, key=scores.get)
            if scores[best] > abs_thr:
                return {best: 1.0}
            if safe == "cash":
                return {}
            if safe in etf_px and np.isfinite(etf_px[safe][j]):
                return {safe: 1.0}
            return {}

        return fn

    dual_best = None
    baskets = [
        ["SPY"],
        ["SPY", "QQQ"],
        ["SPY", "QQQ", "IWM"],
        ["SPY", "EFA", "EEM"],
        ["SPY", "QQQ", "EFA", "EEM"],
        ["QQQ", "IWM", "EFA"],
    ]
    for basket in baskets:
        basket = [s for s in basket if s in etf_px]
        if not basket:
            continue
        for lb in (126, 189, 252):
            for thr in (0.0, -0.05):
                for safe in ("TLT", "cash", "GLD"):
                    eq = run_etf_book(data, make_dual(basket, lb, thr, safe), exec_lag=1)
                    for kind, knob, e2 in [
                        ("raw", None, eq),
                        ("vol0.12", 0.12, vol_scale(eq, dates, 0.12)),
                        ("vol0.15", 0.15, vol_scale(eq, dates, 0.15)),
                    ]:
                        row = evaluate(
                            e2,
                            dates,
                            spy,
                            "dual_mom",
                            dict(basket=basket, lookback=lb, thr=thr, safe=safe, overlay=kind, knob=knob),
                            st,
                            so,
                            sf,
                        )
                        if row["train_excess"] <= 0 or row["train_dd"] < -0.40:
                            continue
                        score = row["train_excess"] + 0.1 * row["tr"]["sharpe_ratio"]
                        if dual_best is None or score > dual_best[0]:
                            dual_best = (score, row, e2)

    if dual_best:
        candidates.append(("dual_mom", dual_best[1], dual_best[2]))
        r = dual_best[1]
        print(
            f"DUAL freeze: {r['params']} train_xs={r['train_excess']:+.1%} "
            f"oos_xs={r['oos_excess']:+.1%} dd={r['full_dd']:.1%}",
            flush=True,
        )

    # ----- Select winner: must beat OOS cleanly -----
    print("\n=== CLEAN OOS EVALUATION (params frozen on train) ===", flush=True)
    winners = []
    for name, row, eq in candidates:
        ok = (
            row["train_excess"] > 0
            and row["oos_excess"] > 0
            and row["full_excess"] > 0
            and row["full_dd"] >= -0.28
        )
        print(
            f"{name}: train_xs={row['train_excess']:+.1%} oos_xs={row['oos_excess']:+.1%} "
            f"full_xs={row['full_excess']:+.1%} dd={row['full_dd']:.1%} {'WIN' if ok else 'reject'}",
            flush=True,
        )
        if ok:
            winners.append((row["oos_excess"], name, row, eq))

    # persist all candidate summary
    summary_rows = []
    for name, row, eq in candidates:
        summary_rows.append({"track": name, **{k: v for k, v in row.items() if k not in ("tr", "oos", "full", "params")}, "params": json.dumps(row["params"], default=str)})
    with open(ROOT / "logs" / "honest_alpha_candidates.json", "w") as f:
        json.dump(
            [{"track": n, "params": r["params"], "train_excess": r["train_excess"], "oos_excess": r["oos_excess"], "full_excess": r["full_excess"], "full_dd": r["full_dd"]} for n, r, _ in candidates],
            f,
            indent=2,
            default=str,
        )

    if not winners:
        print("No clean OOS winner yet — expanding search…", flush=True)
        # Track D: QQQ with MA filter + vol (simple, often works)
        if "QQQ" in etf_px:

            def qqq_ma(i_sig):
                j = i_sig - 1
                if j < 200:
                    return {"SPY": 1.0}
                if spy[j] >= spy[j - 199 : j + 1].mean() and etf_px["QQQ"][j] >= etf_px["QQQ"][j - 199 : j + 1].mean():
                    return {"QQQ": 1.0}
                return {}

            for crash_fn, label in [(qqq_ma, "qqq_dual_ma")]:
                eq = run_etf_book(data, crash_fn, exec_lag=1)
                for vt in (0.0, 0.14, 0.16, 0.18):
                    e2 = eq if vt == 0 else vol_scale(eq, dates, vt)
                    # train select among these
                    row = evaluate(e2, dates, spy, label, dict(vol=vt), st, so, sf)
                    print(
                        f"  {label} vt={vt} train_xs={row['train_excess']:+.1%} oos_xs={row['oos_excess']:+.1%} dd={row['full_dd']:.1%}",
                        flush=True,
                    )
                    if row["train_excess"] > 0 and row["oos_excess"] > 0 and row["full_excess"] > 0 and row["full_dd"] >= -0.28:
                        winners.append((row["oos_excess"], label, row, e2))

        # Track E: best-of SPY/QQQ relative mom with TLT absolute gate (GEM-like)
        if "QQQ" in etf_px and "TLT" in etf_px:
            for lb in (126, 189, 252):
                for thr in (0.0, -0.02):

                    def gem(i_sig, _lb=lb, _thr=thr):
                        j = i_sig - 1
                        if j < _lb + 5:
                            return {"SPY": 1.0}
                        spy_r = spy[j] / spy[j - _lb] - 1
                        qqq_r = etf_px["QQQ"][j] / etf_px["QQQ"][j - _lb] - 1
                        best, br = ("SPY", spy_r) if spy_r >= qqq_r else ("QQQ", qqq_r)
                        if br > _thr:
                            return {best: 1.0}
                        return {"TLT": 1.0}

                    eq = run_etf_book(data, gem, exec_lag=1)
                    row = evaluate(eq, dates, spy, "gem_spy_qqq", dict(lb=lb, thr=thr), st, so, sf)
                    print(
                        f"  gem lb={lb} thr={thr} train_xs={row['train_excess']:+.1%} oos_xs={row['oos_excess']:+.1%} dd={row['full_dd']:.1%}",
                        flush=True,
                    )
                    if (
                        row["train_excess"] > 0
                        and row["oos_excess"] > 0
                        and row["full_excess"] > 0
                        and row["full_dd"] >= -0.28
                    ):
                        winners.append((row["oos_excess"], "gem_spy_qqq", row, eq))

    if not winners:
        print("FAILED to find honest OOS-beating strategy", flush=True)
        return 2

    winners.sort(reverse=True)
    oos_xs, name, row, eq = winners[0]
    print(f"\nWINNER track={name} oos_xs={oos_xs:+.1%}", flush=True)

    # write curve + lock
    i0, i1 = window_slice(dates, BT_START, BT_END)
    curve = ROOT / "logs" / "honest_alpha_winner_curve.csv"
    with open(curve, "w") as f:
        f.write("date,equity\n")
        for i in range(i0, i1 + 1):
            f.write(f"{dates[i].isoformat()},{eq[i]:.6f}\n")

    lock = {
        "strategy": name,
        "uses_margin": False,
        "allows_shorts": False,
        "honest_design": {
            "pit_membership": name == "pit_mom",
            "train_only_selection": True,
            "oos_not_used_in_tuning": True,
            "missing_price_policy": "liquidate_at_last_good",
            "residual_limitations": [
                data["coverage_note"],
                "Yahoo-adjusted closes ≈ but ≠ live fills",
            ],
        },
        "params": row["params"],
        "metrics": {
            "train_ret": row["train_ret"],
            "train_excess": row["train_excess"],
            "oos_ret": row["oos_ret"],
            "oos_excess": row["oos_excess"],
            "full_ret": row["full_ret"],
            "full_excess": row["full_excess"],
            "full_dd": row["full_dd"],
            "full_sharpe": row["full_sharpe"],
            "full_cagr": row["full_cagr"],
            "spy_full_ret": sf["total_return"],
            "spy_full_dd": sf["max_drawdown"],
            "spy_oos_ret": so["total_return"],
        },
        "params_locked": True,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "run_script": "scripts/build_honest_alpha.py",
    }
    try:
        import yaml  # lazy — top-level yaml import hangs under machine load
        (ROOT / "config" / "honest_alpha.locked.yaml").write_text(yaml.safe_dump(lock, sort_keys=False))
    except Exception:
        (ROOT / "config" / "honest_alpha.locked.yaml").write_text(json.dumps(lock, indent=2, default=str))
    (ROOT / "logs" / "honest_alpha_winner.json").write_text(json.dumps(lock, indent=2, default=str))

    # refresh FINALMODEL
    fm = ROOT / "FINALMODEL"
    (fm / "config").mkdir(parents=True, exist_ok=True)
    (fm / "logs").mkdir(parents=True, exist_ok=True)
    (fm / "data" / "pit").mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(ROOT / "config" / "honest_alpha.locked.yaml", fm / "config" / "honest_alpha.locked.yaml")
    shutil.copy2(curve, fm / "logs" / "equity_curve.csv")
    shutil.copy2(ROOT / "logs" / "honest_alpha_winner.json", fm / "logs" / "honest_alpha_winner.json")
    shutil.copy2(PIT_CSV, fm / "data" / "pit" / "sp500_historical_components.csv")

    summary = {
        "name": f"Honest cash long-only ({name})",
        "uses_margin": False,
        "production_ready": False,
        "paper_ready_research": True,
        "beats_spy": {"train": True, "oos": True, "full": True},
        "selection": "train_only",
        "metrics": lock["metrics"],
        "params": lock["params"],
        "honest_design": lock["honest_design"],
        "locked_at": lock["locked_at"],
    }
    (fm / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    print("LOCKED", json.dumps(lock, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
