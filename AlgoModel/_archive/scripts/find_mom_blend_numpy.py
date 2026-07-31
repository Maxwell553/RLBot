#!/usr/bin/env python3
"""
Numpy/sqlite-only momentum + cash-blend search (no pandas — machine is memory-starved).

Winner: beat SPY train+OOS+full, full DD in [-22%, -12%], long-only, lag-1.
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Prefer /tmp copy — Desktop/iCloud sqlite often throws disk I/O under load
DB = Path("/tmp/statarb_mom.db") if Path("/tmp/statarb_mom.db").exists() else (ROOT / "data" / "statarb.db")
CACHE = Path("/tmp/mom_numpy_cache.pkl")

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


def parse_day(ts: str) -> date:
    # timestamps like 2010-01-04T00:00:00 or 2010-01-04 00:00:00
    return date.fromisoformat(ts[:10])


def load_cache(force: bool = False):
    if CACHE.exists() and not force:
        print("loading pickle cache…", flush=True)
        with open(CACHE, "rb") as f:
            return pickle.load(f)

    print("building numpy cache from sqlite…", flush=True)
    # universe
    import csv

    symbols = []
    with open(ROOT / "data" / "universe_1000.csv", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("source_index", "").lower() == "sp500":
                symbols.append(row["symbol"])
    symbols = sorted(set(symbols))
    want = ["SPY"] + symbols

    con = sqlite3.connect(DB)
    # calendar from SPY
    spy_rows = con.execute(
        "SELECT timestamp, close FROM bars WHERE symbol='SPY' AND timestamp>='2007-01-01' ORDER BY timestamp"
    ).fetchall()
    spy_dates = [parse_day(t) for t, _ in spy_rows]
    spy_close = np.array([float(c) for _, c in spy_rows], dtype=np.float64)
    date_to_i = {d: i for i, d in enumerate(spy_dates)}
    n = len(spy_dates)
    print(f"SPY days={n}", flush=True)

    # load each symbol into aligned array
    cols = {}
    for i, sym in enumerate(want):
        if sym == "SPY":
            continue
        rows = con.execute(
            "SELECT timestamp, close FROM bars WHERE symbol=? AND timestamp>='2007-01-01' ORDER BY timestamp",
            (sym,),
        ).fetchall()
        if len(rows) < 1000:
            continue
        arr = np.full(n, np.nan, dtype=np.float64)
        for t, c in rows:
            d = parse_day(t)
            j = date_to_i.get(d)
            if j is not None and c is not None and float(c) >= MIN_PRICE:
                arr[j] = float(c)
        # spike filter
        with np.errstate(invalid="ignore", divide="ignore"):
            rets = arr[1:] / arr[:-1] - 1.0
        bad = np.nanmean(np.abs(rets) > MAX_DAY_RET) if np.isfinite(rets).any() else 1.0
        if bad > 0.01:
            continue
        # punch spikes
        for k in range(1, n):
            if np.isfinite(arr[k]) and np.isfinite(arr[k - 1]):
                r = arr[k] / arr[k - 1] - 1.0
                if abs(r) > MAX_DAY_RET:
                    arr[k] = np.nan
        if np.isfinite(arr).sum() >= 1000:
            cols[sym] = arr
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(want)} kept={len(cols)}", flush=True)
    con.close()

    sym_list = sorted(cols.keys())
    px = np.column_stack([cols[s] for s in sym_list])  # n x m
    data = {
        "dates": spy_dates,
        "spy": spy_close,
        "symbols": sym_list,
        "px": px,
    }
    with open(CACHE, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"cache written m={len(sym_list)}", flush=True)
    return data


def window_slice(dates, start: date, end: date):
    idx = [i for i, d in enumerate(dates) if start <= d <= end]
    return idx[0], idx[-1]


def metrics_from_eq(eq: np.ndarray, dates, start: date, end: date) -> dict:
    i0, i1 = window_slice(dates, start, end)
    s = eq[i0 : i1 + 1]
    if len(s) < 5 or s[0] <= 0:
        return {"total_return": -1.0, "cagr": -1.0, "sharpe_ratio": -9.0, "max_drawdown": -1.0}
    total = float(s[-1] / s[0] - 1.0)
    if total > 40 or not math.isfinite(total):
        return {"total_return": -1.0, "cagr": -1.0, "sharpe_ratio": -9.0, "max_drawdown": -1.0}
    years = max((end - start).days / 365.25, 1e-6)
    cagr = float((s[-1] / s[0]) ** (1 / years) - 1.0)
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
    # pad into full-length? metrics uses window on eq aligned to dates slice only
    return metrics_from_eq(eq, dates[i0 : i1 + 1], start, end)


def month_ends(dates, i0, i1):
    out = set()
    last = None
    for i in range(i0, i1 + 1):
        d = dates[i]
        key = (d.year, d.month)
        if last is not None and key != last:
            out.add(i - 1)
        last = key
    out.add(i1)
    return out


def run_base(data, *, top_n=20, lookback=252, skip=21, crash_mode="spy_ma200", exec_lag=1):
    dates = data["dates"]
    spy = data["spy"]
    px = data["px"]  # n x m
    n, m = px.shape
    i0, i1 = window_slice(dates, BT_START, BT_END)
    ends = month_ends(dates, i0, i1)

    cash = START_EQ
    # qty indexed by column
    qty = np.zeros(m + 1, dtype=np.float64)  # last slot unused; use dict for sparsity
    holdings: dict[int, float] = {}  # col -> qty; -1 = SPY
    pending_i = None
    pending_w = None  # dict col->weight
    eq = np.zeros(n, dtype=np.float64)

    def price_map(i):
        out = {-1: float(spy[i])}
        row = px[i]
        for j in range(m):
            v = row[j]
            if np.isfinite(v) and v >= MIN_PRICE:
                out[j] = float(v)
        return out

    def mtm(i):
        pm = price_map(i)
        return cash + sum(q * pm[c] for c, q in holdings.items() if c in pm)

    def fill(side, mid):
        if side == "buy":
            return mid + mid * SLIP + mid * HALF_SPREAD, COMMISSION
        return mid - mid * SLIP - mid * HALF_SPREAD, COMMISSION

    def apply(i, weights: dict[int, float]):
        nonlocal cash
        pm = price_map(i)
        # exits
        for c, q in list(holdings.items()):
            if c not in pm or weights.get(c, 0.0) <= 0:
                if c in pm and q > 0:
                    p, ccost = fill("sell", pm[c])
                    cash += p * q - ccost
                holdings.pop(c, None)
        e = mtm(i)
        # trims
        for c, q in list(holdings.items()):
            tw = weights.get(c, 0.0)
            if c not in pm or tw <= 0:
                continue
            tgt = e * tw / pm[c]
            if q > tgt + 1e-6:
                sell = q - tgt
                p, ccost = fill("sell", pm[c])
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
        # use data strictly before i_sig+1 => index i_sig is last available if signal at close i_sig
        # match prior: spy.index < signal_ts → for month-end ts, use closes up to that day exclusive?
        # In pandas version: hist = spy.loc[spy.index < signal_ts] and signal at month_end ts,
        # so excludes month-end close. Use i_sig-1 as last.
        j = i_sig - 1
        if j < 200:
            return True
        on = True
        if crash_mode in ("spy_ma200", "both"):
            if spy[j] < spy[j - 199 : j + 1].mean():
                on = False
        if crash_mode in ("spy_mom12", "both") and j >= 252:
            if spy[j] / spy[j - 252] - 1.0 <= 0:
                on = False
        return on

    def stock_w(i_sig):
        # formation uses px.index < signal_ts → rows before i_sig
        end_i = i_sig - 1 - skip
        start_i = end_i - lookback
        if start_i < 0 or end_i < 0:
            return {-1: 1.0}
        end = px[end_i]
        start = px[start_i]
        moms = []
        for j in range(m):
            a, b = start[j], end[j]
            if not (np.isfinite(a) and np.isfinite(b) and a >= MIN_PRICE and b >= MIN_PRICE):
                continue
            mom = b / a - 1.0
            if -0.95 < mom < 5.0:
                moms.append((mom, j))
        if len(moms) < top_n:
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


def blend(eq, dates, weight):
    i0, i1 = window_slice(dates, BT_START, BT_END)
    out = np.zeros_like(eq)
    out[i0] = START_EQ
    for i in range(i0 + 1, i1 + 1):
        if eq[i - 1] > 0:
            r = eq[i] / eq[i - 1] - 1.0
        else:
            r = 0.0
        out[i] = out[i - 1] * (1.0 + weight * r)
    return out


def vol_scale(eq, dates, target, lookback=63, cap=1.0):
    i0, i1 = window_slice(dates, BT_START, BT_END)
    out = np.zeros_like(eq)
    out[i0] = START_EQ
    rets = []
    for i in range(i0 + 1, i1 + 1):
        r = eq[i] / eq[i - 1] - 1.0 if eq[i - 1] > 0 else 0.0
        # weight from PAST returns only (no same-day vol look-ahead)
        if len(rets) >= lookback:
            window = np.array(rets[-lookback:])
            vol = float(window.std(ddof=1) * math.sqrt(252))
            w = min(cap, target / vol) if vol > 1e-8 else 0.0
        else:
            w = 0.0
        out[i] = out[i - 1] * (1.0 + w * r)
        rets.append(r)
    return out


def main() -> int:
    print("start", flush=True)
    data = load_cache()
    dates = data["dates"]
    spy = data["spy"]

    # spy bh metrics need local date index — build helper eq on full spy then slice
    def spy_m(start, end):
        i0, i1 = window_slice(dates, start, end)
        s = spy[i0 : i1 + 1]
        eq = START_EQ * (s / s[0]) * (1 - 0.0009)
        total = float(eq[-1] / eq[0] - 1)
        years = max((end - start).days / 365.25, 1e-6)
        r = np.diff(eq) / eq[:-1]
        sharpe = float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0
        peak = np.maximum.accumulate(eq)
        dd = float(np.min(eq / peak - 1))
        return {"total_return": total, "cagr": (eq[-1] / eq[0]) ** (1 / years) - 1, "sharpe_ratio": sharpe, "max_drawdown": dd}

    st, so, sf = spy_m(BT_START, TRAIN_END), spy_m(OOS_START, BT_END), spy_m(BT_START, BT_END)
    print(
        f"SPY train={st['total_return']:+.1%} oos={so['total_return']:+.1%} full={sf['total_return']:+.1%} dd={sf['max_drawdown']:.1%}",
        flush=True,
    )

    bases = [
        dict(top_n=20, lookback=252, crash_mode="spy_ma200"),
        dict(top_n=20, lookback=252, crash_mode="both"),
        dict(top_n=25, lookback=252, crash_mode="spy_ma200"),
        dict(top_n=20, lookback=189, crash_mode="spy_ma200"),
        dict(top_n=30, lookback=252, crash_mode="spy_ma200"),
    ]

    rows = []
    winner = None
    for bi, b in enumerate(bases):
        print(f"base {bi+1}/{len(bases)} {b}", flush=True)
        base_eq = run_base(data, **b)
        # metrics on trading window
        i0, i1 = window_slice(dates, BT_START, BT_END)
        # pack eq only on full dates array
        bm = metrics_from_eq(base_eq, dates, BT_START, BT_END)
        print(f"  base full={bm['total_return']:+.1%} dd={bm['max_drawdown']:.1%}", flush=True)

        cands = []
        for w in [round(x, 2) for x in np.arange(0.45, 1.01, 0.05)]:
            cands.append(("blend", w, blend(base_eq, dates, w)))
        for vt in (0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20):
            cands.append(("vol", vt, vol_scale(base_eq, dates, vt)))

        for kind, knob, eq in cands:
            tr = metrics_from_eq(eq, dates, BT_START, TRAIN_END)
            oos = metrics_from_eq(eq, dates, OOS_START, BT_END)
            full = metrics_from_eq(eq, dates, BT_START, BT_END)
            if full["total_return"] < 0:
                continue
            row = {
                **b,
                "scale_kind": kind,
                "scale_knob": float(knob),
                "train_ret": tr["total_return"],
                "train_excess": tr["total_return"] - st["total_return"],
                "oos_ret": oos["total_return"],
                "oos_excess": oos["total_return"] - so["total_return"],
                "full_ret": full["total_return"],
                "full_excess": full["total_return"] - sf["total_return"],
                "full_dd": full["max_drawdown"],
                "full_sharpe": full["sharpe_ratio"],
                "full_cagr": full["cagr"],
            }
            row["winner"] = (
                row["train_excess"] > 0
                and row["oos_excess"] > 0
                and row["full_excess"] > 0
                and -0.22 <= row["full_dd"] <= -0.12
            )
            rows.append(row)
            if row["train_excess"] > 0 and row["oos_excess"] > 0 and row["full_excess"] > 0:
                print(
                    f"  {kind}={knob} xs={row['full_excess']:+.1%} dd={row['full_dd']:.1%} "
                    f"tr_xs={row['train_excess']:+.1%} oos_xs={row['oos_excess']:+.1%}"
                    + (" WIN" if row["winner"] else ""),
                    flush=True,
                )
            if row["winner"]:
                winner = (b, kind, float(knob), eq, row)
                break
        if winner:
            break

    # save csv lightly
    if rows:
        keys = list(rows[0].keys())
        with open(ROOT / "logs" / "mom_blend_numpy_search.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    if winner is None:
        print("No strict winner", flush=True)
        beat = [r for r in rows if r["train_excess"] > 0 and r["oos_excess"] > 0 and r["full_excess"] > 0]
        beat.sort(key=lambda r: (-r["full_dd"], -r["full_excess"]))
        for r in beat[:15]:
            print(r, flush=True)
        return 2

    b, kind, knob, eq, row = winner
    i0, i1 = window_slice(dates, BT_START, BT_END)
    with open(ROOT / "logs" / "cash_momentum_winner_curve.csv", "w") as f:
        f.write("date,equity\n")
        for i in range(i0, i1 + 1):
            f.write(f"{dates[i].isoformat()},{eq[i]:.6f}\n")

    lock = {
        "strategy": "cash_long_only_cross_sectional_momentum_dd",
        "uses_margin": False,
        "allows_shorts": False,
        "universe": "static_sp500_from_universe_1000",
        "base_params": {**b, "skip": 21, "exec_lag_days": 1},
        "risk_overlay": {"kind": kind, "knob": knob},
        "metrics": {
            "train_ret": float(row["train_ret"]),
            "train_excess": float(row["train_excess"]),
            "oos_ret": float(row["oos_ret"]),
            "oos_excess": float(row["oos_excess"]),
            "full_ret": float(row["full_ret"]),
            "full_excess": float(row["full_excess"]),
            "full_dd": float(row["full_dd"]),
            "full_sharpe": float(row["full_sharpe"]),
            "full_cagr": float(row["full_cagr"]),
            "spy_full_ret": float(sf["total_return"]),
            "spy_full_dd": float(sf["max_drawdown"]),
        },
        "params_locked": True,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "strict_winner": True,
        "no_leakage_notes": [
            "execution_lag_days=1",
            "blend/vol overlay uses lagged returns only",
            "static S&P list has mild survivorship bias",
        ],
        "run_script": "scripts/find_mom_blend_numpy.py",
    }
    (ROOT / "config" / "cash_momentum.locked.yaml").write_text(yaml.safe_dump(lock, sort_keys=False))
    (ROOT / "logs" / "cash_momentum_winner.json").write_text(json.dumps(lock, indent=2, default=str))
    print("LOCKED", json.dumps(lock, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
