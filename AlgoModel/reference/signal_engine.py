#!/usr/bin/env python3
"""
Self-contained signal engine for pit_cross_sectional_momentum_v1.

No ALTrade imports. Depends only on the Python standard library.

Expected price input:
  prices: dict[str, list[tuple[date, float]]]
    each series sorted ascending by date (trading days only), adjusted close.

Main entrypoint:
  compute_target_weights(signal_date, prices, pit_csv_path, params=None) -> dict[str, float]
  Returns symbol -> portfolio weight (cash implied as 1 - sum(weights)).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

DELIST_ANNOTATION = re.compile(r"^.+-\d{6,}$")


@dataclass(frozen=True)
class Params:
    top_n: int = 30
    lookback_trading_days: int = 189
    skip_trading_days: int = 21
    portfolio_gross_weight: float = 0.9
    min_price: float = 5.0
    mom_min: float = -0.95
    mom_max: float = 5.0

    @classmethod
    def from_locked_json(cls, path: str | Path) -> "Params":
        raw = json.loads(Path(path).read_text())
        p = raw["params"] if "params" in raw else raw
        return cls(
            top_n=int(p["top_n"]),
            lookback_trading_days=int(p.get("lookback_trading_days", p.get("lookback", 189))),
            skip_trading_days=int(p.get("skip_trading_days", p.get("skip", 21))),
            portfolio_gross_weight=float(p.get("portfolio_gross_weight", 0.9)),
            min_price=float(p.get("min_price", 5.0)),
        )


def load_pit_snapshots(pit_csv: str | Path) -> list[tuple[date, set[str]]]:
    rows: list[tuple[date, set[str]]] = []
    with open(pit_csv, newline="") as f:
        for row in csv.DictReader(f):
            d = date.fromisoformat(row["date"][:10])
            mem: set[str] = set()
            for t in row["tickers"].strip().strip('"').split(","):
                t = t.strip()
                if not t or DELIST_ANNOTATION.match(t):
                    continue
                mem.add(t)
            rows.append((d, mem))
    rows.sort(key=lambda x: x[0])
    return rows


def membership_asof(snapshots: list[tuple[date, set[str]]], asof: date) -> set[str]:
    lo, hi = 0, len(snapshots) - 1
    ans = snapshots[0][1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if snapshots[mid][0] <= asof:
            ans = snapshots[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return set(ans)


def _series_to_maps(
    prices: dict[str, list[tuple[date, float]]],
) -> tuple[list[date], dict[str, dict[date, float]]]:
    """Build union calendar from SPY if present else all symbols."""
    if "SPY" in prices and prices["SPY"]:
        cal = [d for d, _ in prices["SPY"]]
    else:
        all_d = sorted({d for series in prices.values() for d, _ in series})
        cal = all_d
    maps: dict[str, dict[date, float]] = {}
    for sym, series in prices.items():
        maps[sym] = {d: float(px) for d, px in series}
    return cal, maps


def _index_on_or_before(cal: list[date], d: date) -> int | None:
    # last calendar index with cal[i] <= d
    lo, hi = 0, len(cal) - 1
    ans = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if cal[mid] <= d:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def compute_target_weights(
    signal_date: date,
    prices: dict[str, list[tuple[date, float]]],
    pit_csv_path: str | Path,
    params: Params | None = None,
    pit_snapshots: list[tuple[date, set[str]]] | None = None,
) -> dict[str, float]:
    """
    Compute long-only target weights for the portfolio (excluding explicit cash key).
    Cash weight = 1 - sum(returned weights).
    """
    p = params or Params()
    snaps = pit_snapshots if pit_snapshots is not None else load_pit_snapshots(pit_csv_path)
    members = membership_asof(snaps, signal_date)

    cal, maps = _series_to_maps(prices)
    i = _index_on_or_before(cal, signal_date)
    if i is None:
        return {}

    end_i = i - 1 - p.skip_trading_days
    start_i = end_i - p.lookback_trading_days
    if start_i < 0 or end_i < 0:
        return _spy_fallback(maps, p)

    end_d, start_d = cal[end_i], cal[start_i]
    scores: list[tuple[float, str]] = []
    for sym in members:
        series = maps.get(sym)
        if not series:
            continue
        a, b = series.get(start_d), series.get(end_d)
        if a is None or b is None:
            continue
        if a < p.min_price or b < p.min_price:
            continue
        mom = b / a - 1.0
        if not (p.mom_min < mom < p.mom_max):
            continue
        scores.append((mom, sym))

    if len(scores) < max(5, p.top_n // 2):
        return _spy_fallback(maps, p)

    scores.sort(reverse=True)
    picks = [sym for _, sym in scores[: p.top_n]]
    if not picks:
        return _spy_fallback(maps, p)
    w = p.portfolio_gross_weight / len(picks)
    return {sym: w for sym in picks}


def _spy_fallback(maps: dict[str, dict[date, float]], p: Params) -> dict[str, float]:
    if "SPY" in maps and maps["SPY"]:
        return {"SPY": p.portfolio_gross_weight}
    return {}


def execution_date(signal_date: date, trading_calendar: Iterable[date], lag: int = 1) -> date | None:
    """Return the trading day `lag` sessions after signal_date."""
    cal = list(trading_calendar)
    try:
        i = cal.index(signal_date)
    except ValueError:
        i = _index_on_or_before(cal, signal_date)
        if i is None:
            return None
    j = i + lag
    if j >= len(cal):
        return None
    return cal[j]


def month_end_signals(trading_calendar: list[date]) -> list[date]:
    """Last session of each calendar month in the calendar."""
    out: list[date] = []
    for i, d in enumerate(trading_calendar):
        if i + 1 == len(trading_calendar) or trading_calendar[i + 1].month != d.month:
            out.append(d)
    return out


def orders_to_targets(
    equity: float,
    positions: dict[str, float],
    marks: dict[str, float],
    targets: dict[str, float],
    *,
    min_notional: float = 50.0,
) -> list[dict]:
    """
    Convert target weights to order intents (sells first).
    positions: symbol -> shares
    marks: symbol -> price
    returns list of {symbol, side, qty}
    """
    # target shares
    tgt_shares: dict[str, float] = {}
    for sym, w in targets.items():
        px = marks.get(sym)
        if px is None or px <= 0:
            continue
        tgt_shares[sym] = (equity * w) / px

    intents: list[dict] = []
    # exits / trims
    for sym, qty in list(positions.items()):
        want = tgt_shares.get(sym, 0.0)
        if qty > want + 1e-8:
            sell = qty - want
            if sell * marks.get(sym, 0) >= min_notional or want <= 0:
                intents.append({"symbol": sym, "side": "sell", "qty": sell})
    # buys
    for sym, want in tgt_shares.items():
        have = positions.get(sym, 0.0)
        if want > have + 1e-8:
            buy = want - have
            if buy * marks.get(sym, 0) >= min_notional:
                intents.append({"symbol": sym, "side": "buy", "qty": buy})
    return intents


if __name__ == "__main__":
    # Tiny smoke test with synthetic prices
    from datetime import timedelta

    start = date(2020, 1, 2)
    cal = []
    d = start
    while len(cal) < 300:
        if d.weekday() < 5:
            cal.append(d)
        d += timedelta(days=1)

    def synth(sym: str, drift: float) -> list[tuple[date, float]]:
        px = 100.0
        out = []
        for i, day in enumerate(cal):
            px *= 1.0 + drift + (0.001 if (i % 17 == 0) else 0.0)
            out.append((day, px))
        return out

    prices = {
        "SPY": synth("SPY", 0.0004),
        "AAA": synth("AAA", 0.0010),
        "BBB": synth("BBB", 0.0002),
        "CCC": synth("CCC", 0.0008),
    }
    # minimal fake PIT: all members every day
    pit = Path("/tmp/_fake_pit.csv")
    pit.write_text("date,tickers\n2020-01-01,\"SPY,AAA,BBB,CCC\"\n")
    sig = month_end_signals(cal)[-1]
    w = compute_target_weights(sig, prices, pit, Params(top_n=2, lookback_trading_days=60, skip_trading_days=5))
    print("signal", sig, "weights", w, "cash", 1 - sum(w.values()))
