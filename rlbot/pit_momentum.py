"""PIT S&P cross-sectional momentum (locked) — signal → target weights.

Adapted from ``FINALMODEL/reference/signal_engine.py``. Locked knobs live in
``FINALMODEL/config/strategy.locked.json``; do not retune.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from rlbot.run_artifacts import PROJECT_ROOT

DELIST_ANNOTATION = re.compile(r"^.+-\d{6,}$")

FINALMODEL_DIR = PROJECT_ROOT / "FINALMODEL"
DEFAULT_LOCKED_CONFIG = FINALMODEL_DIR / "config" / "strategy.locked.json"
DEFAULT_PIT_CSV = FINALMODEL_DIR / "data" / "pit" / "sp500_historical_components.csv"

STRATEGY_ID = "pit_cross_sectional_momentum_v1"
PAPER_RUN_ID = "FINALMODEL"


@dataclass(frozen=True)
class Params:
    top_n: int = 30
    lookback_trading_days: int = 189
    skip_trading_days: int = 21
    execution_lag_trading_days: int = 1
    portfolio_gross_weight: float = 0.9
    min_price: float = 5.0
    mom_min: float = -0.95
    mom_max: float = 5.0

    @classmethod
    def from_locked_json(cls, path: str | Path | None = None) -> "Params":
        pth = Path(path) if path is not None else DEFAULT_LOCKED_CONFIG
        raw = json.loads(pth.read_text(encoding="utf-8"))
        p = raw["params"] if "params" in raw else raw
        clip = p.get("momentum_clip") or {}
        return cls(
            top_n=int(p["top_n"]),
            lookback_trading_days=int(
                p.get("lookback_trading_days", p.get("lookback", 189))
            ),
            skip_trading_days=int(p.get("skip_trading_days", p.get("skip", 21))),
            execution_lag_trading_days=int(p.get("execution_lag_trading_days", 1)),
            portfolio_gross_weight=float(p.get("portfolio_gross_weight", 0.9)),
            min_price=float(p.get("min_price", 5.0)),
            mom_min=float(clip.get("min", -0.95)),
            mom_max=float(clip.get("max", 5.0)),
        )


def to_yahoo_symbol(sym: str) -> str:
    """Map PIT / broker dots to Yahoo (BRK.B → BRK-B)."""
    s = str(sym).strip().upper()
    if not s:
        return s
    return s.replace(".", "-")


def from_yahoo_symbol(sym: str) -> str:
    """Best-effort reverse map (BRK-B → BRK.B) for display; PIT already uses dots."""
    s = str(sym).strip().upper()
    if s.count("-") == 1 and not s.endswith("-"):
        left, right = s.split("-", 1)
        if len(right) <= 2 and right.isalpha():
            return f"{left}.{right}"
    return s


def load_pit_snapshots(pit_csv: str | Path | None = None) -> list[tuple[date, set[str]]]:
    path = Path(pit_csv) if pit_csv is not None else DEFAULT_PIT_CSV
    # Cache parsed membership next to the CSV (and under execution/) — the raw
    # file is multi-MB with ~daily rows × ~500 tickers; reparsing every call
    # is painful on iCloud Desktop.
    cache_candidates = [
        path.with_suffix(path.suffix + ".pkl"),
        PROJECT_ROOT / "execution" / "paper_pit_momentum" / "pit_snapshots.pkl",
    ]
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    for cache in cache_candidates:
        if not cache.is_file():
            continue
        try:
            import pickle

            if cache.stat().st_mtime < mtime:
                continue
            blob = pickle.loads(cache.read_bytes())
            if isinstance(blob, list) and blob and isinstance(blob[0], tuple):
                return blob
        except Exception:  # noqa: BLE001
            continue

    rows: list[tuple[date, set[str]]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = date.fromisoformat(row["date"][:10])
            mem: set[str] = set()
            for t in row["tickers"].strip().strip('"').split(","):
                t = t.strip().upper()
                if not t or DELIST_ANNOTATION.match(t):
                    continue
                mem.add(t)
            rows.append((d, mem))
    rows.sort(key=lambda x: x[0])

    import pickle

    payload = pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL)
    for cache in cache_candidates:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(payload)
        except OSError:
            continue
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
    if "SPY" in prices and prices["SPY"]:
        cal = [d for d, _ in prices["SPY"]]
    else:
        cal = sorted({d for series in prices.values() for d, _ in series})
    maps: dict[str, dict[date, float]] = {}
    for sym, series in prices.items():
        maps[sym.upper()] = {d: float(px) for d, px in series}
    return cal, maps


def _index_on_or_before(cal: list[date], d: date) -> int | None:
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


def _spy_fallback(maps: dict[str, dict[date, float]], p: Params) -> dict[str, float]:
    if "SPY" in maps and maps["SPY"]:
        return {"SPY": p.portfolio_gross_weight}
    return {}


def compute_target_weights(
    signal_date: date,
    prices: dict[str, list[tuple[date, float]]],
    pit_csv_path: str | Path | None = None,
    params: Params | None = None,
    pit_snapshots: list[tuple[date, set[str]]] | None = None,
) -> dict[str, float]:
    """Long-only target weights (no cash key). Cash = 1 - sum(weights)."""
    p = params or Params.from_locked_json()
    snaps = (
        pit_snapshots
        if pit_snapshots is not None
        else load_pit_snapshots(pit_csv_path)
    )
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


def execution_date(
    signal_date: date, trading_calendar: Iterable[date], lag: int = 1
) -> date | None:
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
    out: list[date] = []
    for i, d in enumerate(trading_calendar):
        if i + 1 == len(trading_calendar) or trading_calendar[i + 1].month != d.month:
            out.append(d)
    return out


def last_actionable_signal(
    trading_calendar: list[date],
    as_of: date,
    *,
    lag: int = 1,
) -> tuple[date, date] | None:
    """Most recent (signal_date, trade_date) with trade_date <= as_of."""
    signals = month_end_signals(trading_calendar)
    best: tuple[date, date] | None = None
    for sig in signals:
        if sig > as_of:
            break
        trade = execution_date(sig, trading_calendar, lag=lag)
        if trade is None or trade > as_of:
            continue
        best = (sig, trade)
    return best


def orders_to_targets(
    equity: float,
    positions: dict[str, float],
    marks: dict[str, float],
    targets: dict[str, float],
    *,
    min_notional: float = 50.0,
) -> list[dict]:
    """Convert target weights to order intents (sells first)."""
    tgt_shares: dict[str, float] = {}
    for sym, w in targets.items():
        px = marks.get(sym)
        if px is None or px <= 0:
            continue
        tgt_shares[sym] = (equity * w) / px

    intents: list[dict] = []
    for sym, qty in list(positions.items()):
        want = tgt_shares.get(sym, 0.0)
        if qty > want + 1e-8:
            sell = qty - want
            if sell * marks.get(sym, 0) >= min_notional or want <= 0:
                intents.append({"symbol": sym, "side": "sell", "qty": float(sell)})
    for sym, want in tgt_shares.items():
        have = positions.get(sym, 0.0)
        if want > have + 1e-8:
            buy = want - have
            if buy * marks.get(sym, 0) >= min_notional:
                intents.append({"symbol": sym, "side": "buy", "qty": float(buy)})
    return intents


def weights_with_cash(targets: dict[str, float]) -> dict[str, float]:
    """Explicit Cash + risky weights summing to 1."""
    risky = {str(k).upper(): float(v) for k, v in targets.items() if float(v) > 0}
    invested = float(sum(risky.values()))
    cash = max(0.0, 1.0 - invested)
    out = {"CASH": cash}
    out.update(risky)
    return out
