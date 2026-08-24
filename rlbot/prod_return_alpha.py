"""Yahoo daily OHLC helpers for GeneralEquity1 forward MTM.

Signals come from the locked ``GeneralEquity1/`` pack via
``rlbot.pack_general_equity1``. This module only fetches prices and calendar
masks for paper fills / live 5m marks.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlbot.pack_general_equity1 import (
    PAPER_RUN_ID,
    STRATEGY_ID,
    to_yahoo_symbol,
)
from rlbot.run_artifacts import PROJECT_ROOT

SYMBOLS = ("SPY", "QQQ", "TQQQ", "BIL", "GLD", "TLT")
LEGACY_PAPER_RUN_IDS = frozenset(
    {"GENERAL_EQUITY", "PROD_RETURN_ALPHA", "FINALMODEL"}
)


def week_end_mask(dates: list[date]) -> np.ndarray:
    n = len(dates)
    m = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if dates[i].isocalendar()[1] != dates[i - 1].isocalendar()[1]:
            m[i - 1] = True
    if n:
        m[-1] = True
    return m


def month_end_mask(dates: list[date]) -> np.ndarray:
    n = len(dates)
    m = np.zeros(n, dtype=np.bool_)
    for i in range(n - 1):
        if dates[i + 1].month != dates[i].month:
            m[i] = True
    if n:
        m[-1] = True
    return m


def session_rebalance_flags(
    dates: list[date],
    i: int,
    *,
    calendar_today: date | None = None,
) -> tuple[bool, bool]:
    """Live week-end / month-end. Unlike the backtest masks, does not force the tip True."""
    n = len(dates)
    if i < 0 or i >= n:
        return False, False
    d = dates[i]
    if i + 1 < n:
        week_end = dates[i].isocalendar()[1] != dates[i + 1].isocalendar()[1]
        month_end = dates[i].month != dates[i + 1].month
        return bool(week_end), bool(month_end)
    week_end = d.weekday() == 4
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    month_end = nxt.month != d.month
    if calendar_today is None or calendar_today <= d:
        return bool(week_end), bool(month_end)
    lag = (calendar_today - d).days
    if lag <= 0 or lag > 3 or calendar_today.weekday() >= 5:
        return bool(week_end), bool(month_end)
    same_week = (
        d.isocalendar()[0] == calendar_today.isocalendar()[0]
        and d.isocalendar()[1] == calendar_today.isocalendar()[1]
    )
    if same_week and calendar_today.weekday() == 4:
        week_end = True
    nxt2 = calendar_today + timedelta(days=1)
    while nxt2.weekday() >= 5:
        nxt2 += timedelta(days=1)
    if nxt2.month != calendar_today.month:
        month_end = True
    return bool(week_end), bool(month_end)


def fetch_daily_ohlc(
    symbols: list[str] | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> tuple[list[date], dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    """Daily adjusted OHLC via yfinance. Returns (dates, close_panel, ohlc_by_sym)."""
    import yfinance as yf

    symbols = list(symbols) if symbols is not None else list(SYMBOLS)
    start_d = start or date(2009, 1, 1)
    end_d = end or date.today()
    cache_dir = cache_dir or (PROJECT_ROOT / "execution" / "paper_general_equity1")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "daily_ohlc.npz"

    want = [s.upper() for s in symbols]
    cached_dates: list[date] | None = None
    cached_closes: dict[str, np.ndarray] = {}
    cached_ohlc: dict[str, tuple[np.ndarray, ...]] = {}
    if cache_path.is_file() and not force_refresh:
        try:
            blob = np.load(cache_path, allow_pickle=True)
            cached_dates = [date.fromisoformat(str(x)[:10]) for x in blob["dates"].tolist()]
            for sym in want:
                if f"{sym}_close" in blob.files:
                    o = np.asarray(blob[f"{sym}_open"], dtype=np.float64)
                    h = np.asarray(blob[f"{sym}_high"], dtype=np.float64)
                    l = np.asarray(blob[f"{sym}_low"], dtype=np.float64)
                    c = np.asarray(blob[f"{sym}_close"], dtype=np.float64)
                    cached_closes[sym] = c
                    cached_ohlc[sym] = (o, h, l, c)
        except Exception:  # noqa: BLE001
            cached_dates = None

    tip_stale = True
    if cached_dates:
        tip_stale = cached_dates[-1] < (end_d - timedelta(days=3))

    if not cached_dates or "SPY" not in cached_closes or tip_stale or force_refresh:
        frames: dict[str, pd.DataFrame] = {}
        for sym in want:
            raw = yf.download(
                to_yahoo_symbol(sym),
                start=str(start_d),
                end=str(end_d + timedelta(days=1)),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            frames[sym] = raw[["Open", "High", "Low", "Close"]].copy()
        if "SPY" not in frames:
            raise RuntimeError("yfinance returned no SPY bars for GeneralEquity1")
        idx = frames["SPY"].dropna(subset=["Close"]).index
        dates = [pd.Timestamp(t).date() for t in idx]
        closes: dict[str, np.ndarray] = {}
        ohlc: dict[str, tuple[np.ndarray, ...]] = {}
        payload: dict[str, Any] = {"dates": np.asarray([str(d) for d in dates])}
        for sym in want:
            if sym not in frames:
                continue
            block = frames[sym].reindex(idx).ffill()
            o = block["Open"].to_numpy(dtype=np.float64)
            h = block["High"].to_numpy(dtype=np.float64)
            l = block["Low"].to_numpy(dtype=np.float64)
            c = block["Close"].to_numpy(dtype=np.float64)
            closes[sym] = c
            ohlc[sym] = (o, h, l, c)
            payload[f"{sym}_open"] = o
            payload[f"{sym}_high"] = h
            payload[f"{sym}_low"] = l
            payload[f"{sym}_close"] = c
        np.savez_compressed(cache_path, **payload)
        return dates, closes, ohlc

    assert cached_dates is not None
    return cached_dates, cached_closes, cached_ohlc


def weights_with_cash(weights: dict[str, float]) -> dict[str, float]:
    """Ensure a CASH key; residual → CASH. Does not double-count existing CASH."""
    out = {str(k).upper(): float(v) for k, v in weights.items() if float(v) > 0}
    invested = sum(v for k, v in out.items() if k != "CASH")
    if "CASH" in out:
        tot = sum(out.values())
        if tot <= 1e-12:
            return {"CASH": 1.0}
        return {k: v / tot for k, v in out.items()}
    cash = max(0.0, 1.0 - invested)
    if cash > 1e-12:
        out["CASH"] = cash
    tot = sum(out.values())
    if tot <= 1e-12:
        return {"CASH": 1.0}
    return {k: v / tot for k, v in out.items()}
