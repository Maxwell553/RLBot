"""Live daily OHLC for GeneralEquity1 signals (Yahoo, not frozen bars.db)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ge1_strategy import PANEL_SYMBOLS

LIVE_DIR = Path(__file__).resolve().parent
REPO_ROOT = LIVE_DIR.parent


def cache_dir() -> Path:
    exec_dir = REPO_ROOT / "execution" / "live_trader"
    exec_dir.mkdir(parents=True, exist_ok=True)
    return exec_dir


def load_live_panel(
    *,
    force_refresh: bool = True,
    end: date | None = None,
) -> tuple[list[date], dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    """Prefer the repo Yahoo helper so LiveTrader matches /ops/forward marks.

    Defaults to a fresh Yahoo pull. The on-disk NPZ is only a fallback; pass
    ``force_refresh=False`` for tests or when the operator passed ``--no-refresh-data``.
    """
    try:
        from rlbot.prod_return_alpha import fetch_daily_ohlc

        return fetch_daily_ohlc(
            list(PANEL_SYMBOLS),
            end=end,
            cache_dir=cache_dir(),
            force_refresh=force_refresh,
        )
    except Exception:
        return _yf_fetch(end=end or date.today(), force_refresh=force_refresh)


def _yf_fetch(
    *,
    end: date,
    force_refresh: bool,
) -> tuple[list[date], dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    del force_refresh
    import pandas as pd
    import yfinance as yf

    start = date(2009, 1, 1)
    frames: dict[str, Any] = {}
    for sym in PANEL_SYMBOLS:
        raw = yf.download(
            sym,
            start=str(start),
            end=str(end + timedelta(days=1)),
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
        raise RuntimeError("yfinance returned no SPY bars")
    idx = frames["SPY"].dropna(subset=["Close"]).index
    dates = [pd.Timestamp(t).date() for t in idx]
    closes: dict[str, np.ndarray] = {}
    ohlc: dict[str, tuple[np.ndarray, ...]] = {}
    for sym in PANEL_SYMBOLS:
        if sym not in frames:
            continue
        block = frames[sym].reindex(idx).ffill()
        o = block["Open"].to_numpy(dtype=np.float64)
        h = block["High"].to_numpy(dtype=np.float64)
        l = block["Low"].to_numpy(dtype=np.float64)
        c = block["Close"].to_numpy(dtype=np.float64)
        closes[sym] = c
        ohlc[sym] = (o, h, l, c)
    return dates, closes, ohlc


def panel_to_px(
    dates: list[date],
    closes: dict[str, np.ndarray],
    ohlc: dict[str, tuple[np.ndarray, ...]],
    as_of: date | None = None,
) -> tuple[
    list[date],
    dict[str, np.ndarray],
    dict[str, tuple[np.ndarray, ...]],
    dict[str, float],
]:
    day = as_of or dates[-1]
    keep = [i for i, d in enumerate(dates) if d <= day]
    if not keep:
        raise ValueError(f"no bars on or before {day}")
    i = keep[-1]
    cut = i + 1
    panel_dates = dates[:cut]
    px = {k: np.asarray(v[:cut], dtype=np.float64) for k, v in closes.items()}
    panel_ohlc = {
        k: tuple(np.asarray(x[:cut], dtype=np.float64) for x in tup)
        for k, tup in ohlc.items()
    }
    marks = {
        k: float(v[i])
        for k, v in px.items()
        if np.isfinite(v[i]) and float(v[i]) > 0
    }
    return panel_dates, px, panel_ohlc, marks
