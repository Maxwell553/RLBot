"""Live daily OHLC for CoreEquity signals (IBKR by default; Yahoo fallback)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ce_strategy import PANEL_SYMBOLS

LIVE_DIR = Path(__file__).resolve().parent
REPO_ROOT = LIVE_DIR.parent
PANEL_START = date(2009, 1, 1)

_LAST_SOURCE = "yahoo"


def cache_dir() -> Path:
    exec_dir = REPO_ROOT / "execution" / "live_trader"
    exec_dir.mkdir(parents=True, exist_ok=True)
    return exec_dir


def last_panel_source() -> str:
    return str(_LAST_SOURCE)


def _set_source(source: str) -> None:
    global _LAST_SOURCE
    _LAST_SOURCE = str(source)


def _bar_date(raw: Any) -> date:
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if hasattr(raw, "date") and callable(raw.date):
        try:
            d = raw.date()
            if isinstance(d, date):
                return d
        except Exception:  # noqa: BLE001
            pass
    s = str(raw).strip().replace(".", "-").replace("/", "-")
    return date.fromisoformat(s[:10])


def panel_from_ohlc_frames(
    frames: dict[str, dict[date, tuple[float, float, float, float]]],
    *,
    symbols: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[date], dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    """Align per-symbol OHLC maps onto the SPY session calendar (ffill)."""
    want = [str(s).upper() for s in (symbols or PANEL_SYMBOLS)]
    spy = frames.get("SPY") or {}
    dates = sorted(d for d, px in spy.items() if px[3] > 0 and np.isfinite(px[3]))
    if not dates:
        raise RuntimeError("no SPY bars in live panel")
    closes: dict[str, np.ndarray] = {}
    ohlc: dict[str, tuple[np.ndarray, ...]] = {}
    n = len(dates)
    for sym in want:
        block = frames.get(sym) or {}
        o = np.full(n, np.nan)
        h = np.full(n, np.nan)
        l = np.full(n, np.nan)
        c = np.full(n, np.nan)
        last: tuple[float, float, float, float] | None = None
        for i, d in enumerate(dates):
            row = block.get(d)
            if row is not None and np.isfinite(row[3]) and float(row[3]) > 0:
                last = row
            if last is None:
                continue
            o[i], h[i], l[i], c[i] = last
        closes[sym] = c
        ohlc[sym] = (o, h, l, c)
    return dates, closes, ohlc


def _cache_path(source: str) -> Path:
    return cache_dir() / f"daily_ohlc_{source}.npz"


def _save_cache(
    source: str,
    dates: list[date],
    closes: dict[str, np.ndarray],
    ohlc: dict[str, tuple[np.ndarray, ...]],
) -> None:
    payload: dict[str, Any] = {"dates": np.asarray([str(d) for d in dates])}
    for sym, c in closes.items():
        o, h, l, _c = ohlc[sym]
        payload[f"{sym}_open"] = o
        payload[f"{sym}_high"] = h
        payload[f"{sym}_low"] = l
        payload[f"{sym}_close"] = c
    np.savez_compressed(_cache_path(source), **payload)


def _load_cache(
    source: str,
    *,
    end: date | None,
) -> tuple[list[date], dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]] | None:
    path = _cache_path(source)
    if not path.is_file():
        return None
    try:
        blob = np.load(path, allow_pickle=True)
        dates = [date.fromisoformat(str(x)[:10]) for x in blob["dates"].tolist()]
        closes: dict[str, np.ndarray] = {}
        ohlc: dict[str, tuple[np.ndarray, ...]] = {}
        for sym in PANEL_SYMBOLS:
            if f"{sym}_close" not in blob.files:
                continue
            o = np.asarray(blob[f"{sym}_open"], dtype=np.float64)
            h = np.asarray(blob[f"{sym}_high"], dtype=np.float64)
            l = np.asarray(blob[f"{sym}_low"], dtype=np.float64)
            c = np.asarray(blob[f"{sym}_close"], dtype=np.float64)
            closes[sym] = c
            ohlc[sym] = (o, h, l, c)
        if "SPY" not in closes or not dates:
            return None
        end_d = end or date.today()
        if dates[-1] < (end_d - timedelta(days=4)):
            return None
        return dates, closes, ohlc
    except Exception:  # noqa: BLE001
        return None


def _ibkr_fetch(
    broker: Any,
    *,
    end: date | None,
    force_refresh: bool,
) -> tuple[list[date], dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    end_d = end or date.today()
    if not force_refresh:
        cached = _load_cache("ibkr", end=end_d)
        if cached is not None:
            return cached
    frames = broker.historical_daily_ohlc(list(PANEL_SYMBOLS), start=PANEL_START, end=end_d)
    dates, closes, ohlc = panel_from_ohlc_frames(frames, symbols=PANEL_SYMBOLS)
    _save_cache("ibkr", dates, closes, ohlc)
    return dates, closes, ohlc


def load_live_panel(
    *,
    force_refresh: bool = True,
    end: date | None = None,
    broker: Any | None = None,
    cfg: Any | None = None,
) -> tuple[list[date], dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    """IBKR daily bars by default. Yahoo only if TWS/historical fails and fallback is on.

    Pass ``force_refresh=False`` (``--no-refresh-data``) to use the on-disk NPZ.
    """
    from config import load_config

    cfg = cfg or load_config()
    source = str(getattr(cfg, "data_source", None) or "ibkr").strip().lower()
    yahoo_fallback = bool(getattr(cfg, "yahoo_fallback", True))
    own_broker = False
    if source == "ibkr":
        try:
            if broker is None:
                from ibkr_client import IBKRBroker

                broker = IBKRBroker(cfg)
                broker.connect()
                own_broker = True
            out = _ibkr_fetch(broker, end=end, force_refresh=force_refresh)
            _set_source("ibkr")
            return out
        except Exception:
            if not yahoo_fallback:
                raise
        finally:
            if own_broker and broker is not None:
                try:
                    broker.disconnect()
                except Exception:  # noqa: BLE001
                    pass
    out = _yahoo_fetch(end=end or date.today(), force_refresh=force_refresh)
    _set_source("yahoo")
    return out


def _yahoo_fetch(
    *,
    end: date,
    force_refresh: bool,
) -> tuple[list[date], dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    if not force_refresh:
        cached = _load_cache("yahoo", end=end)
        if cached is not None:
            return cached
    try:
        from rlbot.prod_return_alpha import fetch_daily_ohlc

        dates, closes, ohlc = fetch_daily_ohlc(
            list(PANEL_SYMBOLS),
            end=end,
            cache_dir=cache_dir(),
            force_refresh=force_refresh,
        )
        _save_cache("yahoo", dates, closes, ohlc)
        return dates, closes, ohlc
    except Exception:
        return _yf_fetch(end=end, force_refresh=force_refresh)


def _yf_fetch(
    *,
    end: date,
    force_refresh: bool,
) -> tuple[list[date], dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    del force_refresh
    import pandas as pd
    import yfinance as yf

    frames: dict[str, dict[date, tuple[float, float, float, float]]] = {}
    for sym in PANEL_SYMBOLS:
        raw = yf.download(
            sym,
            start=str(PANEL_START),
            end=str(end + timedelta(days=1)),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        block: dict[date, tuple[float, float, float, float]] = {}
        for ts, row in raw[["Open", "High", "Low", "Close"]].iterrows():
            d = pd.Timestamp(ts).date()
            o, h, l, c = (float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]))
            if c > 0 and np.isfinite(c):
                block[d] = (o, h, l, c)
        frames[sym] = block
    dates, closes, ohlc = panel_from_ohlc_frames(frames, symbols=PANEL_SYMBOLS)
    _save_cache("yahoo", dates, closes, ohlc)
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
