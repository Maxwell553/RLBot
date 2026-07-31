"""Live forward-mark refresh (torch-free): 5m NAV candles for model / EW / SPY.

The ops dashboard polls ``/api/forward``; that endpoint calls
``refresh_forward_mark_live`` so a new 5-minute candle (and the forming bar)
updates from a throttled yfinance pull without re-running RecurrentPPO.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlbot.forward_mark import (
    build_forward_mark_payload,
    load_forward_mark,
    resolve_active_forward_run_id,
    write_forward_mark,
)
from rlbot.rl_config import get_config, load_config, set_config
from rlbot.run_artifacts import PROJECT_ROOT

LIVE_STAMP_NAME = "forward_live_stamp.json"
# Companion RL deploy shown alongside FINALMODEL on /ops/forward.
RL_LIVE_RUN_ID = "LIVE_MODEL"
# Candle grid is 5m; refresh Yahoo about once per new bar.
DEFAULT_MIN_REFRESH_SECONDS = 300
BAR_INTERVAL = "5m"
# Approx US cash-session 5-minute bars (09:30–16:00 ET → 78).
BARS_PER_TRADING_DAY = 78
# yfinance 5m history is capped (~60d); fetch this much lookback.
INTRADAY_LOOKBACK_DAYS = 59


def _session_date_today() -> str:
    """US/Eastern trading calendar date (naive YYYY-MM-DD)."""
    return str(pd.Timestamp.now(tz="America/New_York").date())


def _stamp_path(root: Path | None = None) -> Path:
    base = root or PROJECT_ROOT
    return base / "execution" / LIVE_STAMP_NAME


def _read_stamp(root: Path | None = None) -> dict[str, Any]:
    path = _stamp_path(root)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_stamp(payload: dict[str, Any], root: Path | None = None) -> None:
    path = _stamp_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_weight_key(label: str) -> str:
    return str(label).strip().upper().replace(" ", "")


def _weight_vector(
    weights: dict[str, float] | None,
    asset_labels: list[str],
) -> np.ndarray:
    """Map a weight dict onto ``asset_labels`` (Cash + risky). Missing → 0."""
    n = len(asset_labels)
    out = np.zeros(n, dtype=np.float64)
    if not weights:
        out[0] = 1.0
        return out
    by_key = {_normalize_weight_key(k): float(v) for k, v in weights.items()}
    for i, lab in enumerate(asset_labels):
        out[i] = by_key.get(_normalize_weight_key(lab), 0.0)
    s = float(out.sum())
    if s <= 1e-12:
        out[0] = 1.0
        return out
    return out / s


def _latest_shadow_weights(run_id: str) -> dict[str, float] | None:
    path = PROJECT_ROOT / "execution" / f"shadow_ledger_{run_id}.jsonl"
    if not path.is_file():
        return None
    last: dict[str, float] | None = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tw = rec.get("target_weights")
            if isinstance(tw, dict) and tw:
                last = {str(k): float(v) for k, v in tw.items()}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return last
    return last


def _is_equity_stock_book(
    weights: dict[str, float] | None,
    *,
    run_id: str,
    cfg_asset_keys: list[str],
) -> bool:
    """True when the live book is individual equities (PIT momentum), not the RL sleeve."""
    if str(run_id).upper() == "FINALMODEL":
        return True
    if not weights:
        return False
    cfg = {_normalize_weight_key(k) for k in cfg_asset_keys}
    cfg.add("CASH")
    keys = {_normalize_weight_key(k) for k in weights}
    risky = keys - {"CASH"}
    if not risky:
        return False
    overlap = len(risky & cfg) / len(risky)
    return overlap < 0.5


def _stock_symbols_from_weights(weights: dict[str, float]) -> dict[str, str]:
    """Map display label → Yahoo symbol for equity books (BRK.B → BRK-B)."""
    from rlbot.pit_momentum import to_yahoo_symbol

    out: dict[str, str] = {}
    for k, v in weights.items():
        lab = str(k).strip()
        if not lab or lab.upper() == "CASH" or float(v) <= 0:
            continue
        out[lab.upper() if "." not in lab else lab] = to_yahoo_symbol(lab)
    # Prefer uppercase keys consistently.
    return {str(k).upper(): v for k, v in out.items()}


def _cash_yield_per_bar(cash_daily_yield: float) -> float:
    """Scale config daily cash yield onto one 5m RTH bar."""
    return float(cash_daily_yield) / float(BARS_PER_TRADING_DAY)


def _nav_from_weights(
    closes: np.ndarray,
    weight_rows: list[np.ndarray],
    *,
    initial_cash: float,
    cash_daily_yield: float = 0.0,
) -> np.ndarray:
    """Close-to-close MTM (legacy daily helper; kept for unit tests)."""
    t_bars, n_assets = closes.shape
    if t_bars < 1:
        return np.asarray([initial_cash], dtype=np.float64)
    # Interpret cash_daily_yield as per-bar when used with intraday closes.
    y = float(cash_daily_yield)
    nav = np.empty(t_bars, dtype=np.float64)
    nav[0] = float(initial_cash)
    for t in range(t_bars - 1):
        w = weight_rows[min(t, len(weight_rows) - 1)]
        if w.shape[0] != n_assets + 1:
            ww = np.zeros(n_assets + 1, dtype=np.float64)
            m = min(w.shape[0], ww.shape[0])
            ww[:m] = w[:m]
            w = ww
        cash_w = float(w[0])
        risky = w[1:]
        rets = closes[t + 1] / np.maximum(closes[t], 1e-12) - 1.0
        port = float(np.dot(risky, rets)) + cash_w * y
        nav[t + 1] = nav[t] * (1.0 + port)
    return nav


def _ew_nav(closes: np.ndarray, initial_cash: float) -> np.ndarray:
    t_bars, n = closes.shape
    nav = np.empty(t_bars, dtype=np.float64)
    nav[0] = float(initial_cash)
    if n <= 0:
        nav[:] = float(initial_cash)
        return nav
    w = np.full(n, 1.0 / n, dtype=np.float64)
    for t in range(t_bars - 1):
        rets = closes[t + 1] / np.maximum(closes[t], 1e-12) - 1.0
        nav[t + 1] = nav[t] * (1.0 + float(np.dot(w, rets)))
    return nav


def _spy_nav(spy_close: np.ndarray, initial_cash: float) -> np.ndarray:
    s0 = max(float(spy_close[0]), 1e-12)
    return (np.asarray(spy_close, dtype=np.float64) / s0) * float(initial_cash)


def portfolio_ohlc_candles(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    weight: np.ndarray,
    *,
    initial_cash: float,
    cash_yield_per_bar: float = 0.0,
) -> np.ndarray:
    """Build NAV OHLC candles from aligned asset OHLC and a fixed weight vector.

    ``weight`` is Cash + N risky (sums to 1). Within each bar, O/H/L/C returns are
    taken vs the previous bar's asset closes (bar 0 vs that bar's opens). Highs/lows
    are simultaneous-price approximations (standard for multi-asset candles).
    Returns ``[T, 4]`` with columns open, high, low, close.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if open_.ndim != 2:
        raise ValueError("open/high/low/close must be [T, N]")
    t_bars, n_assets = close.shape
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    if w.shape[0] != n_assets + 1:
        ww = np.zeros(n_assets + 1, dtype=np.float64)
        m = min(w.shape[0], ww.shape[0])
        ww[:m] = w[:m]
        s = float(ww.sum())
        w = ww / s if s > 1e-12 else np.array([1.0] + [0.0] * n_assets)
    cash_w = float(w[0])
    risky = w[1:]
    out = np.empty((t_bars, 4), dtype=np.float64)
    if t_bars < 1:
        return out

    nav_close = float(initial_cash)
    prev_px = np.asarray(open_[0], dtype=np.float64)
    y = float(cash_yield_per_bar)

    for t in range(t_bars):
        # Cash yield accrues once per bar (on H/L/C); open is mark-to-market only.
        if t == 0:
            base = np.maximum(open_[t], 1e-12)
            r_h = high[t] / base - 1.0
            r_l = low[t] / base - 1.0
            r_c = close[t] / base - 1.0
            o_nav = float(initial_cash)
            h_nav = o_nav * (1.0 + float(np.dot(risky, r_h)) + cash_w * y)
            l_nav = o_nav * (1.0 + float(np.dot(risky, r_l)) + cash_w * y)
            c_nav = o_nav * (1.0 + float(np.dot(risky, r_c)) + cash_w * y)
        else:
            base = np.maximum(prev_px, 1e-12)
            r_o = open_[t] / base - 1.0
            r_h = high[t] / base - 1.0
            r_l = low[t] / base - 1.0
            r_c = close[t] / base - 1.0
            o_nav = nav_close * (1.0 + float(np.dot(risky, r_o)))
            h_nav = nav_close * (1.0 + float(np.dot(risky, r_h)) + cash_w * y)
            l_nav = nav_close * (1.0 + float(np.dot(risky, r_l)) + cash_w * y)
            c_nav = nav_close * (1.0 + float(np.dot(risky, r_c)) + cash_w * y)
        hi = max(o_nav, h_nav, l_nav, c_nav)
        lo = min(o_nav, h_nav, l_nav, c_nav)
        out[t] = (o_nav, hi, lo, c_nav)
        nav_close = c_nav
        prev_px = close[t]
    return out


def equal_weight_ohlc_candles(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    initial_cash: float,
) -> np.ndarray:
    n = int(close.shape[1]) if close.ndim == 2 else 0
    if n <= 0:
        t = int(close.shape[0]) if close.ndim >= 1 else 0
        return np.full((t, 4), float(initial_cash), dtype=np.float64)
    w = np.concatenate([[0.0], np.full(n, 1.0 / n, dtype=np.float64)])
    return portfolio_ohlc_candles(
        open_, high, low, close, w, initial_cash=initial_cash, cash_yield_per_bar=0.0
    )


def spy_ohlc_candles(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    initial_cash: float,
) -> np.ndarray:
    """Scale a single-asset OHLC series so the first open equals ``initial_cash``."""
    o = np.asarray(open_, dtype=np.float64).reshape(-1)
    h = np.asarray(high, dtype=np.float64).reshape(-1)
    l = np.asarray(low, dtype=np.float64).reshape(-1)
    c = np.asarray(close, dtype=np.float64).reshape(-1)
    if o.size < 1:
        return np.zeros((0, 4), dtype=np.float64)
    scale = float(initial_cash) / max(float(o[0]), 1e-12)
    out = np.column_stack([o * scale, h * scale, l * scale, c * scale])
    # Enforce OHLC consistency after float scaling.
    out[:, 1] = np.maximum(out[:, 1], np.maximum(out[:, 0], out[:, 3]))
    out[:, 2] = np.minimum(out[:, 2], np.minimum(out[:, 0], out[:, 3]))
    return out


def _candles_to_payload(ts: pd.DatetimeIndex, ohlc: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, t in enumerate(ts):
        stamp = pd.Timestamp(t)
        if stamp.tzinfo is None:
            # Yahoo intraday is exchange-local; keep naive ISO without claiming UTC.
            iso = stamp.isoformat(timespec="minutes")
        else:
            iso = stamp.isoformat(timespec="minutes")
        o, h, l, c = (float(ohlc[i, j]) for j in range(4))
        rows.append({"t": iso, "o": o, "h": h, "l": l, "c": c})
    return rows


def _positions_snapshot(
    weights: dict[str, float],
    *,
    nav: float,
    labels: list[str],
    last_closes: np.ndarray | None,
    tickers: list[str],
) -> list[dict[str, Any]]:
    w = _weight_vector(weights, labels)
    rows: list[dict[str, Any]] = []
    for i, lab in enumerate(labels):
        value = float(w[i] * nav)
        price = None
        if i == 0:
            price = 1.0
        elif last_closes is not None and i - 1 < last_closes.shape[0]:
            price = float(last_closes[i - 1])
        rows.append(
            {
                "label": lab,
                "ticker": "CASH" if i == 0 else (tickers[i - 1] if i - 1 < len(tickers) else lab),
                "weight": float(w[i]),
                "value_usd": value,
                "price": price,
            }
        )
    rows.sort(
        key=lambda r: (-1 if str(r["label"]).lower() == "cash" else 0, -float(r["weight"]))
    )
    return rows


def _yahoo_chart_ohlc(
    symbol: str,
    *,
    range_key: str = "5d",
    interval: str = BAR_INTERVAL,
    timeout_s: float = 12.0,
) -> pd.DataFrame:
    """Fetch one symbol's intraday OHLC via Yahoo's chart API (no yfinance)."""
    import urllib.parse

    params = urllib.parse.urlencode(
        {"range": range_key, "interval": interval, "events": "history"}
    )
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol, safe='')}?{params}"
    )
    try:
        from curl_cffi import requests as curl_requests

        response = curl_requests.get(url, impersonate="chrome", timeout=float(timeout_s))
    except Exception:  # noqa: BLE001 - fall back to stdlib urllib
        try:
            import urllib.request

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            response = None
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not isinstance(result, dict):
            return pd.DataFrame()
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
        if not timestamps:
            return pd.DataFrame()
        idx = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="s", utc=True))
        idx = idx.tz_convert("America/New_York").tz_localize(None)
        frame = pd.DataFrame(
            {
                "Open": quote.get("open") or [None] * len(timestamps),
                "High": quote.get("high") or [None] * len(timestamps),
                "Low": quote.get("low") or [None] * len(timestamps),
                "Close": quote.get("close") or [None] * len(timestamps),
            },
            index=idx,
        )
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame.dropna(subset=["Close"], how="any")

    if getattr(response, "status_code", 200) != 200:
        return pd.DataFrame()
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not isinstance(result, dict):
        return pd.DataFrame()
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
    if not timestamps:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="s", utc=True))
    idx = idx.tz_convert("America/New_York").tz_localize(None)
    frame = pd.DataFrame(
        {
            "Open": quote.get("open") or [None] * len(timestamps),
            "High": quote.get("high") or [None] * len(timestamps),
            "Low": quote.get("low") or [None] * len(timestamps),
            "Close": quote.get("close") or [None] * len(timestamps),
        },
        index=idx,
    )
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.dropna(subset=["Close"], how="any")


def _fetch_intraday_ohlc(
    symbols: dict[str, str],
    *,
    since: str,
    spy_symbol: str = "SPY",
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aligned intraday OHLC for universe labels + SPY on the SPY bar clock.

    Uses Yahoo's chart HTTP API (curl_cffi) instead of yfinance — the latter
    routinely hangs under Desktop iCloud load and left the forward mark stuck
    on a stale ``execution/`` price cache.

    Assets that trade different sessions (FX, Nikkei, FTSE, futures) are
    reindexed onto SPY timestamps and forward-filled so a missing print carries
    the last close as a flat candle (o=h=l=c). Returns
    ``times, open[T,N], high[T,N], low[T,N], close[T,N], spy_o/h/l/c[T]``.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    labels = list(symbols.keys())
    ticker_list = list(dict.fromkeys([*symbols.values(), spy_symbol]))
    # 5d of 5m bars covers the live forward window without the monthly
    # downgrade Yahoo applies to range=max.
    range_key = "5d"
    since_ts = pd.Timestamp(since)

    frames_by_sym: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(ticker_list))) as pool:
        futs = {
            pool.submit(_yahoo_chart_ohlc, sym, range_key=range_key, interval=BAR_INTERVAL): sym
            for sym in ticker_list
        }
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                frames_by_sym[sym] = fut.result()
            except Exception:  # noqa: BLE001
                frames_by_sym[sym] = pd.DataFrame()

    spy_df = frames_by_sym.get(spy_symbol, pd.DataFrame())
    if spy_df.empty:
        raise RuntimeError(f"Yahoo chart returned no {BAR_INTERVAL} SPY rows ({spy_symbol})")
    spy_df = spy_df[spy_df.index >= since_ts]
    if spy_df.empty:
        raise RuntimeError(
            f"no {BAR_INTERVAL} SPY bars on/after {since_ts.date()} ({spy_symbol})"
        )

    clock = spy_df.index
    frames: dict[str, pd.DataFrame] = {}
    for lab, sym in symbols.items():
        df = frames_by_sym.get(sym, pd.DataFrame())
        if df.empty:
            raise RuntimeError(f"missing {BAR_INTERVAL} history for {lab} ({sym})")
        df = df[df.index >= since_ts] if len(df) else df
        close_ff = df["Close"].reindex(clock, method="ffill")
        # Prepend a short bfill only at the head if the asset lists after SPY open.
        if close_ff.isna().any():
            close_ff = close_ff.bfill()
        native = df.reindex(clock)
        aligned = pd.DataFrame(index=clock)
        aligned["Close"] = close_ff
        for col in ("Open", "High", "Low"):
            aligned[col] = native[col].where(native[col].notna(), close_ff)
        if aligned["Close"].isna().any():
            raise RuntimeError(
                f"could not align {BAR_INTERVAL} bars for {lab} ({sym}) onto SPY clock"
            )
        frames[lab] = aligned

    o = np.column_stack([frames[lab]["Open"].to_numpy(dtype=np.float64) for lab in labels])
    h = np.column_stack([frames[lab]["High"].to_numpy(dtype=np.float64) for lab in labels])
    l = np.column_stack([frames[lab]["Low"].to_numpy(dtype=np.float64) for lab in labels])
    c = np.column_stack([frames[lab]["Close"].to_numpy(dtype=np.float64) for lab in labels])
    so = spy_df["Open"].to_numpy(dtype=np.float64)
    sh = spy_df["High"].to_numpy(dtype=np.float64)
    sl = spy_df["Low"].to_numpy(dtype=np.float64)
    sc = spy_df["Close"].to_numpy(dtype=np.float64)
    return pd.DatetimeIndex(clock), o, h, l, c, so, sh, sl, sc


_fetch_30m_ohlc = _fetch_intraday_ohlc  # back-compat


def refresh_forward_mark_live(
    run_id: str | None = None,
    *,
    min_refresh_seconds: int = DEFAULT_MIN_REFRESH_SECONDS,
    force_price_refresh: bool = False,
    root: Path | None = None,
) -> dict[str, Any] | None:
    """Refresh 5m NAV candles/positions for the active (or given) LIVE run.

    - Throttled Yahoo pull for 5-minute OHLC (universe + SPY).
    - Model book: last shadow / mark weights, MTM'd into OHLC candles.
    - Benchmarks: equal-weight and SPY buy-and-hold on the same 5m grid.
    """
    rid = (run_id or "").strip() or (resolve_active_forward_run_id(root) or "")
    if not rid:
        return None

    existing = load_forward_mark(rid)
    stamp = _read_stamp(root)
    now = time.time()
    last = float(stamp.get("prices_fetched_at_unix") or 0.0)
    last_attempt = float(stamp.get("prices_attempt_at_unix") or 0.0)
    need_fetch = force_price_refresh or (now - last >= float(min_refresh_seconds))
    # After a failed/slow Yahoo pull, cool down briefly so /api/forward polls
    # (every ~30s from the UI) do not each block on a fresh download.
    price_cache = PROJECT_ROOT / "execution" / f"forward_prices_5m_{rid}.npz"
    if (
        need_fetch
        and not force_price_refresh
        and price_cache.is_file()
        and (now - last_attempt) < 60.0
    ):
        need_fetch = False

    # Always use the repo config — never touch Runs/<id>/config.yaml (iCloud hang).
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    set_config(load_config(cfg_path))
    cfg = get_config()
    initial_cash = float(
        (existing or {}).get("initial_cash") or cfg.environment.initial_cash
    )
    # Both FINALMODEL and LIVE_MODEL charts start at today's US session.
    holdout_start = _session_date_today()
    holdout_end = (existing or {}).get("holdout_end") or stamp.get("holdout_end")
    cash_yield_bar = _cash_yield_per_bar(
        float(getattr(cfg.reward, "cash_daily_yield", 0.0) or 0.0)
    )
    rl_cash_yield_bar = cash_yield_bar

    latest_w = _latest_shadow_weights(rid) or (existing or {}).get("latest_weights")
    if not isinstance(latest_w, dict) or not latest_w:
        latest_w = {"Cash": 1.0}

    rl_symbols = dict(cfg.universe.assets)
    rl_labels = list(rl_symbols.keys())
    stock_book = _is_equity_stock_book(
        latest_w, run_id=rid, cfg_asset_keys=rl_labels
    )
    stock_symbols: dict[str, str] = {}
    if stock_book:
        stock_symbols = _stock_symbols_from_weights(latest_w)
        if not stock_symbols:
            stock_symbols = {"SPY": "SPY"}
        tickers = list(stock_symbols.keys())
        labels = ["Cash"] + list(stock_symbols.keys())
        # Equity paper book: no cash yield (cash account, idle cash = 0).
        cash_yield_bar = 0.0
    else:
        tickers = list(cfg.universe.tickers)
        labels = ["Cash"] + rl_labels

    # Fetch RL sleeve always (EW-10 + LIVE_MODEL); add stock picks when needed.
    symbols: dict[str, str] = dict(rl_symbols)
    symbols.update(stock_symbols)
    live_vec = _weight_vector(latest_w, labels)
    rl_live_w = _latest_shadow_weights(RL_LIVE_RUN_ID)
    if not isinstance(rl_live_w, dict) or not rl_live_w:
        rl_mark = load_forward_mark(RL_LIVE_RUN_ID)
        rl_live_w = (rl_mark or {}).get("latest_weights") if rl_mark else None
    if not isinstance(rl_live_w, dict) or not rl_live_w:
        rl_live_w = None
    rl_labels_full = ["Cash"] + rl_labels
    rl_vec = (
        _weight_vector(rl_live_w, rl_labels_full)
        if rl_live_w is not None
        else None
    )

    fetched = False
    times: pd.DatetimeIndex
    o = h = l = c = None  # type: ignore[assignment]
    so = sh = sl = sc = None  # type: ignore[assignment]
    fetch_labels = list(symbols.keys())

    def _apply_holdout(
        t: pd.DatetimeIndex,
        oo: np.ndarray,
        hh: np.ndarray,
        ll: np.ndarray,
        cc: np.ndarray,
        so_: np.ndarray,
        sh_: np.ndarray,
        sl_: np.ndarray,
        sc_: np.ndarray,
        *,
        start: Any,
    ) -> tuple[
        pd.DatetimeIndex,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
    ]:
        """Slice to holdout start; if that empties the series, keep all bars."""
        if start:
            mask = t >= pd.Timestamp(start)
            if bool(mask.any()):
                return (
                    t[mask],
                    oo[mask],
                    hh[mask],
                    ll[mask],
                    cc[mask],
                    so_[mask],
                    sh_[mask],
                    sl_[mask],
                    sc_[mask],
                    str(pd.Timestamp(start).date()),
                )
        # No usable start (or start is past last bar): chart whatever we have.
        inferred = str(pd.Timestamp(t[0]).date()) if len(t) else str(pd.Timestamp.utcnow().date())
        return t, oo, hh, ll, cc, so_, sh_, sl_, sc_, inferred

    if need_fetch:
        # Record the attempt up front so a hung Yahoo session does not cause
        # every subsequent poll to re-enter download.
        _write_stamp(
            {
                **stamp,
                "run_id": rid,
                "prices_attempt_at_unix": now,
                "prices_attempt_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            },
            root,
        )
        since = str(holdout_start or (pd.Timestamp.utcnow() - pd.Timedelta(days=INTRADAY_LOOKBACK_DAYS)).date())
        from rlbot.forward_mark import call_with_timeout

        try:
            # Stock book + RL sleeve can be ~40 symbols; allow a longer pull.
            fetch_timeout = 45.0 if (force_price_refresh or stock_book) else 15.0
            times, o, h, l, c, so, sh, sl, sc = call_with_timeout(
                _fetch_intraday_ohlc,
                fetch_timeout,
                symbols,
                since=since,
            )
        except Exception:
            # Fall through to cache / empty handling below.
            if price_cache.is_file():
                need_fetch = False
            else:
                raise
        else:
            times, o, h, l, c, so, sh, sl, sc, holdout_start = _apply_holdout(
                times, o, h, l, c, so, sh, sl, sc, start=holdout_start
            )
            if len(times) < 1:
                raise RuntimeError(
                    f"no {BAR_INTERVAL} bars on/after holdout_start={holdout_start} for live forward"
                )
            price_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                price_cache,
                times=times.astype("datetime64[ns]").to_numpy(),
                open=o,
                high=h,
                low=l,
                close=c,
                spy_open=so,
                spy_high=sh,
                spy_low=sl,
                spy_close=sc,
                labels=np.asarray(fetch_labels),
            )
            fetched = True
            _write_stamp(
                {
                    "run_id": rid,
                    "bar_interval": BAR_INTERVAL,
                    "holdout_start": holdout_start,
                    "prices_fetched_at_unix": now,
                    "prices_fetched_at_utc": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "prices_attempt_at_unix": now,
                    "n_bars": int(len(times)),
                    "last_bar": pd.Timestamp(times[-1]).isoformat(timespec="minutes"),
                },
                root,
            )

    if not fetched and price_cache.is_file():
        blob = np.load(price_cache, allow_pickle=False)
        cached_labels = [str(x) for x in blob["labels"].tolist()] if "labels" in blob.files else []
        if cached_labels != fetch_labels:
            need_fetch = True
        else:
            times = pd.DatetimeIndex(blob["times"])
            o = np.asarray(blob["open"], dtype=np.float64)
            h = np.asarray(blob["high"], dtype=np.float64)
            l = np.asarray(blob["low"], dtype=np.float64)
            c = np.asarray(blob["close"], dtype=np.float64)
            so = np.asarray(blob["spy_open"], dtype=np.float64)
            sh = np.asarray(blob["spy_high"], dtype=np.float64)
            sl = np.asarray(blob["spy_low"], dtype=np.float64)
            sc = np.asarray(blob["spy_close"], dtype=np.float64)
            times, o, h, l, c, so, sh, sl, sc, holdout_start = _apply_holdout(
                times, o, h, l, c, so, sh, sl, sc, start=holdout_start
            )
    if (not fetched) and (need_fetch or not price_cache.is_file() or o is None):
        # Cold start / stale label set — fetch once.
        since = str(holdout_start)
        from rlbot.forward_mark import call_with_timeout

        times, o, h, l, c, so, sh, sl, sc = call_with_timeout(
            _fetch_intraday_ohlc,
            45.0 if (force_price_refresh or stock_book) else 20.0,
            symbols,
            since=since,
        )
        times, o, h, l, c, so, sh, sl, sc, holdout_start = _apply_holdout(
            times, o, h, l, c, so, sh, sl, sc, start=holdout_start
        )
        if len(times) < 1:
            return existing
        price_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            price_cache,
            times=times.astype("datetime64[ns]").to_numpy(),
            open=o,
            high=h,
            low=l,
            close=c,
            spy_open=so,
            spy_high=sh,
            spy_low=sl,
            spy_close=sc,
            labels=np.asarray(fetch_labels),
        )
        fetched = True
        _write_stamp(
            {
                "run_id": rid,
                "bar_interval": BAR_INTERVAL,
                "holdout_start": holdout_start,
                "prices_fetched_at_unix": now,
                "prices_fetched_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "prices_attempt_at_unix": now,
                "n_bars": int(len(times)),
                "last_bar": pd.Timestamp(times[-1]).isoformat(timespec="minutes"),
            },
            root,
        )

    if o is None or len(times) < 1:
        return existing

    lab_index = {lab: i for i, lab in enumerate(fetch_labels)}

    def _cols(labs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = [lab_index[x] for x in labs if x in lab_index]
        if not idx:
            empty = np.zeros((len(times), 0), dtype=np.float64)
            return empty, empty, empty, empty
        return o[:, idx], h[:, idx], l[:, idx], c[:, idx]

    if stock_book:
        book_labs = list(stock_symbols.keys())
        o_book, h_book, l_book, c_book = _cols(book_labs)
        model_ohlc = portfolio_ohlc_candles(
            o_book,
            h_book,
            l_book,
            c_book,
            live_vec,
            initial_cash=initial_cash,
            cash_yield_per_bar=cash_yield_bar,
        )
        last_closes_book = c_book[-1] if c_book.shape[1] else None
    else:
        model_ohlc = portfolio_ohlc_candles(
            o, h, l, c, live_vec, initial_cash=initial_cash, cash_yield_per_bar=cash_yield_bar
        )
        last_closes_book = c[-1]

    # Equal-weight is always the original N-asset research sleeve (not stock picks).
    o_rl, h_rl, l_rl, c_rl = _cols(rl_labels)
    ew_ohlc = equal_weight_ohlc_candles(o_rl, h_rl, l_rl, c_rl, initial_cash=initial_cash)
    spy_ohlc = spy_ohlc_candles(so, sh, sl, sc, initial_cash=initial_cash)

    live_model_ohlc = None
    if rl_vec is not None and o_rl.shape[1] == len(rl_labels):
        live_model_ohlc = portfolio_ohlc_candles(
            o_rl,
            h_rl,
            l_rl,
            c_rl,
            rl_vec,
            initial_cash=initial_cash,
            cash_yield_per_bar=rl_cash_yield_bar,
        )

    nav_model = model_ohlc[:, 3]
    nav_ew = ew_ohlc[:, 3]
    nav_spy = spy_ohlc[:, 3]
    nav_live = live_model_ohlc[:, 3] if live_model_ohlc is not None else None

    w_mat = np.tile(live_vec, (len(times), 1))
    candles_payload: dict[str, Any] = {
        "model": _candles_to_payload(times, model_ohlc),
        "equal_weight": _candles_to_payload(times, ew_ohlc),
        "spy": _candles_to_payload(times, spy_ohlc),
    }
    if live_model_ohlc is not None:
        candles_payload["live_model"] = _candles_to_payload(times, live_model_ohlc)

    note = (
        "FINALMODEL PIT momentum + LIVE_MODEL RL sleeve on today's session. "
        "Equal-weight is the original 10-asset research universe (not stock picks). "
        if stock_book
        else "Live 5-minute NAV candles on the SPY session clock from latest target "
        "weights + Yahoo OHLC (model / EW / SPY). Foreign-session assets carry "
        "the last print forward. "
    ) + f"Prices refresh about every {max(min_refresh_seconds, 60) // 60} min."

    payload = build_forward_mark_payload(
        run_id=rid,
        checkpoint_label=str(
            (existing or {}).get("checkpoint_label")
            or ("locked" if stock_book else "best")
        ),
        dates=times,
        nav_model=nav_model,
        nav_spy=nav_spy,
        nav_ew=nav_ew,
        weights=w_mat,
        asset_labels=labels,
        initial_cash=initial_cash,
        holdout_start=str(holdout_start) if holdout_start else None,
        holdout_end=str(holdout_end) if holdout_end else None,
        note=note,
        bar_interval=BAR_INTERVAL,
        timestamps=[pd.Timestamp(t).isoformat(timespec="minutes") for t in times],
        candles=candles_payload,
        bars_per_year=BARS_PER_TRADING_DAY * 252,
        nav_live_model=nav_live,
    )
    positions = _positions_snapshot(
        latest_w,
        nav=float(nav_model[-1]),
        labels=labels,
        last_closes=last_closes_book,
        tickers=tickers,
    )
    payload["positions"] = positions
    payload["companion_run_id"] = RL_LIVE_RUN_ID if nav_live is not None else None
    payload["live"] = {
        "prices_refreshed": fetched,
        "as_of_bar": pd.Timestamp(times[-1]).isoformat(timespec="minutes"),
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_refresh_seconds": int(min_refresh_seconds),
        "bar_interval": BAR_INTERVAL,
        "source": "yfinance_5m",
        "session_start": str(holdout_start),
    }
    payload["latest_weights"] = {
        labels[i]: float(live_vec[i]) for i in range(len(labels))
    }
    write_forward_mark(payload)
    return payload
