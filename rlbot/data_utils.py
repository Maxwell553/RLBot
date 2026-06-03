"""
Historical OHLCV fetch and alignment for multi-asset portfolio RL.

Tradeable symbols come from ``config.yaml`` → ``universe.assets`` (via
``get_active_symbols()``). Macro context (DXY, TNX, VIX, HY OAS) is
observation-only.

Daily bars are outer-joined across assets with short forward-fill for
holiday gaps. Pre-listing rows are dropped. The panel starts when all
configured assets have real quotes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

def get_active_symbols() -> Dict[str, str]:
    """Tradeable universe label → yfinance symbol from loaded config."""
    from rlbot.rl_config import get_config

    return dict(get_config().universe.assets)


def resolve_tickers(tickers: Sequence[str] | None = None) -> List[str]:
    """Active ticker order: explicit list or config universe."""
    if tickers is not None:
        return list(tickers)
    from rlbot.rl_config import get_config

    return list(get_config().universe.tickers)


def resolve_panel_tickers(
    manifest: dict | None = None,
    cache_tickers: Sequence[str] | None = None,
) -> List[str]:
    """Ticker order for backtest: training manifest, cache array, then config."""
    if manifest:
        uni = manifest.get("universe")
        if isinstance(uni, dict) and uni.get("tickers"):
            return [str(t) for t in uni["tickers"]]
    if cache_tickers is not None:
        return list(cache_tickers)
    return resolve_tickers()


def select_tradeable_columns(
    ohlcv: np.ndarray,
    rsi: np.ndarray,
    macd: np.ndarray,
    fracdiff: np.ndarray,
    trend: np.ndarray,
    panel_tickers: Sequence[str],
    universe_tickers: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Restrict asset-axis arrays to ``universe_tickers`` (subset of ``panel_tickers``, same order)."""
    panel = list(panel_tickers)
    want = list(universe_tickers)
    if panel == want:
        return ohlcv, rsi, macd, fracdiff, trend, want
    try:
        indices = [panel.index(t) for t in want]
    except ValueError as e:
        missing = [t for t in want if t not in panel]
        raise ValueError(
            f"Data cache is missing universe ticker(s) {missing}; "
            f"cache has {panel}. Run with --refresh-data after changing the universe."
        ) from e
    return (
        ohlcv[:, indices],
        rsi[:, indices],
        macd[:, indices],
        fracdiff[:, indices],
        trend[:, indices],
        want,
    )


def _benchmark_ticker_label() -> str:
    from rlbot.rl_config import get_config

    return get_config().universe.benchmark


def benchmark_ohlcv_index(tickers: Sequence[str] | None = None) -> int:
    """OHLCV asset axis index for the benchmark sleeve (not the policy action index)."""
    active = resolve_tickers(tickers)
    benchmark = _benchmark_ticker_label()
    if benchmark not in active:
        raise KeyError(f"Benchmark {benchmark!r} missing from tickers: {active}")
    return active.index(benchmark)


# Macro context features — not tradeable, observation-only
MACRO_SYMBOLS: Dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "TNX": "^TNX",
    "VIX": "^VIX",
}
# ICE BofA US HY OAS (FRED); full history via HYG/IEF proxy + FRED calibration
HY_OAS_FRED_ID = "BAMLH0A0HYM2"
HY_OAS_PROXY_YF = "HYG"  # high-yield ETF vs IEF (BOND10Y) for pre-FRED panel
MACRO_TICKERS: List[str] = list(MACRO_SYMBOLS.keys()) + ["HY_OAS"]
N_MACRO = len(MACRO_TICKERS)
MACRO_VIX_INDEX = MACRO_TICKERS.index("VIX")  # DXY=0, TNX=1, VIX=2, HY_OAS=3

TIMEFRAME = "1d"
FFILL_LIMIT = 5  # forward-fill up to 5 business days for holiday gaps

# Fractional differentiation (Marcos López de Prado): stationary series with memory
DEFAULT_FRACDIFF_D = 0.4


def fracdiff_weights(d: float, weight_eps: float = 1e-14) -> np.ndarray:
    """Weights w_k for Δ^d x_t = sum_k w_k x_{t-k}; w_0=1, w_k = prod_{i=0}^{k-1} (i-d)/(i+1)."""
    w_list = [1.0]
    k = 1
    while True:
        w_k = w_list[-1] * ((k - 1) - d) / k
        w_list.append(w_k)
        if abs(w_k) < weight_eps:
            break
        k += 1
        if k > 50_000:
            break
    return np.array(w_list, dtype=np.float64)


def fracdiff_series_1d(x: np.ndarray, d: float) -> np.ndarray:
    """Causal fractional differentiation via 1D convolution (same as nested loops)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    w = fracdiff_weights(d)
    return np.convolve(x, w, mode="full")[:n]


def compute_fracdiff_panel(
    ohlcv: np.ndarray,
    macro: np.ndarray,
    d: float = DEFAULT_FRACDIFF_D,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fracdiff of log(close) per asset and log(macro) per series.

    Causal: each ``fracdiff[t]`` uses only ``log_price[t-k]`` for ``k >= 0`` (no future bars).
    This is **not** look-ahead; any sharp backtest move at period end is from the policy or
    segment boundaries, not from peeking at future prices in features.
    """
    t, n_assets, _ = ohlcv.shape
    fd = np.zeros((t, n_assets), dtype=np.float64)
    for j in range(n_assets):
        logp = np.log(np.maximum(ohlcv[:, j, 3], 1e-12))
        fd[:, j] = fracdiff_series_1d(logp, d)
    n_m = macro.shape[1]
    fd_m = np.zeros((t, n_m), dtype=np.float64)
    for j in range(n_m):
        logp = np.log(np.maximum(macro[:, j], 1e-12))
        fd_m[:, j] = fracdiff_series_1d(logp, d)
    return fd, fd_m


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_line(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def fetch_fred_daily_series(
    series_id: str,
    since: str,
    until: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch a FRED daily series as ``date`` + ``value`` (graph CSV; recent window only)."""
    import io
    import urllib.error
    import urllib.request

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        with urllib.request.urlopen(url, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"FRED fetch failed for {series_id}: {e}") from e

    df = pd.read_csv(io.StringIO(raw))
    if df.shape[1] < 2:
        raise RuntimeError(f"Unexpected FRED CSV for {series_id}")
    date_col, val_col = df.columns[0], df.columns[1]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    dates = dates.dt.normalize()
    vals = pd.to_numeric(df[val_col], errors="coerce")
    out = pd.DataFrame({"date": dates, "value": vals})
    out = out.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    out = out[out["date"].dt.dayofweek < 5].reset_index(drop=True)

    since_ts = pd.to_datetime(since).normalize()
    out = out[out["date"] >= since_ts]
    if until is not None:
        until_ts = pd.to_datetime(until).normalize()
        out = out[out["date"] <= until_ts]
    return out.reset_index(drop=True)


def _hy_oas_proxy_pct(hyg: np.ndarray, ief: np.ndarray) -> np.ndarray:
    """HY credit stress proxy in % from HYG vs IEF (IG) relative performance."""
    ratio = np.maximum(hyg, 1e-8) / np.maximum(ief, 1e-8)
    return np.clip(3.5 - 12.0 * np.log(ratio), 2.0, 15.0)


def _attach_hy_oas_column(
    merged: pd.DataFrame,
    hyg_close: pd.Series,
    since: str,
    until: Optional[str],
) -> pd.DataFrame:
    """HY OAS in %: FRED where available, HYG/IEF proxy back-filled and overlap-calibrated."""
    col = "HY_OAS_close"
    ief = merged["BOND10Y_close"].astype(np.float64)
    proxy = pd.Series(
        _hy_oas_proxy_pct(hyg_close.to_numpy(dtype=np.float64), ief.to_numpy(dtype=np.float64)),
        index=merged.index,
    )

    print(f"  Fetching HY_OAS (FRED {HY_OAS_FRED_ID})...")
    fred_vals = pd.Series(np.nan, index=merged.index, dtype=np.float64)
    try:
        fred = fetch_fred_daily_series(HY_OAS_FRED_ID, since=since, until=until)
        if not fred.empty:
            fred = fred.rename(columns={"value": col})
            tmp = merged[["date"]].merge(fred, on="date", how="left")
            fred_vals = tmp[col].astype(np.float64)
            print(f"    → {int(fred_vals.notna().sum())} FRED bars on panel")
        else:
            print("    → 0 FRED bars (using proxy only)")
    except RuntimeError as e:
        print(f"    → FRED failed ({e}); proxy only")

    overlap = fred_vals.notna() & proxy.notna()
    if int(overlap.sum()) >= 30:
        a, b = np.polyfit(proxy[overlap].to_numpy(), fred_vals[overlap].to_numpy(), 1)
        proxy_cal = a * proxy + b
        print(f"    → calibrated proxy to FRED (n={int(overlap.sum())}, a={a:.3f}, b={b:.3f})")
    else:
        proxy_cal = proxy

    merged[col] = fred_vals.where(fred_vals.notna(), proxy_cal)
    merged[col] = merged[col].ffill().bfill().fillna(0.0)
    return merged


def _ohlcv_macro_from_merged(
    merged: pd.DataFrame,
    active_tickers: List[str],
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Extract aligned OHLCV + macro panels from a merged daily DataFrame."""
    merged = merged.sort_values("date").reset_index(drop=True)
    idx = pd.DatetimeIndex(merged["date"])
    ohlcv = np.zeros((len(merged), len(active_tickers), 5), dtype=np.float64)
    for j, ticker in enumerate(active_tickers):
        ohlcv[:, j, 0] = merged[f"{ticker}_open"].to_numpy()
        ohlcv[:, j, 1] = merged[f"{ticker}_high"].to_numpy()
        ohlcv[:, j, 2] = merged[f"{ticker}_low"].to_numpy()
        ohlcv[:, j, 3] = merged[f"{ticker}_close"].to_numpy()
        ohlcv[:, j, 4] = merged[f"{ticker}_volume"].to_numpy()
    ohlcv = np.nan_to_num(ohlcv, nan=0.0, posinf=0.0, neginf=0.0)
    ohlcv[:, :, :4] = np.maximum(ohlcv[:, :, :4], 1e-8)

    macro = np.zeros((len(merged), N_MACRO), dtype=np.float64)
    for j, ticker in enumerate(MACRO_TICKERS):
        col = f"{ticker}_close"
        if col in merged.columns:
            macro[:, j] = np.nan_to_num(merged[col].to_numpy(dtype=np.float64), nan=0.0)
            macro[:, j] = np.maximum(macro[:, j], 1e-8)
    return idx, ohlcv, macro


def compute_trend_signals(ohlcv: np.ndarray) -> np.ndarray:
    """Dual-EMA distance per asset: (EMA20 - EMA100) / EMA100 (stationary trend memory)."""
    t, n_assets = ohlcv.shape[0], ohlcv.shape[1]
    trend_signals = np.zeros((t, n_assets), dtype=np.float64)
    for j in range(n_assets):
        close_s = pd.Series(ohlcv[:, j, 3])
        ema_fast = close_s.ewm(span=20, adjust=False).mean()
        ema_slow = close_s.ewm(span=100, adjust=False).mean()
        trend_signals[:, j] = ((ema_fast - ema_slow) / (ema_slow + 1e-12)).to_numpy()
    return trend_signals


def compute_feature_panel(
    ohlcv: np.ndarray,
    macro: np.ndarray,
    fracdiff_d: float = DEFAULT_FRACDIFF_D,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RSI, MACD, trend EMA profile, and fracdiff on a **contiguous** OHLCV slice."""
    t, n_assets = ohlcv.shape[0], ohlcv.shape[1]
    rsi = np.zeros((t, n_assets), dtype=np.float64)
    macd = np.zeros((t, n_assets), dtype=np.float64)
    for j in range(n_assets):
        close_s = pd.Series(ohlcv[:, j, 3])
        rsi[:, j] = _rsi(close_s).fillna(50.0).to_numpy()
        macd_line = _macd_line(close_s)
        macd[:, j] = (macd_line / (close_s.abs() + 1e-12)).fillna(0.0).to_numpy()
    trend_signals = compute_trend_signals(ohlcv)
    fracdiff, fracdiff_macro = compute_fracdiff_panel(ohlcv, macro, d=fracdiff_d)
    return rsi, macd, fracdiff, fracdiff_macro, trend_signals


def _neutralize_feature_warmup(
    rsi: np.ndarray,
    macd: np.ndarray,
    fracdiff: np.ndarray,
    fracdiff_macro: np.ndarray,
    purge_bars: int,
    trend: np.ndarray | None = None,
) -> None:
    """Embargo first ``purge_bars`` of a joined segment (in-place)."""
    if purge_bars <= 0:
        return
    n = min(purge_bars, rsi.shape[0])
    rsi[:n] = 50.0
    macd[:n] = 0.0
    fracdiff[:n] = 0.0
    fracdiff_macro[:n] = 0.0
    if trend is not None:
        trend[:n] = 0.0


def _indicators_from_merged(
    merged: pd.DataFrame,
    active_tickers: List[str],
    fracdiff_d: float = DEFAULT_FRACDIFF_D,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full-timeline features (OK for contiguous OOS backtest / cache)."""
    idx, ohlcv, macro = _ohlcv_macro_from_merged(merged, active_tickers)
    rsi, macd, fracdiff, fracdiff_macro, trend = compute_feature_panel(
        ohlcv, macro, fracdiff_d=fracdiff_d
    )
    return idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend


def _yfinance_daily_history(
    tkr,
    since: str,
    until: Optional[str],
) -> pd.DataFrame:
    """
    yfinance often returns an empty frame for ``start=``/``end=`` (rate limits, Yahoo
    hiccups). Retries and a longer ``period=`` pull + date filter are more reliable.
    """
    import time

    def _h_start_end() -> pd.DataFrame:
        return tkr.history(start=since, end=until, interval="1d", auto_adjust=False)

    def _h_period(period: str) -> pd.DataFrame:
        return tkr.history(period=period, interval="1d", auto_adjust=False)

    df0 = _h_start_end()
    for attempt in range(3):
        if not df0.empty:
            return df0
        time.sleep(0.35 * (2**attempt))
        df0 = _h_start_end()
    if not df0.empty:
        return df0

    since_ts = pd.to_datetime(since).normalize()
    until_ts = pd.to_datetime(until).normalize() if until else None
    for period in ("1y", "2y", "5y", "10y", "max"):
        df0 = _h_period(period)
        if df0.empty:
            time.sleep(0.2)
            continue
        tmp = df0.reset_index()
        tcol = tmp.columns[0]
        dates = pd.to_datetime(tmp[tcol])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        dates = dates.dt.normalize()
        m = dates >= since_ts
        if until_ts is not None:
            m &= dates <= until_ts
        tmpf = tmp.loc[m]
        if not tmpf.empty:
            return tmpf.set_index(tcol)

    return pd.DataFrame()


def fetch_aligned_daily(
    symbols_dict: Dict[str, str] | None = None,
    since: str = "2014-09-01",
    until: Optional[str] = None,
    fracdiff_d: float = DEFAULT_FRACDIFF_D,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Daily OHLCV from yfinance for configured tradeable assets + macro context.

    Each asset is fetched independently, then outer-joined on date. Short
    forward-fill bridges exchange holidays (``FFILL_LIMIT``). **Backward-fill
    is never applied to asset OHLC** — dates before an asset's first real print
    stay NaN and those rows are **dropped**, so the panel begins when every
    asset has genuine prices (no anchoring to a future IPO level).

    Macro series use forward-fill only (no backward-fill into the past).

    Weekends are excluded. Requires ``--refresh-data`` after this logic change
    to rebuild ``data_cache.npz``.
    """
    import yfinance as yf

    symbols = symbols_dict if symbols_dict is not None else get_active_symbols()
    active_tickers = list(symbols.keys())

    per_ticker: Dict[str, pd.DataFrame] = {}

    def _fetch_one(ticker: str, yf_sym: str, required: bool = True) -> None:
        print(f"  Fetching {ticker} ({yf_sym})...")
        obj = yf.Ticker(yf_sym)
        df = _yfinance_daily_history(obj, since, until)
        if df.empty:
            if required:
                raise RuntimeError(f"yfinance returned no rows for {yf_sym} ({ticker})")
            print(f"    → 0 bars (skipped, not required)")
            return
        df = df.reset_index()
        time_col = df.columns[0]
        dates = pd.to_datetime(df[time_col])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        dates = dates.dt.normalize()

        asset_df = pd.DataFrame({
            "date": dates,
            f"{ticker}_open": df["Open"].to_numpy(dtype=np.float64),
            f"{ticker}_high": df["High"].to_numpy(dtype=np.float64),
            f"{ticker}_low": df["Low"].to_numpy(dtype=np.float64),
            f"{ticker}_close": df["Close"].to_numpy(dtype=np.float64),
            f"{ticker}_volume": df["Volume"].fillna(0.0).to_numpy(dtype=np.float64),
        })
        asset_df = asset_df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
        asset_df = asset_df[asset_df["date"].dt.dayofweek < 5].reset_index(drop=True)
        per_ticker[ticker] = asset_df
        print(f"    → {len(asset_df)} daily bars")

    for ticker, yf_sym in symbols.items():
        _fetch_one(ticker, yf_sym, required=True)

    for ticker, yf_sym in MACRO_SYMBOLS.items():
        _fetch_one(ticker, yf_sym, required=False)
    _fetch_one(HY_OAS_PROXY_YF, HY_OAS_PROXY_YF, required=False)

    merged = per_ticker[active_tickers[0]]
    for t in active_tickers[1:]:
        merged = merged.merge(per_ticker[t], on="date", how="outer")

    for t in MACRO_SYMBOLS:
        if t in per_ticker:
            macro_cols = per_ticker[t][["date", f"{t}_close"]]
            merged = merged.merge(macro_cols, on="date", how="left")

    merged = merged.sort_values("date").reset_index(drop=True)

    for ticker in active_tickers:
        cols = [f"{ticker}_{c}" for c in ("open", "high", "low", "close", "volume")]
        merged[cols] = merged[cols].ffill(limit=FFILL_LIMIT)

    # Volume unknown before first print → 0 (no position); do NOT bfill prices
    # from the future IPO — those rows are dropped below.
    for ticker in active_tickers:
        vol_col = f"{ticker}_volume"
        merged[vol_col] = merged[vol_col].fillna(0.0)

    for ticker in MACRO_SYMBOLS:
        col = f"{ticker}_close"
        if col in merged.columns:
            # Forward-fill only; leading NaNs → 0 (macro obs uses log; env clamps)
            merged[col] = merged[col].ffill().fillna(0.0)

    # HY OAS: FRED + HYG/IEF proxy (full history for walk-forward from 2006)
    if HY_OAS_PROXY_YF in per_ticker and "BOND10Y_close" in merged.columns:
        hyg_df = per_ticker[HY_OAS_PROXY_YF][["date", f"{HY_OAS_PROXY_YF}_close"]].rename(
            columns={f"{HY_OAS_PROXY_YF}_close": "_hyg_tmp"}
        )
        merged = merged.merge(hyg_df, on="date", how="left")
        merged["_hyg_tmp"] = merged["_hyg_tmp"].ffill()
        merged = _attach_hy_oas_column(merged, merged["_hyg_tmp"], since=since, until=until)
        merged = merged.drop(columns=["_hyg_tmp"])
    else:
        print("  WARNING: HY_OAS skipped (missing HYG or BOND10Y on panel)")

    asset_cols = []
    for ticker in active_tickers:
        asset_cols.extend([f"{ticker}_{c}" for c in ("open", "high", "low", "close")])
    merged = merged.dropna(subset=asset_cols).reset_index(drop=True)

    if merged.empty:
        raise RuntimeError(
            "No overlapping dates after alignment; check date ranges and asset availability."
        )

    print(
        f"  Aligned: {len(merged)} daily bars, {len(active_tickers)} assets "
        f"(all live; no pre-IPO bfill) + {N_MACRO} macro — "
        f"{merged['date'].iloc[0].date()} .. {merged['date'].iloc[-1].date()}"
    )
    return _indicators_from_merged(merged, active_tickers, fracdiff_d=fracdiff_d)


def clip_index_until(
    index: pd.DatetimeIndex,
    *arrays: np.ndarray,
    until: str | pd.Timestamp,
) -> Tuple[pd.DatetimeIndex, Tuple[np.ndarray, ...]]:
    """Keep rows with ``date <= until`` (inclusive, calendar day)."""
    if len(index) == 0:
        raise ValueError("Empty dataset")
    end = pd.Timestamp(until).normalize()
    mask = index.normalize() <= end
    if not np.any(mask):
        raise ValueError(f"No rows on or before until={end.date()}; extend history or relax --until.")
    clipped = (index[mask], *(a[mask] for a in arrays))
    return clipped[0], clipped[1:]


def reserve_chronological_holdout(
    index: pd.DatetimeIndex,
    *arrays: np.ndarray,
    holdout_days: int = 365,
    train_end: str | pd.Timestamp | None = None,
    holdout_start: str | pd.Timestamp | None = None,
    holdout_end: str | pd.Timestamp | None = None,
) -> Tuple[Tuple, Tuple]:
    """Reserve OOS bars for backtest-only evaluation.

    **Calendar tail (default):** last ``holdout_days`` calendar days → trainable is
    ``date <= last_date - holdout_days``; holdout is the remainder.

    **Explicit dates:** pass ``train_end`` and ``holdout_start`` (and optional
    ``holdout_end``). Trainable is ``date <= train_end``; holdout is
    ``holdout_start <= date <= holdout_end`` (default ``holdout_end`` = last bar).
    Any gap between ``train_end`` and ``holdout_start`` is excluded from both.

    Training and in-training eval must use only the trainable segment; holdout must
    never be passed to ``train_test_split_alternating``.

    Returns
    -------
    trainable : (index, *arrays sliced)
    holdout : (index, *arrays sliced)
    """
    if len(index) == 0:
        raise ValueError("Empty dataset")

    use_dates = train_end is not None or holdout_start is not None
    if use_dates:
        if train_end is None or holdout_start is None:
            raise ValueError(
                "Date-based holdout requires both train_end and holdout_start "
                "(e.g. --train-end 2020-12-31 --holdout-start 2021-01-01)."
            )
        t_end = pd.Timestamp(train_end).normalize()
        h_start = pd.Timestamp(holdout_start).normalize()
        h_end = (
            pd.Timestamp(holdout_end).normalize()
            if holdout_end is not None
            else index[-1].normalize()
        )
        if t_end >= h_start:
            raise ValueError(
                f"train_end ({t_end.date()}) must be strictly before "
                f"holdout_start ({h_start.date()})."
            )
        idx_norm = index.normalize()
        before_mask = idx_norm <= t_end
        hold_mask = (idx_norm >= h_start) & (idx_norm <= h_end)
    else:
        if holdout_days <= 0:
            raise ValueError("holdout_days must be positive")
        cutoff = index[-1] - pd.Timedelta(days=holdout_days)
        before_mask = index <= cutoff
        hold_mask = ~before_mask

    if not np.any(hold_mask):
        raise ValueError(
            "No rows in holdout window; extend data, relax dates, or reduce holdout_days."
        )
    if not np.any(before_mask):
        raise ValueError(
            "No trainable rows before holdout; extend history or relax holdout settings."
        )

    train_idx = index[before_mask]
    hold_idx = index[hold_mask]
    train_arrays = tuple(a[before_mask] for a in arrays)
    hold_arrays = tuple(a[hold_mask] for a in arrays)
    return (train_idx, *train_arrays), (hold_idx, *hold_arrays)


def train_test_split_alternating(
    index: pd.DatetimeIndex,
    ohlcv: np.ndarray,
    macro: np.ndarray,
    block_size: int = 126,
    eval_stride: int = 4,
    fracdiff_d: float = DEFAULT_FRACDIFF_D,
    feature_purge_warmup: int = 25,
) -> Tuple[Tuple, Tuple]:
    """Walk-forward alternating split for representative train/eval regimes.

    Divides the timeline into blocks of ``block_size`` trading bars (~6 months
    at 126).  Every ``eval_stride``-th block is assigned to eval; the rest go
    to train.  Both sets therefore span the full history and contain a mix of
    bull, bear, high-vol, and low-vol periods.

    **Features are computed per block on raw OHLCV only**, so RSI/MACD/fracdiff
    on an eval block never inherit EWM or fracdiff memory from adjacent train
    blocks.  At each join inside a concatenated stream, the first
    ``feature_purge_warmup`` bars are neutralized (RSI=50, MACD/fracdiff=0).

    Contiguous same-label blocks are merged into *segments*.  ``block_boundaries``
    lists join indices so the environment can prevent episodes from spanning them.

    Returns
    -------
    train_tuple : (index, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend, block_boundaries)
    eval_tuple  : (index, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend, block_boundaries)
    """
    n = len(index)
    if n == 0:
        raise ValueError("Empty dataset")

    n_blocks = (n + block_size - 1) // block_size

    train_ranges: List[Tuple[int, int]] = []
    eval_ranges: List[Tuple[int, int]] = []

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        if (b + 1) % eval_stride == 0:
            eval_ranges.append((start, end))
        else:
            train_ranges.append((start, end))

    def _concat_ranges(
        ranges: List[Tuple[int, int]],
    ) -> Tuple[
        pd.DatetimeIndex,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[int],
    ]:
        idx_parts: List[pd.DatetimeIndex] = []
        ohlcv_parts: List[np.ndarray] = []
        rsi_parts: List[np.ndarray] = []
        macd_parts: List[np.ndarray] = []
        macro_parts: List[np.ndarray] = []
        fd_parts: List[np.ndarray] = []
        fdm_parts: List[np.ndarray] = []
        trend_parts: List[np.ndarray] = []
        boundaries: List[int] = []
        offset = 0
        prev_orig_end: Optional[int] = None
        segment_idx = 0

        for orig_start, orig_end in ranges:
            if prev_orig_end is not None and orig_start != prev_orig_end:
                boundaries.append(offset)

            ohlcv_chunk = ohlcv[orig_start:orig_end]
            macro_chunk = macro[orig_start:orig_end]
            rsi_c, macd_c, fd_c, fdm_c, trend_c = compute_feature_panel(
                ohlcv_chunk, macro_chunk, fracdiff_d=fracdiff_d
            )
            if segment_idx > 0:
                _neutralize_feature_warmup(
                    rsi_c, macd_c, fd_c, fdm_c, feature_purge_warmup, trend_c
                )

            chunk_len = orig_end - orig_start
            idx_parts.append(index[orig_start:orig_end])
            ohlcv_parts.append(ohlcv_chunk)
            rsi_parts.append(rsi_c)
            macd_parts.append(macd_c)
            macro_parts.append(macro_chunk)
            fd_parts.append(fd_c)
            fdm_parts.append(fdm_c)
            trend_parts.append(trend_c)
            offset += chunk_len
            prev_orig_end = orig_end
            segment_idx += 1

        cat_idx = idx_parts[0].append(idx_parts[1:]) if len(idx_parts) > 1 else idx_parts[0]
        cat_ohlcv = np.concatenate(ohlcv_parts, axis=0)
        cat_rsi = np.concatenate(rsi_parts, axis=0)
        cat_macd = np.concatenate(macd_parts, axis=0)
        cat_macro = np.concatenate(macro_parts, axis=0)
        cat_fd = np.concatenate(fd_parts, axis=0)
        cat_fdm = np.concatenate(fdm_parts, axis=0)
        cat_trend = np.concatenate(trend_parts, axis=0)
        return cat_idx, cat_ohlcv, cat_rsi, cat_macd, cat_macro, cat_fd, cat_fdm, cat_trend, boundaries

    tr_idx, tr_ohlcv, tr_rsi, tr_macd, tr_macro, tr_fd, tr_fdm, tr_trend, tr_bounds = _concat_ranges(
        train_ranges
    )
    ev_idx, ev_ohlcv, ev_rsi, ev_macd, ev_macro, ev_fd, ev_fdm, ev_trend, ev_bounds = _concat_ranges(
        eval_ranges
    )

    return (
        (tr_idx, tr_ohlcv, tr_rsi, tr_macd, tr_macro, tr_fd, tr_fdm, tr_trend, tr_bounds),
        (ev_idx, ev_ohlcv, ev_rsi, ev_macd, ev_macro, ev_fd, ev_fdm, ev_trend, ev_bounds),
    )


def save_cache(
    path: str,
    index: pd.DatetimeIndex,
    ohlcv: np.ndarray,
    rsi: np.ndarray,
    macd: np.ndarray,
    macro: np.ndarray,
    fracdiff: np.ndarray,
    fracdiff_macro: np.ndarray,
    trend: np.ndarray,
    fracdiff_d: float = DEFAULT_FRACDIFF_D,
    tickers: Sequence[str] | None = None,
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    idx = index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    active_tickers = np.asarray(resolve_tickers(tickers), dtype=object)
    np.savez_compressed(
        str(cache_path),
        index=idx.to_numpy(dtype="datetime64[ns]"),
        ohlcv=ohlcv,
        rsi=rsi,
        macd=macd,
        macro=macro,
        fracdiff=fracdiff,
        fracdiff_macro=fracdiff_macro,
        trend=trend,
        fracdiff_d=np.array([fracdiff_d], dtype=np.float64),
        tickers=active_tickers,
    )


def load_cache(
    path: str,
) -> Tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    z = np.load(path, allow_pickle=True)
    idx = pd.DatetimeIndex(z["index"])
    ohlcv = z["ohlcv"]
    if "tickers" in z.files:
        tickers = [str(x) for x in z["tickers"].tolist()]
    else:
        tickers = resolve_tickers()
        if ohlcv.shape[1] != len(tickers):
            raise ValueError(
                f"Cache has {ohlcv.shape[1]} assets but no tickers array; "
                "rebuild with: python scripts/train.py --refresh-data"
            )
    if "macro" in z:
        macro = z["macro"]
    else:
        macro = np.zeros((len(idx), N_MACRO), dtype=np.float64)
    if macro.shape[1] != N_MACRO:
        raise ValueError(
            f"Cache macro has {macro.shape[1]} series but code expects {N_MACRO} "
            f"({MACRO_TICKERS}). Rebuild: python scripts/train.py --refresh-data "
            "(writes .cache/data_cache.npz)"
        )
    d = float(z["fracdiff_d"][0]) if "fracdiff_d" in z else DEFAULT_FRACDIFF_D
    if "fracdiff" in z and "fracdiff_macro" in z:
        fd = z["fracdiff"]
        fdm = z["fracdiff_macro"]
        if fdm.shape[1] != N_MACRO:
            fd, fdm = compute_fracdiff_panel(ohlcv, macro, d=d)
    else:
        fd, fdm = compute_fracdiff_panel(ohlcv, macro, d=d)
    if "trend" in z.files:
        trend = z["trend"]
    else:
        trend = compute_trend_signals(ohlcv)
    if ohlcv.shape[1] != len(tickers):
        raise ValueError(
            f"Cache ohlcv has {ohlcv.shape[1]} assets but tickers lists {len(tickers)}: {tickers}"
        )
    return idx, ohlcv, z["rsi"], z["macd"], macro, fd, fdm, trend, tickers
