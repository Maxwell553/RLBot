"""
Historical OHLCV fetch + alignment for multi-asset global portfolio RL.

Daily data via yfinance for 10 global assets:
  S&P 500 (SPY), Gold (GLD), Crude Oil WTI (USO), EUR/USD, USD/JPY,
  Nikkei 225, FTSE 100, 10-Year Treasury (IEF), Copper (HG=F),
  Emerging Markets (EEM).

Macro context features (observation-only, not tradeable):
  DXY (Dollar Index), 10-Year Treasury Yield (^TNX).

Assets trade on different exchanges/schedules; daily bars are aligned via
outer-join and short forward-fill for holiday gaps. **Pre-listing rows are
dropped** (no backward-fill of future IPO prices). The panel starts when all
assets have real quotes. Weekends are excluded.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Readable ticker labels → yfinance symbols
YF_SYMBOLS: Dict[str, str] = {
    "SP500":   "SPY",
    "GOLD":    "GLD",
    "OIL":     "USO",
    "EURUSD":  "EURUSD=X",
    "USDJPY":  "USDJPY=X",
    "NIKKEI":  "^N225",
    "FTSE":    "^FTSE",
    "BOND10Y": "IEF",
    "COPPER":  "HG=F",
    "EM":      "EEM",
}

TICKERS: List[str] = list(YF_SYMBOLS.keys())

# Macro context features — not tradeable, observation-only
MACRO_SYMBOLS: Dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "TNX": "^TNX",
}
MACRO_TICKERS: List[str] = list(MACRO_SYMBOLS.keys())
N_MACRO = len(MACRO_TICKERS)

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
    """Apply fractional differentiation to a 1D series (e.g. log prices)."""
    w = fracdiff_weights(d)
    K = len(w)
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        acc = 0.0
        up = min(t + 1, K)
        for k in range(up):
            acc += w[k] * x[t - k]
        out[t] = acc
    return out


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


def _indicators_from_merged(
    merged: pd.DataFrame,
    fracdiff_d: float = DEFAULT_FRACDIFF_D,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    merged = merged.sort_values("date").reset_index(drop=True)
    idx = pd.DatetimeIndex(merged["date"])
    ohlcv = np.zeros((len(merged), len(TICKERS), 5), dtype=np.float64)
    rsi = np.zeros((len(merged), len(TICKERS)), dtype=np.float64)
    macd = np.zeros((len(merged), len(TICKERS)), dtype=np.float64)
    for j, ticker in enumerate(TICKERS):
        ohlcv[:, j, 0] = merged[f"{ticker}_open"].to_numpy()
        ohlcv[:, j, 1] = merged[f"{ticker}_high"].to_numpy()
        ohlcv[:, j, 2] = merged[f"{ticker}_low"].to_numpy()
        ohlcv[:, j, 3] = merged[f"{ticker}_close"].to_numpy()
        ohlcv[:, j, 4] = merged[f"{ticker}_volume"].to_numpy()
        close_s = merged[f"{ticker}_close"]
        # No bfill: undefined RSI at series start uses neutral 50 (not a future RSI).
        rsi[:, j] = _rsi(close_s).fillna(50.0).to_numpy()
        macd_line = _macd_line(close_s)
        macd[:, j] = (macd_line / (close_s.abs() + 1e-12)).fillna(0.0).to_numpy()
    ohlcv = np.nan_to_num(ohlcv, nan=0.0, posinf=0.0, neginf=0.0)
    ohlcv[:, :, :4] = np.maximum(ohlcv[:, :, :4], 1e-8)

    macro = np.zeros((len(merged), N_MACRO), dtype=np.float64)
    for j, ticker in enumerate(MACRO_TICKERS):
        col = f"{ticker}_close"
        if col in merged.columns:
            macro[:, j] = np.nan_to_num(merged[col].to_numpy(dtype=np.float64), nan=0.0)
            macro[:, j] = np.maximum(macro[:, j], 1e-8)

    fracdiff, fracdiff_macro = compute_fracdiff_panel(ohlcv, macro, d=fracdiff_d)
    return idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro


def fetch_aligned_daily(
    since: str = "2014-09-01",
    until: Optional[str] = None,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Daily OHLCV from yfinance for all 10 global assets + macro context.

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

    per_ticker: Dict[str, pd.DataFrame] = {}

    def _fetch_one(ticker: str, yf_sym: str, required: bool = True) -> None:
        print(f"  Fetching {ticker} ({yf_sym})...")
        obj = yf.Ticker(yf_sym)
        df = obj.history(start=since, end=until, interval="1d", auto_adjust=False)
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

    for ticker, yf_sym in YF_SYMBOLS.items():
        _fetch_one(ticker, yf_sym, required=True)

    for ticker, yf_sym in MACRO_SYMBOLS.items():
        _fetch_one(ticker, yf_sym, required=False)

    merged = per_ticker[TICKERS[0]]
    for t in TICKERS[1:]:
        merged = merged.merge(per_ticker[t], on="date", how="outer")

    for t in MACRO_TICKERS:
        if t in per_ticker:
            macro_cols = per_ticker[t][["date", f"{t}_close"]]
            merged = merged.merge(macro_cols, on="date", how="left")

    merged = merged.sort_values("date").reset_index(drop=True)

    for ticker in TICKERS:
        cols = [f"{ticker}_{c}" for c in ("open", "high", "low", "close", "volume")]
        merged[cols] = merged[cols].ffill(limit=FFILL_LIMIT)

    # Volume unknown before first print → 0 (no position); do NOT bfill prices
    # from the future IPO — those rows are dropped below.
    for ticker in TICKERS:
        vol_col = f"{ticker}_volume"
        merged[vol_col] = merged[vol_col].fillna(0.0)

    for ticker in MACRO_TICKERS:
        col = f"{ticker}_close"
        if col in merged.columns:
            # Forward-fill only; leading NaNs → 0 (macro obs uses log; env clamps)
            merged[col] = merged[col].ffill().fillna(0.0)

    asset_cols = []
    for ticker in TICKERS:
        asset_cols.extend([f"{ticker}_{c}" for c in ("open", "high", "low", "close")])
    merged = merged.dropna(subset=asset_cols).reset_index(drop=True)

    if merged.empty:
        raise RuntimeError(
            "No overlapping dates after alignment; check date ranges and asset availability."
        )

    print(
        f"  Aligned: {len(merged)} daily bars (all assets live; no pre-IPO bfill) "
        f"+ {N_MACRO} macro — {merged['date'].iloc[0].date()} .. {merged['date'].iloc[-1].date()}"
    )
    return _indicators_from_merged(merged, fracdiff_d=DEFAULT_FRACDIFF_D)


def train_test_split_last_days(
    index: pd.DatetimeIndex,
    *arrays: np.ndarray,
    test_days: int = 60,
) -> Tuple[Tuple, Tuple]:
    """Hold out the last `test_days` calendar days."""
    if len(index) == 0:
        raise ValueError("Empty dataset")
    cutoff = index[-1] - pd.Timedelta(days=test_days)
    train_mask = index <= cutoff
    test_mask = ~train_mask
    train_idx = index[train_mask]
    test_idx = index[test_mask]
    return (train_idx, *(a[train_mask] for a in arrays)), (test_idx, *(a[test_mask] for a in arrays))


def reserve_chronological_holdout(
    index: pd.DatetimeIndex,
    *arrays: np.ndarray,
    holdout_days: int = 365,
) -> Tuple[Tuple, Tuple]:
    """Reserve the last ``holdout_days`` *calendar* rows for backtest-only OOS evaluation.

    Training and in-training eval must use only the **left** segment; the **right**
    segment must never be passed to ``train_test_split_alternating`` or the policy
    will have seen those bars.

    Returns
    -------
    trainable : (index, *arrays sliced)
        All bars with ``date <= (last_date - holdout_days)``.
    holdout : (index, *arrays sliced)
        The trailing calendar window used exclusively by ``backtest.py``.
    """
    if holdout_days <= 0:
        raise ValueError("holdout_days must be positive (use train_test_split_last_days for ad-hoc slices).")
    if len(index) == 0:
        raise ValueError("Empty dataset")
    cutoff = index[-1] - pd.Timedelta(days=holdout_days)
    before_mask = index <= cutoff
    hold_mask = ~before_mask
    if not np.any(hold_mask):
        raise ValueError(
            f"No rows in holdout window (holdout_days={holdout_days}); "
            "reduce holdout_days or extend data."
        )
    if not np.any(before_mask):
        raise ValueError(
            "Holdout consumes all data; reduce holdout_days or extend history."
        )
    train_idx = index[before_mask]
    hold_idx = index[hold_mask]
    train_arrays = tuple(a[before_mask] for a in arrays)
    hold_arrays = tuple(a[hold_mask] for a in arrays)
    return (train_idx, *train_arrays), (hold_idx, *hold_arrays)


def train_test_split_alternating(
    index: pd.DatetimeIndex,
    ohlcv: np.ndarray,
    rsi: np.ndarray,
    macd: np.ndarray,
    macro: np.ndarray,
    fracdiff: np.ndarray,
    fracdiff_macro: np.ndarray,
    block_size: int = 126,
    eval_stride: int = 4,
) -> Tuple[Tuple, Tuple]:
    """Walk-forward alternating split for representative train/eval regimes.

    Divides the timeline into blocks of ``block_size`` trading bars (~6 months
    at 126).  Every ``eval_stride``-th block is assigned to eval; the rest go
    to train.  Both sets therefore span the full history and contain a mix of
    bull, bear, high-vol, and low-vol periods.

    Contiguous same-label blocks are merged into *segments*.  When an eval
    block is removed from the train stream (or vice versa), a discontinuity
    exists at the join.  ``block_boundaries`` lists these join indices so the
    environment can prevent episodes from spanning them.

    Returns
    -------
    train_tuple : (index, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, block_boundaries)
    eval_tuple  : (index, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, block_boundaries)
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
    ) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        idx_parts: List[pd.DatetimeIndex] = []
        ohlcv_parts: List[np.ndarray] = []
        rsi_parts: List[np.ndarray] = []
        macd_parts: List[np.ndarray] = []
        macro_parts: List[np.ndarray] = []
        fd_parts: List[np.ndarray] = []
        fdm_parts: List[np.ndarray] = []
        boundaries: List[int] = []
        offset = 0
        prev_orig_end: Optional[int] = None

        for orig_start, orig_end in ranges:
            if prev_orig_end is not None and orig_start != prev_orig_end:
                boundaries.append(offset)
            chunk_len = orig_end - orig_start
            idx_parts.append(index[orig_start:orig_end])
            ohlcv_parts.append(ohlcv[orig_start:orig_end])
            rsi_parts.append(rsi[orig_start:orig_end])
            macd_parts.append(macd[orig_start:orig_end])
            macro_parts.append(macro[orig_start:orig_end])
            fd_parts.append(fracdiff[orig_start:orig_end])
            fdm_parts.append(fracdiff_macro[orig_start:orig_end])
            offset += chunk_len
            prev_orig_end = orig_end

        cat_idx = idx_parts[0].append(idx_parts[1:]) if len(idx_parts) > 1 else idx_parts[0]
        cat_ohlcv = np.concatenate(ohlcv_parts, axis=0)
        cat_rsi = np.concatenate(rsi_parts, axis=0)
        cat_macd = np.concatenate(macd_parts, axis=0)
        cat_macro = np.concatenate(macro_parts, axis=0)
        cat_fd = np.concatenate(fd_parts, axis=0)
        cat_fdm = np.concatenate(fdm_parts, axis=0)
        return cat_idx, cat_ohlcv, cat_rsi, cat_macd, cat_macro, cat_fd, cat_fdm, boundaries

    tr_idx, tr_ohlcv, tr_rsi, tr_macd, tr_macro, tr_fd, tr_fdm, tr_bounds = _concat_ranges(train_ranges)
    ev_idx, ev_ohlcv, ev_rsi, ev_macd, ev_macro, ev_fd, ev_fdm, ev_bounds = _concat_ranges(eval_ranges)

    return (
        (tr_idx, tr_ohlcv, tr_rsi, tr_macd, tr_macro, tr_fd, tr_fdm, tr_bounds),
        (ev_idx, ev_ohlcv, ev_rsi, ev_macd, ev_macro, ev_fd, ev_fdm, ev_bounds),
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
    fracdiff_d: float = DEFAULT_FRACDIFF_D,
) -> None:
    idx = index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    np.savez_compressed(
        path,
        index=idx.to_numpy(dtype="datetime64[ns]"),
        ohlcv=ohlcv,
        rsi=rsi,
        macd=macd,
        macro=macro,
        fracdiff=fracdiff,
        fracdiff_macro=fracdiff_macro,
        fracdiff_d=np.array([fracdiff_d], dtype=np.float64),
    )


def load_cache(
    path: str,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    idx = pd.DatetimeIndex(z["index"])
    if "macro" in z:
        macro = z["macro"]
    else:
        macro = np.zeros((len(idx), N_MACRO), dtype=np.float64)
    ohlcv = z["ohlcv"]
    if "fracdiff" in z and "fracdiff_macro" in z:
        fd = z["fracdiff"]
        fdm = z["fracdiff_macro"]
        d = float(z["fracdiff_d"][0]) if "fracdiff_d" in z else DEFAULT_FRACDIFF_D
    else:
        d = DEFAULT_FRACDIFF_D
        fd, fdm = compute_fracdiff_panel(ohlcv, macro, d=d)
    return idx, ohlcv, z["rsi"], z["macd"], macro, fd, fdm
