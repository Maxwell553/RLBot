"""
Historical OHLCV fetch and alignment for multi-asset portfolio RL.

Tradeable symbols come from ``config.yaml`` → ``universe.assets`` (via
``get_active_symbols()``). Macro context (DXY, TNX, VIX, HY OAS) is
observation-only.

Daily bars are outer-joined across assets with short forward-fill for holiday
gaps; there is **no global row drop**. Pre-listing rows are kept and live-masked
(``asset_live`` = 0): prices are back-filled with the first listing print for
continuity, per-asset features are neutralized on pre-live bars, and the env
forces pre-live weights to zero. Post-delisting bars keep the last real price
(unlimited forward-fill) so a dead position liquidates at its last real close.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

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
    asset_live: np.ndarray | None = None,
    asset_vol: np.ndarray | None = None,
    macro_vol: np.ndarray | None = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Restrict asset-axis arrays to ``universe_tickers`` (subset of ``panel_tickers``, same order)."""
    panel = list(panel_tickers)
    want = list(universe_tickers)
    if asset_live is None:
        asset_live = np.ones((ohlcv.shape[0], ohlcv.shape[1]), dtype=np.float64)
    if asset_vol is None or macro_vol is None:
        raise ValueError(
            "asset_vol and macro_vol are required (rebuild cache: --refresh-data)"
        )
    if panel == want:
        return ohlcv, rsi, macd, fracdiff, trend, want, asset_live, asset_vol, macro_vol
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
        asset_live[:, indices],
        asset_vol[:, indices],
        macro_vol,
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


def _calibrate_hy_proxy_expanding(
    proxy: np.ndarray,
    fred: np.ndarray,
    *,
    min_overlap: int = 30,
) -> np.ndarray:
    """Causal affine calibration via expanding OLS (O(n), no per-bar polyfit)."""
    proxy = np.asarray(proxy, dtype=np.float64)
    fred = np.asarray(fred, dtype=np.float64)
    n = len(proxy)
    out = np.empty(n, dtype=np.float64)
    a, b = 1.0, 0.0
    n_eff = 0
    sx = sy = sxx = sxy = 0.0
    for i in range(n):
        if np.isfinite(fred[i]) and np.isfinite(proxy[i]):
            x = float(proxy[i])
            y = float(fred[i])
            n_eff += 1
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
        if n_eff >= min_overlap:
            denom = n_eff * sxx - sx * sx
            if abs(denom) > 1e-12:
                a = (n_eff * sxy - sx * sy) / denom
                b = (sy - a * sx) / n_eff
        if np.isfinite(proxy[i]):
            out[i] = a * proxy[i] + b
        else:
            out[i] = np.nan
    return out


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

    proxy_cal = pd.Series(
        _calibrate_hy_proxy_expanding(
            proxy.to_numpy(dtype=np.float64),
            fred_vals.to_numpy(dtype=np.float64),
            min_overlap=30,
        ),
        index=merged.index,
    )
    overlap_n = int((fred_vals.notna() & proxy.notna()).sum())
    if overlap_n >= 30:
        print(
            f"    → causal expanding proxy→FRED calibration "
            f"(overlap bars={overlap_n})"
        )

    merged[col] = fred_vals.where(fred_vals.notna(), proxy_cal)
    # Forward-fill only — never bfill: a truncated FRED fetch would otherwise push a
    # future calibrated spread level into pre-coverage bars. Leading NaNs → 0 (the env
    # treats 0 as missing, same convention as the other macro series).
    merged[col] = merged[col].ffill().fillna(0.0)
    return merged


def _ohlcv_macro_from_merged(
    merged: pd.DataFrame,
    active_tickers: List[str],
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Extract aligned OHLCV, macro, and per-asset live masks from a merged daily DataFrame."""
    merged = merged.sort_values("date").reset_index(drop=True)
    idx = pd.DatetimeIndex(merged["date"])
    ohlcv = np.zeros((len(merged), len(active_tickers), 5), dtype=np.float64)
    asset_live = np.zeros((len(merged), len(active_tickers)), dtype=np.float64)
    for j, ticker in enumerate(active_tickers):
        live_col = f"{ticker}_live"
        if live_col in merged.columns:
            asset_live[:, j] = merged[live_col].to_numpy(dtype=np.float64)
        else:
            asset_live[:, j] = 1.0
        ohlcv[:, j, 0] = merged[f"{ticker}_open"].to_numpy()
        ohlcv[:, j, 1] = merged[f"{ticker}_high"].to_numpy()
        ohlcv[:, j, 2] = merged[f"{ticker}_low"].to_numpy()
        ohlcv[:, j, 3] = merged[f"{ticker}_close"].to_numpy()
        ohlcv[:, j, 4] = merged[f"{ticker}_volume"].to_numpy()
    ohlcv = np.nan_to_num(ohlcv, nan=0.0, posinf=0.0, neginf=0.0)
    ohlcv[:, :, :4] = np.maximum(ohlcv[:, :, :4], 1e-8)
    asset_live = np.clip(asset_live, 0.0, 1.0)

    macro = np.zeros((len(merged), N_MACRO), dtype=np.float64)
    for j, ticker in enumerate(MACRO_TICKERS):
        col = f"{ticker}_close"
        if col in merged.columns:
            macro[:, j] = np.nan_to_num(merged[col].to_numpy(dtype=np.float64), nan=0.0)
            macro[:, j] = np.maximum(macro[:, j], 1e-8)
    return idx, ohlcv, macro, asset_live


def _trailing_log_return_std(series: np.ndarray, lookback: int) -> np.ndarray:
    """Per-bar std of log returns; window ``[max(t-lookback,1), t]`` (env-aligned)."""
    t = len(series)
    out = np.zeros(t, dtype=np.float64)
    for i in range(t):
        start = max(i - lookback, 1)
        window = series[start : i + 1]
        if len(window) >= 2:
            rets = np.diff(np.log(window + 1e-12))
            out[i] = float(rets.std())
    return out


def compute_realized_vol_panels(
    ohlcv: np.ndarray,
    macro: np.ndarray,
    lookback: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trailing log-return std per bar (matches ``MultiAssetPortfolioEnv`` lookback)."""
    t, n_assets = ohlcv.shape[0], ohlcv.shape[1]
    asset_vol = np.zeros((t, n_assets), dtype=np.float64)
    closes = ohlcv[:, :, 3]
    for j in range(n_assets):
        asset_vol[:, j] = _trailing_log_return_std(closes[:, j], lookback)
    macro_vol = np.zeros((t, N_MACRO), dtype=np.float64)
    for j in range(N_MACRO):
        macro_vol[:, j] = _trailing_log_return_std(macro[:, j], lookback)
    return asset_vol, macro_vol


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
    lookback: int = 20,
    asset_live: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RSI, MACD, trend, fracdiff, and trailing realized vol panels on a contiguous slice.

    When ``asset_live`` (t, n_assets; 1 = real print) is given, per-asset features are
    neutralized on pre-live/dead bars. Pre-IPO prices are back-filled with the first
    listing price for continuity, and fracdiff of a *constant* log price is nonzero for
    hundreds of bars — without this mask the observation would encode the future
    listing price level before the asset exists.
    """
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
    asset_vol, macro_vol = compute_realized_vol_panels(ohlcv, macro, lookback=lookback)
    if asset_live is not None:
        dead = np.asarray(asset_live, dtype=np.float64) < 0.5
        rsi[dead] = 50.0
        macd[dead] = 0.0
        fracdiff[dead] = 0.0
        trend_signals[dead] = 0.0
        asset_vol[dead] = 0.0
    return rsi, macd, fracdiff, fracdiff_macro, trend_signals, asset_vol, macro_vol


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
) -> Tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Full-timeline features (OK for contiguous OOS backtest / cache)."""
    idx, ohlcv, macro, asset_live = _ohlcv_macro_from_merged(merged, active_tickers)
    rsi, macd, fracdiff, fracdiff_macro, trend, asset_vol, macro_vol = compute_feature_panel(
        ohlcv, macro, fracdiff_d=fracdiff_d, asset_live=asset_live
    )
    return idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend, asset_vol, macro_vol, asset_live


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

    # auto_adjust=True → dividend + split adjusted OHLC (total-return prices). Without
    # it, SPY/IEF/EEM/GLD returns are price-only: bond ETF coupons (most of IEF's
    # return) and ~1.5-2%/yr equity dividends vanish, biasing the agent, the
    # cap-weighted benchmark, and every baseline. Price indices (^N225, ^FTSE), FX,
    # and futures have no distributions — adjustment is a no-op for them (their
    # price-return nature is a documented universe caveat, not fixable here).
    def _h_start_end() -> pd.DataFrame:
        return tkr.history(start=since, end=until, interval="1d", auto_adjust=True)

    def _h_period(period: str) -> pd.DataFrame:
        return tkr.history(period=period, interval="1d", auto_adjust=True)

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
) -> Tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Daily OHLCV from yfinance for configured tradeable assets + macro context.

    Each asset is fetched independently, then outer-joined on date. Short
    forward-fill bridges exchange holidays (``FFILL_LIMIT``). Pre-IPO rows keep
    calendar history: ``{ticker}_live`` is 0 before the first real close, OHLC is
    filled with ``1e-8`` for safe numerics (no backward-fill from IPO levels).

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

    # Volume unknown before first print → 0 (no position).
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

    for ticker in active_tickers:
        close_col = f"{ticker}_close"
        merged[f"{ticker}_live"] = np.where(merged[close_col].notna(), 1.0, 0.0)

    asset_fill_cols = []
    for ticker in active_tickers:
        asset_fill_cols.extend([f"{ticker}_{c}" for c in ("open", "high", "low", "close")])
    # Order matters and the live mask above is already frozen:
    # 1. Unlimited ffill: post-delisting/halt bars keep the LAST REAL price, so a dead
    #    position is force-liquidated at its last real close — not at a near-zero
    #    filler (and a mid-history gap is never back-filled with the future
    #    resumption price).
    # 2. bfill: pre-IPO rows get the first listing price (flat returns → neutral
    #    RSI/MACD; fracdiff/vol are neutralized on pre-live bars via asset_live in
    #    compute_feature_panel, so the future listing level never reaches the obs).
    # 3. 1e-8 only for assets with no data at all in the window.
    merged[asset_fill_cols] = merged[asset_fill_cols].ffill()
    merged[asset_fill_cols] = merged[asset_fill_cols].bfill()
    merged[asset_fill_cols] = merged[asset_fill_cols].fillna(1e-8)

    if merged.empty:
        raise RuntimeError(
            "No overlapping dates after alignment; check date ranges and asset availability."
        )

    live_counts = {t: int(merged[f"{t}_live"].sum()) for t in active_tickers}
    print(
        f"  Aligned: {len(merged)} daily bars, {len(active_tickers)} assets "
        f"(live-masked pre-IPO; no global row drop) + {N_MACRO} macro — "
        f"{merged['date'].iloc[0].date()} .. {merged['date'].iloc[-1].date()}"
    )
    for t, n_live in live_counts.items():
        if n_live < len(merged):
            print(f"    {t}: {n_live}/{len(merged)} bars with real prints")
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


class WalkforwardEnvPack(NamedTuple):
    """
    Arrays from ``train_test_split_alternating`` in canonical order.

    Tuple layout: ``(idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend,
    asset_vol, macro_vol, block_boundaries, asset_live)``. Use ``env_kwargs()`` when building
    ``MultiAssetPortfolioEnv`` — do not unpack positionally into the env constructor.
    """

    idx: pd.DatetimeIndex
    ohlcv: np.ndarray
    rsi: np.ndarray
    macd: np.ndarray
    macro: np.ndarray
    fracdiff: np.ndarray
    fracdiff_macro: np.ndarray
    trend: np.ndarray
    asset_vol: np.ndarray
    macro_vol: np.ndarray
    block_boundaries: list
    asset_live: np.ndarray

    @classmethod
    def from_tuple(cls, row: tuple) -> WalkforwardEnvPack:
        if len(row) != 12:
            raise ValueError(f"expected 12-tuple walk-forward pack, got len={len(row)}")
        return cls(*row)

    def env_kwargs(self) -> dict:
        """Keyword args for ``MultiAssetPortfolioEnv`` (feature order matches this pack)."""
        return {
            "ohlcv": self.ohlcv,
            "rsi": self.rsi,
            "macd": self.macd,
            "macro": self.macro,
            "fracdiff": self.fracdiff,
            "fracdiff_macro": self.fracdiff_macro,
            "trend": self.trend,
            "asset_realized_vol": self.asset_vol,
            "macro_realized_vol": self.macro_vol,
            "block_boundaries": self.block_boundaries,
            "asset_live": self.asset_live,
        }


def align_panel_to_timeline(
    source_index: pd.DatetimeIndex,
    target_index: pd.DatetimeIndex,
    *arrays: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Row-select panel arrays from ``source_index`` onto ``target_index`` (e.g. after holdout cut)."""
    if len(source_index) == len(target_index) and source_index.equals(target_index):
        return arrays
    loc = source_index.get_indexer(target_index)
    if np.any(loc < 0):
        raise ValueError("target_index contains dates not found in source_index")
    return tuple(np.asarray(a)[loc] for a in arrays)


def train_test_split_alternating(
    index: pd.DatetimeIndex,
    ohlcv: np.ndarray,
    macro: np.ndarray,
    asset_live: np.ndarray | None = None,
    block_size: int = 126,
    eval_stride: int = 4,
    fracdiff_d: float = DEFAULT_FRACDIFF_D,
    feature_purge_warmup: int = 25,
    feature_split_mode: str = "independent",
    feature_preroll_bars: int = 252,
    *,
    rsi: np.ndarray | None = None,
    macd: np.ndarray | None = None,
    fracdiff: np.ndarray | None = None,
    fracdiff_macro: np.ndarray | None = None,
    trend: np.ndarray | None = None,
    asset_vol: np.ndarray | None = None,
    macro_vol: np.ndarray | None = None,
) -> Tuple[Tuple, Tuple]:
    """Walk-forward alternating split for representative train/eval regimes.

    Divides the timeline into blocks of ``block_size`` trading bars (~6 months
    at 126).  Every ``eval_stride``-th block is assigned to eval; the rest go
    to train.  Both sets therefore span the full history and contain a mix of
    bull, bear, high-vol, and low-vol periods.

    ``feature_split_mode`` controls how features reach the blocks:

    - ``"independent"`` (default): features are **recomputed per contiguous segment** via
      ``compute_feature_panel`` over the segment plus a causal preroll of up to
      ``feature_preroll_bars`` earlier panel bars (sliced off after computation), so
      slow indicators (EMA-100 trend, MACD, fracdiff) are warmed up with real history
      instead of a truncation transient. Segment-head bars whose preroll is shorter
      than ``feature_purge_warmup`` (the panel head) are neutralized via
      ``_neutralize_feature_warmup``. Any precomputed feature arrays passed in are
      ignored. Each segment's features depend only on a bounded, uniform history
      window — train and eval feature distributions stay comparable, and eval blocks
      do not share exact continuous-panel indicator state with the train blocks
      around them.
    - ``"continuous"``: features are precomputed on the contiguous trainable
      timeline and **sliced** into blocks. Pass precomputed ``rsi`` / ``macd`` /
      ``fracdiff`` / ``fracdiff_macro`` / ``trend`` / ``asset_vol`` / ``macro_vol`` (e.g.
      from ``load_cache``) to avoid recomputing on every launch; otherwise they are built
      once via ``compute_feature_panel``. This matches continuous backtest memory, so
      ``feature_purge_warmup`` is **not** applied (segment heads carry indicator state
      continuous with the original timeline).

    Block joins still set ``block_boundaries`` so episodes do not cross calendar gaps.

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
    if asset_live is None:
        asset_live = np.ones((n, ohlcv.shape[1]), dtype=np.float64)
    elif asset_live.shape != (n, ohlcv.shape[1]):
        raise ValueError(
            f"asset_live shape {asset_live.shape} != ({n}, {ohlcv.shape[1]})"
        )

    n_blocks = (n + block_size - 1) // block_size

    if feature_split_mode not in ("continuous", "independent"):
        raise ValueError(
            f"feature_split_mode must be 'continuous' or 'independent', got {feature_split_mode!r}"
        )
    independent = feature_split_mode == "independent"

    if independent:
        # Per-segment recompute + purge happens inside _concat_ranges; ignore any
        # precomputed continuous-panel features that were passed in.
        rsi_g = macd_g = fd_g = fdm_g = trend_g = avol_g = mvol_g = None
    else:
        precomputed = (rsi, macd, fracdiff, fracdiff_macro, trend, asset_vol, macro_vol)
        if all(x is not None for x in precomputed):
            rsi_g, macd_g, fd_g, fdm_g, trend_g, avol_g, mvol_g = precomputed  # type: ignore[misc]
            for name, arr in (
                ("rsi", rsi_g),
                ("macd", macd_g),
                ("fracdiff", fd_g),
                ("fracdiff_macro", fdm_g),
                ("trend", trend_g),
                ("asset_vol", avol_g),
                ("macro_vol", mvol_g),
            ):
                if arr.shape[0] != n:
                    raise ValueError(f"{name} length {arr.shape[0]} != timeline length {n}")
        else:
            if any(x is not None for x in precomputed):
                raise ValueError(
                    "train_test_split_alternating: pass all of rsi, macd, fracdiff, "
                    "fracdiff_macro, trend, asset_vol, macro_vol, or none to recompute"
                )
            rsi_g, macd_g, fd_g, fdm_g, trend_g, avol_g, mvol_g = compute_feature_panel(
                ohlcv, macro, fracdiff_d=fracdiff_d, asset_live=asset_live
            )

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
        np.ndarray,
        List[int],
    ]:
        if not ranges:
            raise ValueError(
                "train_test_split_alternating: empty block set (block_size too large or "
                f"eval_stride too small for {n} bars); widen the date range or adjust split."
            )
        idx_parts: List[pd.DatetimeIndex] = []
        ohlcv_parts: List[np.ndarray] = []
        live_parts: List[np.ndarray] = []
        rsi_parts: List[np.ndarray] = []
        macd_parts: List[np.ndarray] = []
        macro_parts: List[np.ndarray] = []
        fd_parts: List[np.ndarray] = []
        fdm_parts: List[np.ndarray] = []
        trend_parts: List[np.ndarray] = []
        avol_parts: List[np.ndarray] = []
        mvol_parts: List[np.ndarray] = []
        boundaries: List[int] = []
        offset = 0

        if independent:
            # Merge contiguous block ranges into segments, then recompute features per
            # segment over [seg_start - preroll, seg_end) and slice the preroll off.
            # The causal preroll gives every indicator real warmup history (EMA-100,
            # MACD, fracdiff transients decay over hundreds of bars — a 25-bar purge
            # alone left every eval bar in a 126-bar block feature-distribution-shifted
            # vs training). Only bars whose preroll is shorter than
            # ``feature_purge_warmup`` (the panel head) are still neutralized.
            segments: List[List[int]] = []
            for s, e in ranges:
                if segments and segments[-1][1] == s:
                    segments[-1][1] = e
                else:
                    segments.append([s, e])
            for k, (seg_start, seg_end) in enumerate(segments):
                if k > 0:
                    boundaries.append(offset)
                pre = min(max(int(feature_preroll_bars), 0), seg_start)
                comp_start = seg_start - pre  # causal: earlier panel history only
                ohlcv_chunk = ohlcv[seg_start:seg_end]
                macro_chunk = macro[seg_start:seg_end]
                rsi_c, macd_c, fd_c, fdm_c, trend_c, avol_c, mvol_c = compute_feature_panel(
                    ohlcv[comp_start:seg_end],
                    macro[comp_start:seg_end],
                    fracdiff_d=fracdiff_d,
                    asset_live=asset_live[comp_start:seg_end],
                )
                # Neutralize the computed window's warmup head; with pre >= warmup these
                # bars all fall inside the preroll and are sliced away below.
                _neutralize_feature_warmup(
                    rsi_c, macd_c, fd_c, fdm_c, feature_purge_warmup, trend=trend_c
                )
                rsi_c, macd_c, fd_c, fdm_c, trend_c, avol_c, mvol_c = (
                    arr[pre:] for arr in (rsi_c, macd_c, fd_c, fdm_c, trend_c, avol_c, mvol_c)
                )
                idx_parts.append(index[seg_start:seg_end])
                ohlcv_parts.append(ohlcv_chunk)
                live_parts.append(asset_live[seg_start:seg_end])
                rsi_parts.append(rsi_c)
                macd_parts.append(macd_c)
                macro_parts.append(macro_chunk)
                fd_parts.append(fd_c)
                fdm_parts.append(fdm_c)
                trend_parts.append(trend_c)
                avol_parts.append(avol_c)
                mvol_parts.append(mvol_c)
                offset += seg_end - seg_start
        else:
            prev_orig_end: Optional[int] = None
            for orig_start, orig_end in ranges:
                if prev_orig_end is not None and orig_start != prev_orig_end:
                    boundaries.append(offset)

                ohlcv_chunk = ohlcv[orig_start:orig_end]
                idx_parts.append(index[orig_start:orig_end])
                ohlcv_parts.append(ohlcv_chunk)
                live_parts.append(asset_live[orig_start:orig_end])
                rsi_parts.append(rsi_g[orig_start:orig_end])
                macd_parts.append(macd_g[orig_start:orig_end])
                macro_parts.append(macro[orig_start:orig_end])
                fd_parts.append(fd_g[orig_start:orig_end])
                fdm_parts.append(fdm_g[orig_start:orig_end])
                trend_parts.append(trend_g[orig_start:orig_end])
                avol_parts.append(avol_g[orig_start:orig_end])
                mvol_parts.append(mvol_g[orig_start:orig_end])
                offset += orig_end - orig_start
                prev_orig_end = orig_end

        cat_idx = idx_parts[0].append(idx_parts[1:]) if len(idx_parts) > 1 else idx_parts[0]
        cat_ohlcv = np.concatenate(ohlcv_parts, axis=0)
        cat_live = np.concatenate(live_parts, axis=0)
        cat_rsi = np.concatenate(rsi_parts, axis=0)
        cat_macd = np.concatenate(macd_parts, axis=0)
        cat_macro = np.concatenate(macro_parts, axis=0)
        cat_fd = np.concatenate(fd_parts, axis=0)
        cat_fdm = np.concatenate(fdm_parts, axis=0)
        cat_trend = np.concatenate(trend_parts, axis=0)
        cat_avol = np.concatenate(avol_parts, axis=0)
        cat_mvol = np.concatenate(mvol_parts, axis=0)
        return (
            cat_idx,
            cat_ohlcv,
            cat_live,
            cat_rsi,
            cat_macd,
            cat_macro,
            cat_fd,
            cat_fdm,
            cat_trend,
            cat_avol,
            cat_mvol,
            boundaries,
        )

    (
        tr_idx,
        tr_ohlcv,
        tr_live,
        tr_rsi,
        tr_macd,
        tr_macro,
        tr_fd,
        tr_fdm,
        tr_trend,
        tr_avol,
        tr_mvol,
        tr_bounds,
    ) = _concat_ranges(train_ranges)
    (
        ev_idx,
        ev_ohlcv,
        ev_live,
        ev_rsi,
        ev_macd,
        ev_macro,
        ev_fd,
        ev_fdm,
        ev_trend,
        ev_avol,
        ev_mvol,
        ev_bounds,
    ) = _concat_ranges(eval_ranges)

    return (
        (
            tr_idx,
            tr_ohlcv,
            tr_rsi,
            tr_macd,
            tr_macro,
            tr_fd,
            tr_fdm,
            tr_trend,
            tr_avol,
            tr_mvol,
            tr_bounds,
            tr_live,
        ),
        (
            ev_idx,
            ev_ohlcv,
            ev_rsi,
            ev_macd,
            ev_macro,
            ev_fd,
            ev_fdm,
            ev_trend,
            ev_avol,
            ev_mvol,
            ev_bounds,
            ev_live,
        ),
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
    asset_vol: np.ndarray,
    macro_vol: np.ndarray,
    asset_live: np.ndarray | None = None,
    fracdiff_d: float = DEFAULT_FRACDIFF_D,
    tickers: Sequence[str] | None = None,
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    idx = index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    active_tickers = np.asarray(resolve_tickers(tickers), dtype=object)
    if asset_live is None:
        asset_live = np.ones((ohlcv.shape[0], ohlcv.shape[1]), dtype=np.float64)
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
        asset_vol=asset_vol,
        macro_vol=macro_vol,
        asset_live=asset_live,
        fracdiff_d=np.array([fracdiff_d], dtype=np.float64),
        tickers=active_tickers,
        # Distribution-adjusted (total-return) OHLC since the auto_adjust=True switch;
        # load_cache warns when an old price-return cache is loaded.
        prices_adjusted=np.array([True]),
    )


def load_cache(
    path: str,
    expected_fracdiff_d: float | None = None,
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
    if "prices_adjusted" not in z.files or not bool(np.asarray(z["prices_adjusted"]).ravel()[0]):
        print(
            f"[data] WARNING: cache {path} predates the total-return switch "
            "(price-return OHLC: ETF dividends/coupons missing). For NEW training or "
            "published numbers, rebuild with --refresh-data; for reproducing an old "
            "run's OOS numbers from its snapshot this is expected — do not rebuild."
        )
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
    if "asset_live" in z.files:
        asset_live = z["asset_live"]
    else:
        asset_live = np.ones((len(idx), ohlcv.shape[1]), dtype=np.float64)
    d = float(z["fracdiff_d"][0]) if "fracdiff_d" in z else DEFAULT_FRACDIFF_D
    recompute_fd = not ("fracdiff" in z.files and "fracdiff_macro" in z.files)
    if not recompute_fd:
        fd = z["fracdiff"]
        fdm = z["fracdiff_macro"]
        recompute_fd = fdm.shape[1] != N_MACRO
    if expected_fracdiff_d is not None and abs(float(expected_fracdiff_d) - d) > 1e-12:
        # Without this check, changing data.fracdiff_d in config without --refresh-data
        # would silently train on stale-d features while the manifest records the new d.
        print(
            f"[data] cache fracdiff_d={d} != config fracdiff_d={expected_fracdiff_d}; "
            "recomputing the fracdiff panel from cached prices."
        )
        d = float(expected_fracdiff_d)
        recompute_fd = True
    if recompute_fd:
        fd, fdm = compute_fracdiff_panel(ohlcv, macro, d=d)
        fd[np.asarray(asset_live, dtype=np.float64) < 0.5] = 0.0  # pre-live neutral
    if "trend" in z.files:
        trend = z["trend"]
    else:
        trend = compute_trend_signals(ohlcv)
    if "asset_vol" in z.files and "macro_vol" in z.files:
        asset_vol = z["asset_vol"]
        macro_vol = z["macro_vol"]
    else:
        asset_vol, macro_vol = compute_realized_vol_panels(ohlcv, macro)
    if ohlcv.shape[1] != len(tickers):
        raise ValueError(
            f"Cache ohlcv has {ohlcv.shape[1]} assets but tickers lists {len(tickers)}: {tickers}"
        )
    return idx, ohlcv, z["rsi"], z["macd"], macro, fd, fdm, trend, asset_vol, macro_vol, asset_live, tickers
