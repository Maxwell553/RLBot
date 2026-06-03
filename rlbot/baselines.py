"""
Passive OOS benchmark NAV paths (buy-and-hold style).

Multi-asset portfolios combine **simple** returns cross-sectionally each day
(``w · (exp(r_log) - 1)``), then compound — not a linear mix of asset log returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rlbot.data_utils import benchmark_ohlcv_index, resolve_tickers

DEFAULT_VOL_LOOKBACK = 20


def portfolio_step_nav(
    prev_nav: float,
    close: np.ndarray,
    t_prev: int,
    t_curr: int,
    weights: np.ndarray,
) -> float:
    """One step: simple-return aggregation, then NAV compound."""
    n_assets = close.shape[1]
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != n_assets:
        raise ValueError(f"weights must have length {n_assets}, got {w.shape[0]}")
    asset_log_rets = np.log((close[t_curr] + 1e-12) / (close[t_prev] + 1e-12))
    simple_rets = np.expm1(asset_log_rets)
    port_simple_ret = float(np.dot(w, simple_rets))
    return float(prev_nav * (1.0 + port_simple_ret))


def realized_vol_at_bar(
    close: np.ndarray,
    t: int,
    lookback: int = DEFAULT_VOL_LOOKBACK,
) -> np.ndarray:
    """Per-asset realized vol of daily log returns (matches trading_env)."""
    n_assets = close.shape[1]
    start = max(t - lookback, 1)
    window = close[start : t + 1]
    if len(window) < 2:
        return np.zeros(n_assets, dtype=np.float64)
    rets = np.diff(np.log(window + 1e-12), axis=0)
    return rets.std(axis=0)


def benchmark_buyhold_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    *,
    tickers: list[str] | None = None,
    benchmark_col: int | None = None,
) -> np.ndarray:
    """Benchmark sleeve buy-and-hold NAV aligned to model ``navs`` (length and start)."""
    col = (
        benchmark_ohlcv_index(tickers)
        if benchmark_col is None
        else benchmark_col
    )
    close = ohlcv_window[:, col, 3].astype(np.float64, copy=False)
    i0, i1 = int(start_bar), int(start_bar + len(navs) - 1)
    s0 = max(close[i0], 1e-12)
    return (close[i0 : i1 + 1] / s0) * float(navs[0])


def equal_weight_buyhold_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
) -> np.ndarray:
    """Equal-weight (1/N) daily-rebalanced passive book on all tradeable assets."""
    close = ohlcv_window[:, :, 3].astype(np.float64, copy=False)
    n_assets = close.shape[1]
    i0 = int(start_bar)
    n = len(navs)
    w = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    out[0] = float(navs[0])
    for k in range(1, n):
        t_prev = i0 + k - 1
        t_curr = i0 + k
        out[k] = portfolio_step_nav(out[k - 1], close, t_prev, t_curr, w)
    return out


def balanced_6040_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    test_idx: pd.DatetimeIndex,
    tickers: list[str] | None = None,
) -> np.ndarray:
    """60% SP500 / 40% BOND10Y, rebalanced on the first bar of each calendar month."""
    close = ohlcv_window[:, :, 3].astype(np.float64, copy=False)
    n_assets = close.shape[1]
    active = resolve_tickers(tickers)
    spy_i = benchmark_ohlcv_index(active)
    if "BOND10Y" not in active:
        raise KeyError("balanced_6040_nav requires BOND10Y in the tradeable universe")
    bond_i = active.index("BOND10Y")
    w = np.zeros(n_assets, dtype=np.float64)
    w[spy_i] = 0.6
    w[bond_i] = 0.4
    i0 = int(start_bar)
    n = len(navs)
    out = np.empty(n, dtype=np.float64)
    out[0] = float(navs[0])
    prev_month: tuple[int, int] | None = None
    for k in range(1, n):
        t_prev = i0 + k - 1
        t_curr = i0 + k
        bar_ts = test_idx[t_curr]
        month_key = (int(bar_ts.year), int(bar_ts.month))
        if prev_month != month_key:
            w = np.zeros(n_assets, dtype=np.float64)
            w[spy_i] = 0.6
            w[bond_i] = 0.4
            prev_month = month_key
        out[k] = portfolio_step_nav(out[k - 1], close, t_prev, t_curr, w)
    return out


def naive_risk_parity_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    lookback: int = DEFAULT_VOL_LOOKBACK,
) -> np.ndarray:
    """Inverse-vol weights on all assets, daily rebalance, fully invested."""
    close = ohlcv_window[:, :, 3].astype(np.float64, copy=False)
    n_assets = close.shape[1]
    i0 = int(start_bar)
    n = len(navs)
    w_eq = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    out[0] = float(navs[0])
    for k in range(1, n):
        t_prev = i0 + k - 1
        t_curr = i0 + k
        vol = realized_vol_at_bar(close, t_curr, lookback)
        if t_curr < lookback or not np.any(vol > 1e-12):
            w = w_eq
        else:
            inv = 1.0 / np.maximum(vol, 1e-12)
            w = inv / inv.sum()
        out[k] = portfolio_step_nav(out[k - 1], close, t_prev, t_curr, w)
    return out


def benchmark_metrics(navs: np.ndarray) -> tuple[float, float, float]:
    """Total return, annualized Sharpe from daily log rets, max drawdown."""
    navs = np.asarray(navs, dtype=np.float64)
    total_return = float(navs[-1] / navs[0] - 1.0)
    log_rets = np.diff(np.log(np.maximum(navs, 1e-12)))
    if log_rets.size < 2:
        sharpe = float("nan")
    else:
        sharpe = float(np.mean(log_rets) / (np.std(log_rets) + 1e-12) * np.sqrt(252))
    peak = np.maximum.accumulate(navs)
    dd = (navs - peak) / np.maximum(peak, 1e-12)
    max_dd = float(dd.min())
    return total_return, sharpe, max_dd
