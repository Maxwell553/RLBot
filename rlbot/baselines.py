"""
Passive OOS benchmark NAV paths (buy-and-hold style).

Execution matches ``MultiAssetPortfolioEnv``: MTM at ``close[t]``, rebalance at
``open[t+1]``, mark at ``close[t+1]``. Multi-asset books aggregate **simple**
returns cross-sectionally, then compound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rlbot.data_utils import benchmark_ohlcv_index, resolve_tickers
from rlbot.rl_config import get_config

DEFAULT_VOL_LOOKBACK = 20


def _live_mask(
    asset_live: np.ndarray | None,
    t: int,
    n_assets: int,
) -> np.ndarray:
    """Boolean mask of tradeable assets at bar ``t`` (excludes pre-IPO / flat bfill rows)."""
    if asset_live is None:
        return np.ones(n_assets, dtype=bool)
    row = np.asarray(asset_live[t], dtype=np.float64).reshape(-1)
    if row.shape[0] != n_assets:
        raise ValueError(f"asset_live[{t}] must have length {n_assets}, got {row.shape[0]}")
    return row > 0.5


def _weights_on_live(weights: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Zero dead assets and renormalize to a fully invested risky book."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1) * live.astype(np.float64)
    total = float(w.sum())
    if total > 1e-12:
        return w / total
    n_live = int(live.sum())
    if n_live < 1:
        return np.zeros_like(w)
    out = np.zeros_like(w)
    out[live] = 1.0 / n_live
    return out


def _transaction_cost_arrays(n_assets: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tc = get_config().transaction_costs
    slip = tc.slippage_array()
    fee = tc.tx_fee_array()
    hold = tc.daily_holding_cost_array()
    if slip.shape[0] != n_assets:
        raise ValueError(f"slippage length {slip.shape[0]} != n_assets {n_assets}")
    return slip, fee, hold


def portfolio_step_nav(
    prev_nav: float,
    ohlcv: np.ndarray,
    t: int,
    weights: np.ndarray,
    *,
    prev_weights: np.ndarray | None = None,
    slippage: np.ndarray | None = None,
    tx_fee: np.ndarray | None = None,
    daily_holding: np.ndarray | None = None,
    fee_scale: float = 1.0,
    asset_live: np.ndarray | None = None,
) -> float:
    """
    One step aligned with the RL env at decision bar ``t``.

    Overnight return on ``prev_weights`` (close[t] → open[t+1]), rebalance at
    open[t+1], then intraday return on ``weights`` (open[t+1] → close[t+1]).
    """
    n_assets = ohlcv.shape[1]
    live = _live_mask(asset_live, t, n_assets)
    w_tgt = _weights_on_live(
        np.asarray(weights, dtype=np.float64).reshape(-1), live
    )
    if w_tgt.shape[0] != n_assets:
        raise ValueError(f"weights must have length {n_assets}, got {w_tgt.shape[0]}")

    close_pre = ohlcv[t, :, 3]
    open_exec = ohlcv[t + 1, :, 0]
    close_post = ohlcv[t + 1, :, 3]

    if prev_weights is None:
        w_prev = w_tgt.copy()
    else:
        w_prev = _weights_on_live(
            np.asarray(prev_weights, dtype=np.float64).reshape(-1), live
        )

    log_on = np.log((open_exec + 1e-12) / (close_pre + 1e-12))
    simple_on = np.expm1(log_on)

    if slippage is None and tx_fee is None and daily_holding is None:
        log_id = np.log((close_post + 1e-12) / (open_exec + 1e-12))
        simple_id = np.expm1(log_id)
        r_on = float(np.dot(w_prev, simple_on))
        r_id = float(np.dot(w_tgt, simple_id))
        return float(prev_nav * (1.0 + r_on) * (1.0 + r_id))

    slip = np.zeros(n_assets) if slippage is None else np.asarray(slippage, dtype=np.float64)
    fee = np.zeros(n_assets) if tx_fee is None else np.asarray(tx_fee, dtype=np.float64)
    hold = np.zeros(n_assets) if daily_holding is None else np.asarray(daily_holding, dtype=np.float64)
    fs = float(fee_scale)

    holding_cost = float(prev_nav * np.dot(w_prev, hold) * fs)
    nav_mid = max(prev_nav - holding_cost, 1e-12)
    nav_mid = float(nav_mid * (1.0 + float(np.dot(w_prev, simple_on))))

    gross = float(np.dot(w_prev, 1.0 + simple_on))
    if gross > 1e-12:
        w_drift = w_prev * (1.0 + simple_on) / gross
    else:
        w_drift = w_prev.copy()

    delta = w_tgt - w_drift
    trade_cost = float(np.sum(np.abs(delta) * nav_mid * (slip + fee) * fs))
    nav_mid = max(nav_mid - trade_cost, 1e-12)

    log_id = np.log((close_post + 1e-12) / (open_exec + 1e-12))
    simple_id = np.expm1(log_id)
    return float(nav_mid * (1.0 + float(np.dot(w_tgt, simple_id))))


def _asset_vol_fully_warmed(
    asset_live: np.ndarray | None,
    t: int,
    asset_i: int,
    lookback: int,
) -> bool:
    """True when ``asset_i`` has a live print on every bar in the vol window through ``t``."""
    if asset_live is None:
        return True
    window_start = max(t - lookback, 0)
    window_len = t + 1 - window_start
    return int(np.sum(asset_live[window_start : t + 1, asset_i] > 0.5)) >= window_len


def _risk_parity_weights(
    close: np.ndarray,
    t: int,
    lookback: int,
    live: np.ndarray,
    *,
    asset_live: np.ndarray | None,
    n_assets: int,
) -> np.ndarray:
    """Inverse-vol weights using only data available through decision bar ``t``."""
    vol = realized_vol_at_bar(close, t, lookback)
    inv = np.zeros(n_assets, dtype=np.float64)
    live_idx = np.where(live)[0]
    for i in live_idx:
        if _asset_vol_fully_warmed(asset_live, t, int(i), lookback):
            v = float(vol[i])
        else:
            warmed = vol[live]
            v = float(np.mean(warmed[warmed > 1e-12])) if np.any(warmed > 1e-12) else 1.0
        if v > 1e-12:
            inv[i] = 1.0 / v
    if float(inv.sum()) > 1e-12:
        return inv / inv.sum()
    return _weights_on_live(np.ones(n_assets), live)


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
    asset_live: np.ndarray | None = None,
    fee_scale: float = 1.0,
    apply_costs: bool = True,
) -> np.ndarray:
    """Benchmark sleeve buy-and-hold NAV aligned to model ``navs`` (length and start)."""
    col = (
        benchmark_ohlcv_index(tickers)
        if benchmark_col is None
        else benchmark_col
    )
    close = ohlcv_window[:, col, 3].astype(np.float64, copy=False)
    i0, i1 = int(start_bar), int(start_bar + len(navs) - 1)
    n = len(navs)
    out = np.empty(n, dtype=np.float64)
    out[0] = float(navs[0])
    if not apply_costs:
        s0 = max(close[i0], 1e-12)
        return (close[i0 : i1 + 1] / s0) * float(navs[0])

    n_assets = ohlcv_window.shape[1]
    slip, fee, hold = _transaction_cost_arrays(n_assets)
    w = np.zeros(n_assets, dtype=np.float64)
    w[col] = 1.0
    prev_w = w.copy()
    for k in range(1, n):
        t = i0 + k - 1
        out[k] = portfolio_step_nav(
            out[k - 1],
            ohlcv_window,
            t,
            w,
            prev_weights=prev_w,
            slippage=slip,
            tx_fee=fee,
            daily_holding=hold,
            fee_scale=fee_scale,
            asset_live=asset_live,
        )
        prev_w = w
    return out


def cash_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    **_,
) -> np.ndarray:
    """100% cash / no-trade: flat NAV (no risk-free accrual in the daily bar model)."""
    del ohlcv_window, start_bar
    out = np.empty(len(navs), dtype=np.float64)
    out[:] = float(navs[0])
    return out


def benchmark_only_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    *,
    tickers: list[str] | None = None,
    benchmark_col: int | None = None,
    asset_live: np.ndarray | None = None,
    fee_scale: float = 1.0,
    apply_costs: bool = True,
) -> np.ndarray:
    """100% benchmark sleeve (config ``universe.benchmark``), buy-and-hold with costs."""
    return benchmark_buyhold_nav(
        navs,
        ohlcv_window,
        start_bar,
        tickers=tickers,
        benchmark_col=benchmark_col,
        asset_live=asset_live,
        fee_scale=fee_scale,
        apply_costs=apply_costs,
    )


def equal_weight_buyhold_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    *,
    asset_live: np.ndarray | None = None,
    fee_scale: float = 1.0,
    apply_costs: bool = True,
) -> np.ndarray:
    """Equal-weight (1/N) daily-rebalanced passive book on all tradeable assets."""
    close = ohlcv_window[:, :, 3].astype(np.float64, copy=False)
    n_assets = close.shape[1]
    i0 = int(start_bar)
    n = len(navs)
    slip = fee = hold = None
    if apply_costs:
        slip, fee, hold = _transaction_cost_arrays(n_assets)
    out = np.empty(n, dtype=np.float64)
    out[0] = float(navs[0])
    prev_w = np.zeros(n_assets, dtype=np.float64)
    for k in range(1, n):
        t = i0 + k - 1
        live = _live_mask(asset_live, t, n_assets)
        w = _weights_on_live(np.full(n_assets, 1.0 / max(int(live.sum()), 1)), live)
        out[k] = portfolio_step_nav(
            out[k - 1],
            ohlcv_window,
            t,
            w,
            prev_weights=prev_w,
            slippage=slip,
            tx_fee=fee,
            daily_holding=hold,
            fee_scale=fee_scale,
            asset_live=asset_live,
        )
        prev_w = w.copy()
    return out


def equal_weight_daily_cost_aware_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    *,
    asset_live: np.ndarray | None = None,
    fee_scale: float = 1.0,
) -> np.ndarray:
    """Equal-weight 1/N among live assets, **daily** rebalance, slippage + fees + holding costs."""
    return equal_weight_buyhold_nav(
        navs,
        ohlcv_window,
        start_bar,
        asset_live=asset_live,
        fee_scale=fee_scale,
        apply_costs=True,
    )


def equal_weight_monthly_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    test_idx: pd.DatetimeIndex,
    *,
    asset_live: np.ndarray | None = None,
    fee_scale: float = 1.0,
    apply_costs: bool = True,
) -> np.ndarray:
    """Equal-weight 1/N among live assets, rebalanced on the first bar of each calendar month."""
    n_assets = ohlcv_window.shape[1]
    i0 = int(start_bar)
    n = len(navs)
    slip = fee = hold = None
    if apply_costs:
        slip, fee, hold = _transaction_cost_arrays(n_assets)
    out = np.empty(n, dtype=np.float64)
    out[0] = float(navs[0])
    prev_month: tuple[int, int] | None = None
    prev_w = np.zeros(n_assets, dtype=np.float64)
    w = prev_w.copy()
    for k in range(1, n):
        t = i0 + k - 1
        bar_ts = test_idx[t + 1]
        month_key = (int(bar_ts.year), int(bar_ts.month))
        live = _live_mask(asset_live, t, n_assets)
        if prev_month != month_key:
            w = _weights_on_live(np.full(n_assets, 1.0 / max(int(live.sum()), 1)), live)
            prev_month = month_key
        else:
            w = _weights_on_live(w, live)
        out[k] = portfolio_step_nav(
            out[k - 1],
            ohlcv_window,
            t,
            w,
            prev_weights=prev_w,
            slippage=slip,
            tx_fee=fee,
            daily_holding=hold,
            fee_scale=fee_scale,
            asset_live=asset_live,
        )
        prev_w = w.copy()
    return out


def balanced_6040_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    test_idx: pd.DatetimeIndex,
    tickers: list[str] | None = None,
    *,
    asset_live: np.ndarray | None = None,
    fee_scale: float = 1.0,
    apply_costs: bool = True,
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
    slip = fee = hold = None
    if apply_costs:
        slip, fee, hold = _transaction_cost_arrays(n_assets)
    i0 = int(start_bar)
    n = len(navs)
    out = np.empty(n, dtype=np.float64)
    out[0] = float(navs[0])
    prev_month: tuple[int, int] | None = None
    prev_w = np.zeros(n_assets, dtype=np.float64)
    for k in range(1, n):
        t = i0 + k - 1
        bar_ts = test_idx[t + 1]
        month_key = (int(bar_ts.year), int(bar_ts.month))
        if prev_month != month_key:
            raw = np.zeros(n_assets, dtype=np.float64)
            raw[spy_i] = 0.6
            raw[bond_i] = 0.4
            live = _live_mask(asset_live, t, n_assets)
            w = _weights_on_live(raw, live)
            prev_month = month_key
        else:
            live = _live_mask(asset_live, t, n_assets)
            w = _weights_on_live(w, live)
        out[k] = portfolio_step_nav(
            out[k - 1],
            ohlcv_window,
            t,
            w,
            prev_weights=prev_w,
            slippage=slip,
            tx_fee=fee,
            daily_holding=hold,
            fee_scale=fee_scale,
            asset_live=asset_live,
        )
        prev_w = w.copy()
    return out


def naive_risk_parity_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    lookback: int = DEFAULT_VOL_LOOKBACK,
    *,
    asset_live: np.ndarray | None = None,
    fee_scale: float = 1.0,
    apply_costs: bool = True,
) -> np.ndarray:
    """Inverse-vol weights on listed assets only, daily rebalance, fully invested."""
    close = ohlcv_window[:, :, 3].astype(np.float64, copy=False)
    n_assets = close.shape[1]
    i0 = int(start_bar)
    n = len(navs)
    slip = fee = hold = None
    if apply_costs:
        slip, fee, hold = _transaction_cost_arrays(n_assets)
    out = np.empty(n, dtype=np.float64)
    out[0] = float(navs[0])
    prev_w = np.zeros(n_assets, dtype=np.float64)
    for k in range(1, n):
        t = i0 + k - 1
        live = _live_mask(asset_live, t, n_assets)
        w = _risk_parity_weights(
            close,
            t,
            lookback,
            live,
            asset_live=asset_live,
            n_assets=n_assets,
        )
        out[k] = portfolio_step_nav(
            out[k - 1],
            ohlcv_window,
            t,
            w,
            prev_weights=prev_w,
            slippage=slip,
            tx_fee=fee,
            daily_holding=hold,
            fee_scale=fee_scale,
            asset_live=asset_live,
        )
        prev_w = w.copy()
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
