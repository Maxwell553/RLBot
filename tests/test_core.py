"""Core math and allocation smoke tests (no network, no full training loop)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlbot.data_utils import (
    N_MACRO,
    MACRO_TICKERS,
    _hy_oas_proxy_pct,
    compute_trend_signals,
    fracdiff_weights,
)
from rlbot.rl_config import (
    UNIVERSE_MAX_ASSETS,
    UNIVERSE_MIN_ASSETS,
    get_config,
    slice_config_to_n_assets,
    validate_config_for_universe,
    validate_universe_asset_count,
)


def test_benchmark_cap_weights_normalize() -> None:
    cfg = get_config()
    w = cfg.reward.benchmark_cap_weights_array()
    assert w.shape == (cfg.universe.n_assets,)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1]  # SP500-heavy


def test_fracdiff_weights_start_at_one() -> None:
    w = fracdiff_weights(0.4)
    assert w[0] == pytest.approx(1.0)
    assert len(w) > 10


def test_hy_oas_proxy_in_sane_range() -> None:
    hyg = np.array([80.0, 75.0, 90.0])
    ief = np.array([100.0, 100.0, 100.0])
    spread = _hy_oas_proxy_pct(hyg, ief)
    assert spread.shape == (3,)
    assert np.all(spread >= 2.0)
    assert np.all(spread <= 15.0)


def test_macro_ticker_count() -> None:
    assert N_MACRO == 4
    assert "VIX" in MACRO_TICKERS
    assert "HY_OAS" in MACRO_TICKERS


def test_universe_asset_count_bounds() -> None:
    with pytest.raises(ValueError, match="between"):
        validate_universe_asset_count(UNIVERSE_MIN_ASSETS - 1)
    with pytest.raises(ValueError, match="between"):
        validate_universe_asset_count(UNIVERSE_MAX_ASSETS + 1)
    validate_universe_asset_count(UNIVERSE_MIN_ASSETS)
    validate_universe_asset_count(UNIVERSE_MAX_ASSETS)
    validate_universe_asset_count(get_config().universe.n_assets)


def test_validate_config_for_universe_mismatch() -> None:
    cfg = get_config()
    with pytest.raises(ValueError, match="universe.assets"):
        validate_config_for_universe(cfg, cfg.universe.n_assets + 1)


def test_slice_config_to_n_assets() -> None:
    full = get_config()
    n = 7
    sliced = slice_config_to_n_assets(full, n)
    assert sliced.universe.n_assets == n
    assert sliced.universe.tickers == full.universe.tickers[:n]
    assert full.universe.benchmark in sliced.universe.assets
    w = sliced.reward.benchmark_cap_weights_array()
    assert w.shape == (n,)
    assert np.isclose(w.sum(), 1.0)
    assert len(sliced.transaction_costs.slippage) == n
    validate_config_for_universe(sliced, n)


def test_slice_config_to_n_assets_rejects_over_config() -> None:
    full = get_config()
    with pytest.raises(ValueError, match="defines only"):
        slice_config_to_n_assets(full, full.universe.n_assets + 1)


def test_trend_signals_shape() -> None:
    t, n = 120, get_config().universe.n_assets
    ohlcv = np.random.rand(t, n, 5) * 50 + 100
    ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
    trend = compute_trend_signals(ohlcv)
    assert trend.shape == (t, n)
    assert np.all(np.isfinite(trend[-1]))


def test_portfolio_step_simple_return_identity() -> None:
    from rlbot.baselines import portfolio_step_nav

    n_assets = get_config().universe.n_assets
    t = 30
    close = np.cumsum(np.random.rand(t, n_assets) * 0.01, axis=0) + 100.0
    w = np.full(n_assets, 1.0 / n_assets)
    prev = 100_000.0
    nav_next = portfolio_step_nav(prev, close, 10, 11, w)
    log_r = np.log((close[11] + 1e-12) / (close[10] + 1e-12))
    expected = prev * (1.0 + float(np.dot(w, np.expm1(log_r))))
    assert nav_next == pytest.approx(expected)


def test_equal_weight_buyhold_nav_length() -> None:
    from rlbot.baselines import benchmark_buyhold_nav, equal_weight_buyhold_nav

    n_assets = get_config().universe.n_assets
    start = 2
    navs = np.linspace(100_000.0, 105_000.0, 20)
    t = start + len(navs)
    ohlcv = np.random.rand(t, n_assets, 5) * 100 + 50
    ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
    ew = equal_weight_buyhold_nav(navs, ohlcv, start)
    spy = benchmark_buyhold_nav(navs, ohlcv, start)
    assert len(ew) == len(navs)
    assert len(spy) == len(navs)
    assert ew[0] == pytest.approx(navs[0])


def test_balanced_6040_and_risk_parity_nav() -> None:
    from rlbot.baselines import balanced_6040_nav, naive_risk_parity_nav

    n_assets = get_config().universe.n_assets
    tickers = get_config().universe.tickers
    start = 25
    navs = np.full(15, 100_000.0)
    t = start + len(navs)
    ohlcv = np.random.rand(t, n_assets, 5) * 100 + 50
    ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
    idx = pd.date_range("2020-01-01", periods=t, freq="B")
    nav_6040 = balanced_6040_nav(navs, ohlcv, start, idx, tickers=tickers)
    nav_rp = naive_risk_parity_nav(navs, ohlcv, start, lookback=20)
    assert len(nav_6040) == len(navs)
    assert len(nav_rp) == len(navs)
    assert nav_6040[0] == pytest.approx(100_000.0)
    assert nav_rp[0] == pytest.approx(100_000.0)
