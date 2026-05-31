"""Core math and allocation smoke tests (no network, no full training loop)."""

from __future__ import annotations

import numpy as np
import pytest

from data_utils import (
    N_MACRO,
    MACRO_TICKERS,
    TICKERS,
    _hy_oas_proxy_pct,
    compute_trend_signals,
    fracdiff_weights,
)
from rl_config import get_config


def test_benchmark_cap_weights_normalize() -> None:
    w = get_config().reward.benchmark_cap_weights_array()
    assert w.shape == (10,)
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


def test_trend_signals_shape() -> None:
    t, n = 120, len(TICKERS)
    ohlcv = np.random.rand(t, n, 5) * 50 + 100
    ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
    trend = compute_trend_signals(ohlcv)
    assert trend.shape == (t, n)
    assert np.all(np.isfinite(trend[-1]))


def test_equal_weight_buyhold_nav_length() -> None:
    from backtest import _equal_weight_buyhold_nav, _spy_buyhold_nav

    n_assets, start = 10, 2
    navs = np.linspace(100_000.0, 105_000.0, 20)
    t = start + len(navs)
    ohlcv = np.random.rand(t, n_assets, 5) * 100 + 50
    ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
    ew = _equal_weight_buyhold_nav(navs, ohlcv, start)
    spy = _spy_buyhold_nav(navs, ohlcv, start)
    assert len(ew) == len(navs)
    assert len(spy) == len(navs)
    assert ew[0] == pytest.approx(navs[0])
