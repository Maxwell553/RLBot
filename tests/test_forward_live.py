"""Torch-free live forward MTM helpers (30m candles + legacy close path)."""

from __future__ import annotations

import numpy as np

from rlbot.forward_live import (
    _ew_nav,
    _nav_from_weights,
    _spy_nav,
    _weight_vector,
    equal_weight_ohlc_candles,
    portfolio_ohlc_candles,
    spy_ohlc_candles,
)


def test_weight_vector_normalizes_cash_alias() -> None:
    labels = ["Cash", "SP500", "GOLD"]
    w = _weight_vector({"CASH": 0.5, "SP500": 0.25, "GOLD": 0.25}, labels)
    assert abs(w.sum() - 1.0) < 1e-9
    assert abs(w[0] - 0.5) < 1e-9


def test_nav_from_weights_cash_only_flat_without_yield() -> None:
    closes = np.array([[100.0, 50.0], [110.0, 55.0], [121.0, 60.5]], dtype=np.float64)
    w = [_weight_vector({"Cash": 1.0}, ["Cash", "A", "B"])] * 3
    nav = _nav_from_weights(closes, w, initial_cash=100_000.0, cash_daily_yield=0.0)
    assert nav.shape == (3,)
    assert abs(nav[0] - 100_000.0) < 1e-6
    assert abs(nav[-1] - 100_000.0) < 1e-6


def test_nav_from_weights_full_invest_tracks_asset() -> None:
    closes = np.array([[100.0], [110.0], [121.0]], dtype=np.float64)
    w = [_weight_vector({"Cash": 0.0, "A": 1.0}, ["Cash", "A"])] * 3
    nav = _nav_from_weights(closes, w, initial_cash=100_000.0)
    assert abs(nav[-1] / nav[0] - 1.21) < 1e-9


def test_benchmark_navs() -> None:
    closes = np.array([[100.0, 200.0], [110.0, 180.0]], dtype=np.float64)
    ew = _ew_nav(closes, 100_000.0)
    # mean return = 0.5*(0.10 + -0.10) = 0
    assert abs(ew[-1] - 100_000.0) < 1e-6
    spy = _spy_nav(np.array([50.0, 55.0]), 100_000.0)
    assert abs(spy[-1] - 110_000.0) < 1e-6


def test_portfolio_ohlc_full_invest_tracks_asset_bar() -> None:
    # One asset: open 100 → close 110 on bar 0; bar 1 open 110 → close 121.
    open_ = np.array([[100.0], [110.0]], dtype=np.float64)
    high = np.array([[112.0], [122.0]], dtype=np.float64)
    low = np.array([[99.0], [109.0]], dtype=np.float64)
    close = np.array([[110.0], [121.0]], dtype=np.float64)
    w = _weight_vector({"Cash": 0.0, "A": 1.0}, ["Cash", "A"])
    candles = portfolio_ohlc_candles(
        open_, high, low, close, w, initial_cash=100_000.0, cash_yield_per_bar=0.0
    )
    assert candles.shape == (2, 4)
    assert abs(candles[0, 0] - 100_000.0) < 1e-6
    assert abs(candles[0, 3] - 110_000.0) < 1e-6
    assert abs(candles[1, 3] - 121_000.0) < 1e-6
    assert candles[0, 1] >= candles[0, 0] and candles[0, 1] >= candles[0, 3]
    assert candles[0, 2] <= candles[0, 0] and candles[0, 2] <= candles[0, 3]


def test_cash_park_candles_earn_yield() -> None:
    open_ = np.array([[100.0], [100.0]], dtype=np.float64)
    high = open_.copy()
    low = open_.copy()
    close = open_.copy()
    w = _weight_vector({"Cash": 1.0}, ["Cash", "A"])
    candles = portfolio_ohlc_candles(
        open_, high, low, close, w, initial_cash=100_000.0, cash_yield_per_bar=0.001
    )
    # Bar 0: open=initial; close earns one bar of yield from the open base.
    assert abs(candles[0, 0] - 100_000.0) < 1e-6
    assert abs(candles[0, 3] - 100_100.0) < 1e-6


def test_spy_and_ew_candle_helpers() -> None:
    o = np.array([50.0, 55.0])
    h = np.array([51.0, 56.0])
    l = np.array([49.0, 54.0])
    c = np.array([50.5, 55.5])
    spy = spy_ohlc_candles(o, h, l, c, initial_cash=100_000.0)
    assert abs(spy[0, 0] - 100_000.0) < 1e-6
    assert abs(spy[1, 3] / spy[0, 0] - 55.5 / 50.0) < 1e-9

    open_ = np.array([[100.0, 200.0], [110.0, 180.0]])
    high = open_ + 1
    low = open_ - 1
    close = np.array([[110.0, 180.0], [121.0, 162.0]])
    ew = equal_weight_ohlc_candles(open_, high, low, close, initial_cash=100_000.0)
    assert ew.shape == (2, 4)
    assert abs(ew[0, 0] - 100_000.0) < 1e-6
