"""60/40 benchmark gracefully unavailable on universes without BOND10Y (P1-5).

Torch-free: tests the KeyError trigger that backtest.py now guards (so detailed stats
and the dashboard skip 60/40 instead of crashing on e.g. --n-assets 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlbot.baselines import balanced_6040_nav
from rlbot.rl_config import get_config


def test_default_universe_includes_bond10y() -> None:
    assert "BOND10Y" in get_config().universe.tickers


def test_6040_raises_without_bond10y() -> None:
    T, n = 30, 5
    ohlcv = np.ones((T, n, 5), dtype=np.float64)
    navs = np.full(T, 1e5, dtype=np.float64)
    idx = pd.date_range("2020-01-01", periods=T, freq="B")
    tickers = ["SP500", "GOLD", "OIL", "EURUSD", "USDJPY"]  # no BOND10Y
    with pytest.raises(KeyError):
        balanced_6040_nav(navs, ohlcv, 0, idx, tickers=tickers)
