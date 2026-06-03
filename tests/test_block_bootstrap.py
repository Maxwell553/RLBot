"""Stationary block bootstrap for Sharpe inference."""

from __future__ import annotations

import numpy as np

from scripts.backtest import block_bootstrap_log_rets, block_bootstrap_sharpe_percentiles


def test_block_bootstrap_preserves_length_and_finite() -> None:
    rng = np.random.default_rng(0)
    log_rets = rng.normal(0.0003, 0.01, size=120)
    sharpes = block_bootstrap_log_rets(log_rets, n_resamples=200, avg_block_size=10, seed=1)
    assert sharpes.shape == (200,)
    assert np.all(np.isfinite(sharpes))


def test_block_bootstrap_percentiles_ordered() -> None:
    rng = np.random.default_rng(2)
    log_rets = rng.normal(0.0002, 0.012, size=200)
    lo, med, hi = block_bootstrap_sharpe_percentiles(
        log_rets, n_resamples=500, avg_block_size=8, seed=3
    )
    assert lo <= med <= hi
