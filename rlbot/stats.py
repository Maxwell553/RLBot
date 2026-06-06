"""Pure-numpy return statistics (annualized Sharpe, stationary block bootstrap).

Kept dependency-free (numpy only) so unit tests and analysis tools can import it
without pulling in torch via ``scripts/backtest.py``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def sharpe_ann_from_log_rets(log_rets: np.ndarray) -> float:
    """Annualized (252) Sharpe from daily log returns; NaN if < 2 samples."""
    log_rets = np.asarray(log_rets, dtype=np.float64).reshape(-1)
    if log_rets.size < 2:
        return float("nan")
    return float(np.mean(log_rets) / (np.std(log_rets) + 1e-12) * np.sqrt(252))


def block_bootstrap_log_rets(
    log_rets: np.ndarray,
    n_resamples: int = 5000,
    avg_block_size: int = 10,
    seed: int = 42,
    *,
    progress: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Stationary (Politis–Romano style) block bootstrap Sharpe samples.

    Builds synthetic return series by stitching contiguous blocks; block breaks
    arrive with geometric probability ``1 / avg_block_size``. Indices wrap
    circularly to preserve length without edge truncation.
    """
    log_rets = np.asarray(log_rets, dtype=np.float64).reshape(-1)
    n = log_rets.size
    if n < 2:
        return np.full(n_resamples, np.nan, dtype=np.float64)
    boot_sharpes = np.empty(n_resamples, dtype=np.float64)
    p = 1.0 / max(int(avg_block_size), 1)
    rng = np.random.default_rng(seed)
    emit = log_fn if log_fn is not None else print

    log_stride = max(n_resamples // 5, 1) if progress and n_resamples >= 500 else 0
    for b in range(n_resamples):
        if log_stride and b > 0 and b % log_stride == 0:
            emit(f"[stats] bootstrap: {b}/{n_resamples} resamples...")
        sim_idx = np.empty(n, dtype=np.int64)
        curr_idx = int(rng.integers(0, n))
        for i in range(n):
            sim_idx[i] = curr_idx
            if rng.random() < p:
                curr_idx = int(rng.integers(0, n))
            else:
                curr_idx = (curr_idx + 1) % n
        sample_rets = log_rets[sim_idx]
        boot_sharpes[b] = sharpe_ann_from_log_rets(sample_rets)
    return boot_sharpes


def block_bootstrap_sharpe_percentiles(
    log_rets: np.ndarray,
    n_resamples: int = 5000,
    avg_block_size: int = 10,
    seed: int = 42,
    *,
    progress: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float, float]:
    """2.5 / 50 / 97.5 percentiles of block-bootstrap annualized Sharpe."""
    sharpes = block_bootstrap_log_rets(
        log_rets,
        n_resamples=n_resamples,
        avg_block_size=avg_block_size,
        seed=seed,
        progress=progress,
        log_fn=log_fn,
    )
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(sharpes, 2.5)),
        float(np.percentile(sharpes, 50.0)),
        float(np.percentile(sharpes, 97.5)),
    )
