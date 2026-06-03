"""Eval rollout: full-segment episodes and dynamic episode count."""

from __future__ import annotations

import numpy as np

from rlbot.rl_config import get_config
from rlbot.trading_env import MultiAssetPortfolioEnv


def _minimal_env(
    n_bars: int = 200,
    boundaries: list[int] | None = None,
    *,
    random_start: bool = False,
) -> MultiAssetPortfolioEnv:
    n_a = get_config().universe.n_assets
    n_m = 4
    ohlcv = np.random.rand(n_bars, n_a, 5) * 50 + 100.0
    ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
    rsi = np.full((n_bars, n_a), 50.0)
    macd = np.zeros((n_bars, n_a))
    trend = np.zeros((n_bars, n_a))
    fd = np.zeros((n_bars, n_a))
    fdm = np.zeros((n_bars, n_m))
    macro = np.full((n_bars, n_m), 10.0)
    return MultiAssetPortfolioEnv(
        ohlcv,
        rsi,
        macd,
        fracdiff=fd,
        fracdiff_macro=fdm,
        trend=trend,
        macro=macro,
        random_start=random_start,
        max_episode_steps=63,
        domain_randomize=False,
        block_boundaries=boundaries,
    )


def test_get_segments_returns_block_list() -> None:
    env = _minimal_env(boundaries=[80, 140])
    segs = env.get_segments()
    assert segs is not None
    assert len(segs) >= 2


def test_deterministic_reset_covers_full_segment_not_63_cap() -> None:
    env = _minimal_env(n_bars=250, boundaries=[120])
    segs = env.get_segments()
    assert segs is not None and len(segs) == 2
    earliest, seg_end = segs[0]
    env.reset()
    expected_steps = max(seg_end - earliest - 2, 1)
    assert env._current_ep_max_steps == expected_steps
    assert env._current_ep_max_steps > 63


def test_eval_cycles_one_segment_per_reset() -> None:
    env = _minimal_env(n_bars=250, boundaries=[120])
    segs = env.get_segments()
    assert segs is not None
    n_seg = len(segs)
    starts = []
    for _ in range(n_seg):
        env.reset()
        starts.append(env._t)
    assert len(set(starts)) == n_seg
