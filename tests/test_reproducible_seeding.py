"""Env-level determinism: reseed_on_reset=False + same env_seed reproduces episode
starts, domain-randomization draws, and observations; reseed_on_reset=True diverges.
apply_deterministic_seeds is torch-gated."""

from __future__ import annotations

import numpy as np
import pytest

from rlbot.rl_config import get_config
from rlbot.trading_env import MultiAssetPortfolioEnv

_N = get_config().universe.n_assets
_N_ACT = _N + 1
_T = 400


def _panels():
    rng = np.random.default_rng(0)  # panel itself is fixed across env instances
    rets = rng.normal(0.0005, 0.01, size=(_T, _N))
    price = 100.0 * np.exp(np.cumsum(rets, axis=0))
    ohlcv = np.zeros((_T, _N, 5), dtype=np.float64)
    for c in range(4):
        ohlcv[:, :, c] = price
    ohlcv[:, :, 4] = 1e6
    macro = np.full((_T, 4), 15.0)
    return ohlcv, macro


def _make_env(*, env_seed: int | None, reseed_on_reset: bool) -> MultiAssetPortfolioEnv:
    ohlcv, macro = _panels()
    return MultiAssetPortfolioEnv(
        ohlcv,
        np.full((_T, _N), 50.0),
        np.zeros((_T, _N)),
        macro=macro,
        fracdiff=np.zeros((_T, _N)),
        fracdiff_macro=np.zeros((_T, 4)),
        trend=np.zeros((_T, _N)),
        random_start=True,
        domain_randomize=True,  # exercise the obs_lag / fee_scale draws
        max_episode_steps=50,
        lookback=5,
        env_seed=env_seed,
        reseed_on_reset=reseed_on_reset,
        action_smoothing_alpha=0.0,
    )


def _episode_trace(env: MultiAssetPortfolioEnv, n_resets: int = 8):
    """Per-reset (start index, obs_lag, fee_scale, first obs) sequence."""
    trace = []
    for _ in range(n_resets):
        obs, _ = env.reset()
        trace.append((env._t, env.obs_lag, env.fee_scale, obs.copy()))
    return trace


def test_same_env_seed_reproduces_episodes_and_dr_draws() -> None:
    a = _make_env(env_seed=123, reseed_on_reset=False)
    b = _make_env(env_seed=123, reseed_on_reset=False)
    ta, tb = _episode_trace(a), _episode_trace(b)
    for (t1, lag1, fee1, obs1), (t2, lag2, fee2, obs2) in zip(ta, tb):
        assert t1 == t2  # identical episode start indices
        assert lag1 == lag2  # identical domain-randomized obs_lag
        assert fee1 == fee2  # identical domain-randomized fee_scale
        np.testing.assert_array_equal(obs1, obs2)
    # the trace is not degenerate (random starts actually vary across resets)
    assert len({t for t, _, _, _ in ta}) > 1


def test_same_env_seed_reproduces_rollout() -> None:
    a = _make_env(env_seed=7, reseed_on_reset=False)
    b = _make_env(env_seed=7, reseed_on_reset=False)
    a.reset()
    b.reset()
    rng = np.random.default_rng(99)
    for _ in range(10):
        action = rng.uniform(-3.0, 3.0, size=_N_ACT)
        oa, ra, term_a, trunc_a, ia = a.step(action)
        ob, rb, term_b, trunc_b, ib = b.step(action)
        assert ra == rb
        assert (term_a, trunc_a) == (term_b, trunc_b)
        assert ia["nav"] == ib["nav"]
        np.testing.assert_array_equal(oa, ob)
        if term_a or trunc_a:
            break


def test_different_env_seeds_diverge() -> None:
    a = _make_env(env_seed=1, reseed_on_reset=False)
    b = _make_env(env_seed=2, reseed_on_reset=False)
    ta, tb = _episode_trace(a, 12), _episode_trace(b, 12)
    assert [x[0] for x in ta] != [x[0] for x in tb] or [x[2] for x in ta] != [x[2] for x in tb]


def test_reseed_on_reset_diverges_despite_same_seed() -> None:
    """reseed_on_reset=True draws fresh OS entropy per episode: two same-seed envs
    (almost surely) produce different start/fee sequences over many resets."""
    a = _make_env(env_seed=42, reseed_on_reset=True)
    b = _make_env(env_seed=42, reseed_on_reset=True)
    n = 25
    ta, tb = _episode_trace(a, n), _episode_trace(b, n)
    fee_a = [x[2] for x in ta]
    fee_b = [x[2] for x in tb]
    starts_a = [x[0] for x in ta]
    starts_b = [x[0] for x in tb]
    # 25 continuous Beta draws colliding across independent OS-entropy streams is ~impossible
    assert fee_a != fee_b or starts_a != starts_b


def test_apply_deterministic_seeds_makes_numpy_reproducible() -> None:
    pytest.importorskip("torch")
    from rlbot.rl_config import apply_deterministic_seeds

    apply_deterministic_seeds(1234)
    first = np.random.rand(8)
    py_first = __import__("random").random()
    apply_deterministic_seeds(1234)
    second = np.random.rand(8)
    py_second = __import__("random").random()
    np.testing.assert_array_equal(first, second)
    assert py_first == py_second
