"""freeze_vec_normalize_for_inference locks running stats for OOS rollouts.
Torch/SB3-gated: skips cleanly when stable_baselines3 is not installed."""

from __future__ import annotations

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from rlbot.vecnorm_utils import freeze_vec_normalize_for_inference  # noqa: E402


class _TinyEnv(gym.Env):
    observation_space = spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def __init__(self) -> None:
        super().__init__()
        self._rng = np.random.default_rng(0)
        self._i = 0

    def _obs(self) -> np.ndarray:
        return self._rng.normal(1.0, 2.0, size=3).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = 0
        return self._obs(), {}

    def step(self, action):
        self._i += 1
        return self._obs(), float(self._rng.normal()), False, self._i >= 200, {}


def _warmed_vecnorm() -> VecNormalize:
    venv = DummyVecEnv([_TinyEnv])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    vn.reset()
    for _ in range(20):
        vn.step(np.zeros((1, 2), dtype=np.float32))
    return vn


def test_freeze_sets_flags_and_keeps_obs_norm() -> None:
    vn = _warmed_vecnorm()
    out = freeze_vec_normalize_for_inference(vn)
    assert out is vn
    assert vn.training is False
    assert vn.norm_reward is False
    assert vn.norm_obs is True


def test_frozen_stats_do_not_update_on_step() -> None:
    vn = _warmed_vecnorm()
    freeze_vec_normalize_for_inference(vn)
    count = float(vn.obs_rms.count)
    mean = vn.obs_rms.mean.copy()
    var = vn.obs_rms.var.copy()
    for _ in range(15):
        vn.step(np.zeros((1, 2), dtype=np.float32))
    assert float(vn.obs_rms.count) == count  # no OOS data leaked into running stats
    np.testing.assert_array_equal(vn.obs_rms.mean, mean)
    np.testing.assert_array_equal(vn.obs_rms.var, var)


def test_frozen_normalize_obs_uses_training_stats() -> None:
    vn = _warmed_vecnorm()
    freeze_vec_normalize_for_inference(vn)
    raw = np.array([[0.5, -1.0, 2.0]], dtype=np.float32)
    normed = vn.normalize_obs(raw)
    expected = np.clip(
        (raw - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon), -10.0, 10.0
    )
    np.testing.assert_allclose(normed, expected, rtol=1e-6)
    # obs normalization is active (not identity)
    assert not np.allclose(normed, raw)
