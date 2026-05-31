"""
Read-only VecNormalize for inference / backtest / live rollout.
"""

from __future__ import annotations

from stable_baselines3.common.vec_env import VecNormalize


def freeze_vec_normalize_for_inference(vec_env: VecNormalize) -> VecNormalize:
    """
    Load-time safety for production inference: freeze training statistics
    so running mean/variance indicators are locked out-of-sample.

    SB3 skips ``_update_obs_rms`` / reward RMS updates when ``training`` is False.
    """
    vec_env.training = False
    vec_env.norm_reward = False
    vec_env.norm_obs = True
    return vec_env
