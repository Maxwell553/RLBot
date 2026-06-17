"""In-training eval cadence helpers (torch-free)."""

from __future__ import annotations


def eval_freq_vector_steps(global_freq_steps: int, n_envs: int) -> int:
    """Convert global-step eval interval to SB3 vector-step ``eval_freq``."""
    n = max(int(n_envs), 1)
    return max(int(global_freq_steps) // n, 1)


def active_eval_global_freq(
    num_timesteps: int,
    *,
    post_gate_global_freq: int,
    pre_gate_global_freq: int,
    best_model_min_step: int,
) -> int:
    if best_model_min_step > 0 and num_timesteps < best_model_min_step:
        return int(pre_gate_global_freq)
    return int(post_gate_global_freq)


def should_run_scheduled_eval(
    *,
    n_calls: int,
    last_eval_n_calls: int,
    num_timesteps: int,
    post_gate_global_freq: int,
    pre_gate_global_freq: int,
    best_model_min_step: int,
    n_envs: int,
    post_gate_eval_forced: bool,
) -> bool:
    """Whether to trigger an eval rollout on this vector step."""
    if n_calls <= 0:
        return False
    if (
        best_model_min_step > 0
        and num_timesteps >= best_model_min_step
        and not post_gate_eval_forced
    ):
        return True
    global_freq = active_eval_global_freq(
        num_timesteps,
        post_gate_global_freq=post_gate_global_freq,
        pre_gate_global_freq=pre_gate_global_freq,
        best_model_min_step=best_model_min_step,
    )
    if global_freq <= 0:
        return False
    freq_vec = eval_freq_vector_steps(global_freq, n_envs)
    if last_eval_n_calls <= 0:
        return n_calls >= freq_vec
    return (n_calls - last_eval_n_calls) >= freq_vec
