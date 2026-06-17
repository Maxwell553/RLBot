"""Tests for two-speed in-training eval cadence."""

from __future__ import annotations

from rlbot.eval_schedule import (
    active_eval_global_freq,
    eval_freq_vector_steps,
    should_run_scheduled_eval,
)


def test_eval_freq_vector_steps() -> None:
    assert eval_freq_vector_steps(500_000, 16) == 31_250
    assert eval_freq_vector_steps(3_000_000, 16) == 187_500


def test_active_eval_global_freq_switches_at_gate() -> None:
    assert active_eval_global_freq(
        10_000_000,
        post_gate_global_freq=500_000,
        pre_gate_global_freq=3_000_000,
        best_model_min_step=29_250_000,
    ) == 3_000_000
    assert active_eval_global_freq(
        30_000_000,
        post_gate_global_freq=500_000,
        pre_gate_global_freq=3_000_000,
        best_model_min_step=29_250_000,
    ) == 500_000


def test_should_run_scheduled_eval_pre_gate_interval() -> None:
    gate = 29_250_000
    n_envs = 16
    freq_vec = eval_freq_vector_steps(3_000_000, n_envs)
    assert not should_run_scheduled_eval(
        n_calls=freq_vec - 1,
        last_eval_n_calls=0,
        num_timesteps=3_000_000 - n_envs,
        post_gate_global_freq=500_000,
        pre_gate_global_freq=3_000_000,
        best_model_min_step=gate,
        n_envs=n_envs,
        post_gate_eval_forced=False,
    )
    assert should_run_scheduled_eval(
        n_calls=freq_vec,
        last_eval_n_calls=0,
        num_timesteps=3_000_000,
        post_gate_global_freq=500_000,
        pre_gate_global_freq=3_000_000,
        best_model_min_step=gate,
        n_envs=n_envs,
        post_gate_eval_forced=False,
    )


def test_should_force_eval_when_crossing_gate() -> None:
    assert should_run_scheduled_eval(
        n_calls=100,
        last_eval_n_calls=50,
        num_timesteps=29_250_000,
        post_gate_global_freq=500_000,
        pre_gate_global_freq=3_000_000,
        best_model_min_step=29_250_000,
        n_envs=16,
        post_gate_eval_forced=False,
    )


def test_post_gate_uses_500k_interval() -> None:
    n_envs = 16
    freq_vec = eval_freq_vector_steps(500_000, n_envs)
    last = 1_000_000 // n_envs
    assert should_run_scheduled_eval(
        n_calls=last + freq_vec,
        last_eval_n_calls=last,
        num_timesteps=30_000_000,
        post_gate_global_freq=500_000,
        pre_gate_global_freq=3_000_000,
        best_model_min_step=29_250_000,
        n_envs=n_envs,
        post_gate_eval_forced=True,
    )
