from rlbot.training_progress import (
    absolute_progress_done,
    absolute_progress_remaining,
    churn_scale_at_step,
    lr_schedule_with_floor_for_budget,
    resolve_learn_timesteps,
)


def test_resolve_learn_timesteps_fresh_run() -> None:
    steps, reset = resolve_learn_timesteps(budget=50_000_000, start=0, resume=False)
    assert steps == 50_000_000
    assert reset is True


def test_resolve_learn_timesteps_crash_resume() -> None:
    steps, reset = resolve_learn_timesteps(budget=50_000_000, start=38_000_000, resume=True)
    assert steps == 12_000_000
    assert reset is False


def test_resolve_learn_timesteps_already_at_budget() -> None:
    steps, reset = resolve_learn_timesteps(budget=50_000_000, start=50_000_000, resume=True)
    assert steps == 0
    assert reset is False


def test_absolute_progress_done_and_remaining() -> None:
    assert absolute_progress_done(0, 50_000_000) == 0.0
    assert absolute_progress_done(25_000_000, 50_000_000) == 0.5
    assert absolute_progress_done(60_000_000, 50_000_000) == 1.0
    assert absolute_progress_remaining(38_000_000, 50_000_000) == 0.24


def test_churn_scale_zero_while_frictionless() -> None:
    fee_free, fee_ramp = 6_500_000, 29_250_000
    assert churn_scale_at_step(
        fee_free,
        fee_free_until=fee_free,
        fee_ramp_end=fee_ramp,
        churn_ramp_floor=0.10,
    ) == 0.0
    assert churn_scale_at_step(
        fee_free - 1,
        fee_free_until=fee_free,
        fee_ramp_end=fee_ramp,
        churn_ramp_floor=0.10,
    ) == 0.0


def test_churn_scale_ramps_with_fee_window() -> None:
    fee_free, fee_ramp = 100, 200
    mid = churn_scale_at_step(
        150,
        fee_free_until=fee_free,
        fee_ramp_end=fee_ramp,
        churn_ramp_floor=0.10,
    )
    assert mid == 0.55
    assert churn_scale_at_step(
        fee_ramp,
        fee_free_until=fee_free,
        fee_ramp_end=fee_ramp,
        churn_ramp_floor=0.10,
    ) == 1.0


def test_lr_schedule_uses_absolute_budget_not_session_progress() -> None:
    schedule = lr_schedule_with_floor_for_budget(3e-4, 1e-6, 50_000_000)
    schedule.sync_num_timesteps(38_000_000)
    lr_at_resume = schedule(1.0)
    schedule.sync_num_timesteps(0)
    lr_at_start = schedule(1.0)
    assert lr_at_resume < lr_at_start
