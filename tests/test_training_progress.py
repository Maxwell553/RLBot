import pytest

from rlbot.training_progress import (
    absolute_progress_done,
    absolute_progress_remaining,
    churn_scale_at_step,
    lr_at_step,
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


def test_lr_phase_aware_holds_through_dr_widening_then_decays() -> None:
    initial, floor, budget, hold = 3e-4, 1e-6, 50_000_000, 35_000_000
    kw = dict(initial_lr=initial, floor_lr=floor, budget=budget, hold_until_step=hold)
    # Full LR held through the entire widening phase (gate, mid-widen, full-DR start).
    assert lr_at_step(0, **kw) == pytest.approx(initial)
    assert lr_at_step(22_500_000, **kw) == pytest.approx(initial)
    assert lr_at_step(hold - 1, **kw) == pytest.approx(initial)
    # Decays after the hold, hitting the floor at budget end.
    mid = lr_at_step(hold + 7_500_000, **kw)  # halfway through the decay span
    assert floor < mid < initial
    assert mid == pytest.approx(floor + (initial - floor) * 0.5, rel=1e-6)
    assert lr_at_step(budget, **kw) == pytest.approx(floor, abs=1e-9)


def test_lr_phase_aware_schedule_wrapper_and_legacy_equivalence() -> None:
    # hold_until_step=0 reproduces the legacy global cosine exactly.
    legacy = lr_schedule_with_floor_for_budget(3e-4, 1e-6, 50_000_000)
    zero_hold = lr_schedule_with_floor_for_budget(3e-4, 1e-6, 50_000_000, hold_until_step=0)
    for t in (0, 10_000_000, 29_250_000, 41_750_000, 50_000_000):
        legacy.sync_num_timesteps(t)
        zero_hold.sync_num_timesteps(t)
        assert legacy(1.0) == pytest.approx(zero_hold(1.0))
    # Phase-aware wrapper matches lr_at_step.
    pa = lr_schedule_with_floor_for_budget(
        3e-4, 1e-6, 50_000_000, hold_until_step=35_000_000
    )
    pa.sync_num_timesteps(30_000_000)
    assert pa(1.0) == pytest.approx(3e-4)
    pa.sync_num_timesteps(42_500_000)
    assert pa(1.0) == pytest.approx(
        lr_at_step(
            42_500_000,
            initial_lr=3e-4,
            floor_lr=1e-6,
            budget=50_000_000,
            hold_until_step=35_000_000,
        )
    )
