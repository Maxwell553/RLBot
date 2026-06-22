from rlbot.training_progress import resolve_learn_timesteps


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
