"""Training progress helpers (resume budget + global progress bar)."""

from __future__ import annotations

import math

try:
    from tqdm.rich import tqdm
except ImportError:  # pragma: no cover - optional SB3 extra
    tqdm = None  # type: ignore[misc, assignment]

from stable_baselines3.common.callbacks import BaseCallback


def absolute_progress_done(num_timesteps: int, budget: int) -> float:
    """Fraction of the absolute training budget completed (0 → 1, capped)."""
    budget_i = max(1, int(budget))
    return min(1.0, max(0.0, float(num_timesteps) / float(budget_i)))


def absolute_progress_remaining(num_timesteps: int, budget: int) -> float:
    """Remaining fraction of the absolute training budget (1 → 0)."""
    return 1.0 - absolute_progress_done(num_timesteps, budget)


def churn_scale_at_step(
    t: int,
    *,
    fee_free_until: int,
    fee_ramp_end: int,
    churn_ramp_floor: float,
) -> float:
    """Churn/turnover penalty scale aligned with the fee ramp (0 while frictionless).

    Stays at 0 through ``fee_free_until`` (same as fees). During the fee-ramp window,
    scales ``churn_ramp_floor`` → 1.0 in lockstep with the linear fee override.
    """
    if t < fee_free_until:
        return 0.0
    if t >= fee_ramp_end:
        return 1.0
    span = max(fee_ramp_end - fee_free_until, 1)
    fee_progress = float(t - fee_free_until) / float(span)
    if fee_progress <= 0.0:
        return 0.0
    floor = float(churn_ramp_floor)
    return floor + (1.0 - floor) * fee_progress


def lr_at_step(
    t: int,
    *,
    initial_lr: float,
    floor_lr: float,
    budget: int,
    hold_until_step: int = 0,
) -> float:
    """LR value at absolute step ``t``.

    ``hold_until_step == 0`` (legacy "cosine"): global cosine ``initial → floor``
    over the whole budget.
    ``hold_until_step > 0`` ("phase_aware"): hold ``initial_lr`` through the
    curriculum/DR-widening phase, then cosine-decay to ``floor_lr`` over the
    remaining ``budget - hold_until_step`` steps — the policy keeps a meaningful
    LR while fee/lag conditions are still changing and anneals only once the
    training distribution is stationary.
    """
    budget_i = max(1, int(budget))
    hold = max(0, min(int(hold_until_step), budget_i))
    t_i = max(0, int(t))
    if hold <= 0:
        progress_done = min(1.0, t_i / budget_i)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress_done))
        return floor_lr + (initial_lr - floor_lr) * cosine
    if t_i < hold:
        return float(initial_lr)
    span = max(budget_i - hold, 1)
    progress_done = min(1.0, (t_i - hold) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress_done))
    return floor_lr + (initial_lr - floor_lr) * cosine


def lr_schedule_with_floor_for_budget(
    initial_lr: float,
    floor_lr: float,
    budget: int,
    *,
    hold_until_step: int = 0,
):
    """Cosine LR keyed to absolute ``num_timesteps / budget`` (resume-safe).

    Stable-Baselines3 passes ``progress_remaining`` relative to the current
    ``learn()`` call; this schedule ignores that argument and reads
    ``sync_num_timesteps`` instead so crash-resume stays on the global budget curve.
    ``hold_until_step > 0`` enables the phase-aware hold (see :func:`lr_at_step`).
    """
    state = {"num_timesteps": 0}

    def schedule(_progress_remaining: float) -> float:
        return lr_at_step(
            state["num_timesteps"],
            initial_lr=initial_lr,
            floor_lr=floor_lr,
            budget=budget,
            hold_until_step=hold_until_step,
        )

    def sync_num_timesteps(num_timesteps: int) -> None:
        state["num_timesteps"] = int(num_timesteps)

    schedule.sync_num_timesteps = sync_num_timesteps  # type: ignore[attr-defined]
    return schedule


def resolve_learn_timesteps(*, budget: int, start: int, resume: bool) -> tuple[int, bool]:
    """Return (learn_timesteps, reset_num_timesteps) for ``model.learn()``.

    Stable-Baselines3 adds ``start`` to the ``total_timesteps`` argument when
    ``reset_num_timesteps=False``, so on crash-resume we must pass only the
    *remaining* steps to hit the absolute ``budget``, not the full budget again.
    """
    budget_i = int(budget)
    start_i = int(start)
    if resume:
        return max(0, budget_i - start_i), False
    return budget_i, True


class BudgetProgressBarCallback(BaseCallback):
    """Progress bar against the absolute training budget (not session-only steps)."""

    def __init__(self, budget_timesteps: int) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "BudgetProgressBarCallback requires tqdm+rich "
                "(pip install stable-baselines3[extra])"
            )
        self.budget_timesteps = int(budget_timesteps)
        self.pbar: tqdm | None = None

    def _on_training_start(self) -> None:
        start = int(self.model.num_timesteps)
        remaining = max(0, self.budget_timesteps - start)
        self.pbar = tqdm(
            total=remaining,
            desc=self._desc(start),
            unit="step",
            unit_scale=True,
        )

    def _on_step(self) -> bool:
        assert self.pbar is not None
        self.pbar.update(self.training_env.num_envs)
        self.pbar.set_description(self._desc(int(self.model.num_timesteps)))
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.refresh()
            self.pbar.close()
            self.pbar = None

    def _desc(self, current: int) -> str:
        pct = 100.0 * current / self.budget_timesteps if self.budget_timesteps else 100.0
        return f"{current:,}/{self.budget_timesteps:,} ({pct:.1f}%)"
