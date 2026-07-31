"""SB3-only progress bar callback (imports torch / stable-baselines3)."""

from __future__ import annotations

try:
    from tqdm.rich import tqdm
except ImportError:  # pragma: no cover - optional SB3 extra
    tqdm = None  # type: ignore[misc, assignment]

from stable_baselines3.common.callbacks import BaseCallback


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
