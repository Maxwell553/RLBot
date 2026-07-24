"""Checkpoint selection helpers (trailing aggregate, confirmation, stress blend).

Kept torch-free for unit tests. Wired into ``EvalNavBestModelCallback`` via config knobs
with parser defaults that preserve legacy single-eval immediate-replace behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

TrailingAgg = Literal["median", "mean"]


def trailing_aggregate(
    scores: Sequence[float],
    *,
    window: int,
    agg: TrailingAgg = "median",
) -> float:
    """Aggregate the last ``window`` scores (or all if fewer). ``window<=1`` → last score."""
    if not scores:
        return float("-inf")
    w = max(1, int(window))
    tail = [float(s) for s in scores[-w:]]
    if agg == "mean":
        return float(np.mean(tail))
    if agg == "median":
        return float(np.median(tail))
    raise ValueError(f"unknown trailing agg {agg!r}")


def post_gate_scores(
    timesteps: Sequence[int],
    scores: Sequence[float],
    *,
    gate_step: int,
) -> list[float]:
    """Scores at/after ``gate_step`` (``gate_step<=0`` → all scores).

    Trailing selection must use this filtered series so pre-gate / fee-ramp
    friction-regime evals cannot enter the median/mean window.
    """
    n = min(len(timesteps), len(scores))
    if n <= 0:
        return []
    gate = int(gate_step)
    if gate <= 0:
        return [float(scores[i]) for i in range(n)]
    return [float(scores[i]) for i in range(n) if int(timesteps[i]) >= gate]


@dataclass(frozen=True)
class ConfirmationState:
    """Pending confirmation of a candidate best score."""

    candidate_score: float
    confirms_needed: int
    confirms_seen: int = 0

    @property
    def confirmed(self) -> bool:
        return self.confirms_needed <= 0 or self.confirms_seen >= self.confirms_needed


def update_confirmation(
    state: ConfirmationState | None,
    *,
    score: float,
    best_score: float,
    confirms_needed: int,
) -> tuple[ConfirmationState | None, bool]:
    """Advance confirmation state.

    Returns ``(new_state, should_replace_best)``.
    ``confirms_needed<=0`` → immediate replace when ``score > best_score``.
    """
    need = max(0, int(confirms_needed))
    if score <= best_score:
        return None, False
    if need <= 0:
        return None, True
    if state is None or score > state.candidate_score + 1e-12:
        # New candidate — start (or restart) confirmation streak.
        new = ConfirmationState(candidate_score=score, confirms_needed=need, confirms_seen=1)
        return new, new.confirmed
    # Same candidate sustained.
    new = ConfirmationState(
        candidate_score=state.candidate_score,
        confirms_needed=state.confirms_needed,
        confirms_seen=state.confirms_seen + 1,
    )
    return new, new.confirmed


def blend_canonical_stress(
    canonical: float,
    stress: float | None,
    *,
    stress_weight: float = 0.3,
) -> float:
    """Blend canonical eval score with an optional stress-suite score."""
    w = float(np.clip(stress_weight, 0.0, 1.0))
    if stress is None or w <= 0.0:
        return float(canonical)
    return float((1.0 - w) * canonical + w * float(stress))


@dataclass(frozen=True)
class SelectionDiagnostics:
    """Components of a best-checkpoint decision for logging / audit."""

    raw_score: float
    trailing_score: float
    selection_score: float
    stress_score: float | None
    trailing_window: int
    trailing_agg: str
    confirms_needed: int
    confirms_seen: int
    gate_open: bool
    replaced_best: bool

    def to_dict(self) -> dict:
        return {
            "raw_score": self.raw_score,
            "trailing_score": self.trailing_score,
            "selection_score": self.selection_score,
            "stress_score": self.stress_score,
            "trailing_window": self.trailing_window,
            "trailing_agg": self.trailing_agg,
            "confirms_needed": self.confirms_needed,
            "confirms_seen": self.confirms_seen,
            "gate_open": self.gate_open,
            "replaced_best": self.replaced_best,
        }


# Fixed stress multipliers (deterministic; independent of domain-randomization draws).
# Stress eval must price the passive benchmark at the same fee_scale as the agent
# (see EvalNavBestModelCallback._run_stress_eval) for a like-for-like excess score.
STRESS_FEE_SCALE = 1.5
STRESS_OBS_LAG = 2
