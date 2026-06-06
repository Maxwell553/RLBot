"""Aggregate the per-step reward decomposition the env emits in ``info['rew_decomp/*']``.

Kept torch-free so the aggregation logic is unit-testable without SB3/torch. The SB3
callback in ``scripts/train.py`` feeds it ``self.locals['infos']`` each step and logs the
summary to TensorBoard + a rolling JSON. Surfaces the review's reward-asymmetry finding
(inactivity dwarfs participation/churn) via each term's share of absolute reward.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

REWARD_TERMS = ("return", "sortino", "inactivity", "participation", "churn", "drawdown")


class RewardDecompAccumulator:
    """Running per-term sums (signed + absolute) over emitted ``rew_decomp/*`` values."""

    def __init__(self) -> None:
        self._sum = {k: 0.0 for k in REWARD_TERMS}
        self._abs_sum = {k: 0.0 for k in REWARD_TERMS}
        self._count = 0

    def update(self, infos: Iterable[Mapping]) -> None:
        for info in infos:
            if not isinstance(info, Mapping):
                continue
            seen = False
            for k in REWARD_TERMS:
                v = info.get(f"rew_decomp/{k}")
                if v is None:
                    continue
                fv = float(v)
                if not np.isfinite(fv):
                    continue
                self._sum[k] += fv
                self._abs_sum[k] += abs(fv)
                seen = True
            if seen:
                self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def summary(self) -> dict:
        """Per-term mean and share of total absolute reward (empty until any update)."""
        n = max(self._count, 1)
        means = {k: self._sum[k] / n for k in REWARD_TERMS}
        abs_means = {k: self._abs_sum[k] / n for k in REWARD_TERMS}
        total_abs = sum(abs_means.values()) or 1.0
        shares = {k: abs_means[k] / total_abs for k in REWARD_TERMS}
        return {
            "count": self._count,
            "mean": means,
            "abs_mean": abs_means,
            "abs_share": shares,
        }

    def reset(self) -> None:
        self._sum = {k: 0.0 for k in REWARD_TERMS}
        self._abs_sum = {k: 0.0 for k in REWARD_TERMS}
        self._count = 0
