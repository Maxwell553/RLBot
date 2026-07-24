"""Aggregate the per-step reward decomposition the env emits in ``info['rew_decomp/*']``.

Kept torch-free so the aggregation logic is unit-testable without SB3/torch. The SB3
callback in ``scripts/train.py`` feeds it ``self.locals['infos']`` each step and logs the
summary to TensorBoard + a rolling JSONL history. Surfaces the review's reward-asymmetry
finding (inactivity dwarfs participation/churn) via each term's share of absolute reward.

Also tracks reward quantiles, drawdown increase vs level split, and vol-penalty activation.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

REWARD_TERMS = (
    "return",
    "benchmark",
    "sortino",
    "inactivity",
    "participation",
    "churn",
    "turnover",
    "drawdown",
    "drawdown_penalty",
    "drawdown_increase",
    "drawdown_level",
    "concentration",
    "exposure_risk",
    "volatility",
)

# Extra info keys (not part of abs_share mass) tracked for observability.
_EXTRA_KEYS = (
    "volatility_excess_raw",
    "volatility_active",
)


class RewardDecompAccumulator:
    """Running per-term sums (signed + absolute) over emitted ``rew_decomp/*`` values."""

    def __init__(self, *, quantile_sample_cap: int = 50_000) -> None:
        self._sum = {k: 0.0 for k in REWARD_TERMS}
        self._abs_sum = {k: 0.0 for k in REWARD_TERMS}
        self._count = 0
        self._extra_sum = {k: 0.0 for k in _EXTRA_KEYS}
        self._extra_count = {k: 0 for k in _EXTRA_KEYS}
        self._vol_active_count = 0
        self._reward_samples: list[float] = []
        self._quantile_sample_cap = max(1000, int(quantile_sample_cap))

    def update(self, infos: Iterable[Mapping]) -> None:
        for info in infos:
            if not isinstance(info, Mapping):
                continue
            seen = False
            step_total = 0.0
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
                # Quantile of reconstructed step reward: skip amp accounting and
                # the increase/level split children (already in drawdown_penalty).
                if k in ("drawdown", "drawdown_increase", "drawdown_level"):
                    continue
                step_total += fv
            if seen:
                self._count += 1
                if len(self._reward_samples) < self._quantile_sample_cap:
                    self._reward_samples.append(step_total)

            for k in _EXTRA_KEYS:
                v = info.get(f"rew_decomp/{k}")
                if v is None:
                    continue
                fv = float(v)
                if not np.isfinite(fv):
                    continue
                self._extra_sum[k] += fv
                self._extra_count[k] += 1
            active = info.get("rew_decomp/volatility_active")
            if active is not None and float(active) > 0.5:
                self._vol_active_count += 1

    @property
    def count(self) -> int:
        return self._count

    def summary(self) -> dict:
        """Per-term mean and share of total absolute reward (empty until any update)."""
        n = max(self._count, 1)
        means = {k: self._sum[k] / n for k in REWARD_TERMS}
        abs_means = {k: self._abs_sum[k] / n for k in REWARD_TERMS}
        # Abs-share: include amp accounting (`drawdown`) as historically, but exclude
        # the increase/level split children (already counted inside `drawdown_penalty`).
        share_keys = [
            k for k in REWARD_TERMS if k not in ("drawdown_increase", "drawdown_level")
        ]
        total_abs = sum(abs_means[k] for k in share_keys) or 1.0
        shares = {k: (abs_means[k] / total_abs if k in share_keys else 0.0) for k in REWARD_TERMS}
        # Report split children relative to the same denominator for readability.
        for k in ("drawdown_increase", "drawdown_level"):
            shares[k] = abs_means[k] / total_abs

        extras: dict[str, float] = {}
        for k in _EXTRA_KEYS:
            c = max(self._extra_count[k], 1)
            extras[k] = self._extra_sum[k] / c
        extras["volatility_activation_rate"] = (
            float(self._vol_active_count) / float(n) if self._count else 0.0
        )

        quantiles: dict[str, float] = {}
        if self._reward_samples:
            arr = np.asarray(self._reward_samples, dtype=np.float64)
            quantiles = {
                "min": float(np.min(arr)),
                "p01": float(np.percentile(arr, 1)),
                "p05": float(np.percentile(arr, 5)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(np.max(arr)),
            }

        return {
            "count": self._count,
            "mean": means,
            "abs_mean": abs_means,
            "abs_share": shares,
            "extras": extras,
            "reward_quantiles": quantiles,
        }

    def reset(self) -> None:
        self._sum = {k: 0.0 for k in REWARD_TERMS}
        self._abs_sum = {k: 0.0 for k in REWARD_TERMS}
        self._count = 0
        self._extra_sum = {k: 0.0 for k in _EXTRA_KEYS}
        self._extra_count = {k: 0 for k in _EXTRA_KEYS}
        self._vol_active_count = 0
        self._reward_samples = []
