"""Reward-decomposition accumulator (P1-1). Torch-free."""

from __future__ import annotations

from rlbot.reward_logging import REWARD_TERMS, RewardDecompAccumulator


def _info(**terms: float) -> dict:
    return {f"rew_decomp/{k}": v for k, v in terms.items()}


def test_empty_summary_is_safe() -> None:
    acc = RewardDecompAccumulator()
    s = acc.summary()
    assert s["count"] == 0
    assert set(s["mean"]) == set(REWARD_TERMS)


def test_accumulates_means_and_shares() -> None:
    acc = RewardDecompAccumulator()
    acc.update([_info(return_=0.0)])  # ignored key (not a real term) → contributes nothing
    acc.update(
        [
            {"rew_decomp/return": -25.0, "rew_decomp/participation": 1.0,
             "rew_decomp/inactivity": -2.5, "rew_decomp/churn": -1.2,
             "rew_decomp/sortino": 5.0, "rew_decomp/drawdown": -5.0},
        ]
    )
    s = acc.summary()
    assert s["count"] == 1
    assert s["mean"]["inactivity"] == -2.5
    assert s["mean"]["participation"] == 1.0
    # return (with downside amp) should dominate shaping terms at realistic magnitudes
    assert s["abs_share"]["return"] > s["abs_share"]["inactivity"]
    assert abs(sum(s["abs_share"].values()) - 1.0) < 1e-9


def test_ignores_non_finite_and_missing() -> None:
    acc = RewardDecompAccumulator()
    acc.update([{"rew_decomp/return": float("nan"), "rew_decomp/churn": -1.0}])
    s = acc.summary()
    assert s["count"] == 1
    assert s["mean"]["churn"] == -1.0
    assert s["mean"]["return"] == 0.0  # nan skipped


def test_reset_clears() -> None:
    acc = RewardDecompAccumulator()
    acc.update([_info(churn=-1.0)])
    acc.reset()
    assert acc.summary()["count"] == 0
