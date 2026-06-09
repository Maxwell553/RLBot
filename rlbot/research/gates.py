"""Evaluation tiers + OOS firewall for the auto-research loop.

The holdout windows are read **once** per pre-registered candidate. Tiers T1–T3 are
freely runnable by an agent; T4 (final OOS) requires explicit promotion and is refused
for a variant already scored at T4 (multiple-testing guard).
"""

from __future__ import annotations

from typing import Iterable, Mapping

# tier -> (label, touches_oos, needs_promotion)
TIERS: dict[int, tuple[str, bool, bool]] = {
    0: ("static tests + leakage checks", False, False),
    1: ("smoke train (tiny budget, no OOS)", False, False),
    2: ("short dev train, in-training eval only", False, False),
    3: ("multi-seed / multi-window, no final OOS", False, False),
    4: ("pre-registered full train, OOS read once", True, True),
    5: ("paper / shadow trading", True, True),
}


def tier_label(tier: int) -> str:
    return TIERS.get(int(tier), ("unknown", False, True))[0]


def tier_touches_oos(tier: int) -> bool:
    return TIERS.get(int(tier), ("", True, True))[1]


def tier_needs_promotion(tier: int) -> bool:
    return TIERS.get(int(tier), ("", True, True))[2]


def assert_tier_allowed(tier: int, *, promoted: bool) -> None:
    """Raise unless a tier that touches OOS / needs promotion has been promoted."""
    if int(tier) not in TIERS:
        raise ValueError(f"unknown evaluation tier {tier!r} (valid: {sorted(TIERS)})")
    if tier_needs_promotion(tier) and not promoted:
        raise PermissionError(
            f"tier {tier} ({tier_label(tier)}) touches the OOS holdout and requires "
            f"explicit promotion (--promote); refusing to run automatically."
        )


# Registry statuses that never gate: "failed" marks a variant whose train or backtest
# subprocess crashed — informational only. The OOS read itself is marked by the
# "oos_read_attempt" record written immediately BEFORE any backtest, so a pre-read
# train crash (no attempt record) can never brick a relaunch.
_UNSCORED_OOS_STATUSES = {"oos_read_attempt", "failed"}


def assert_no_repeat_oos(
    records: Iterable[Mapping],
    variant_id: str,
    *,
    allow_failed_rescore: bool = False,
) -> None:
    """Multiple-testing guard: refuse a second OOS (tier>=4) read for the same variant.

    Scored results block permanently. ``oos_read_attempt`` records (a read that crashed
    before scoring) block by default; ``allow_failed_rescore=True`` permits a retry only
    when no scored tier>=4 result exists. ``failed`` records alone never block — when
    the holdout was actually read, the preceding attempt record is what blocks.
    """
    relevant = [
        r for r in records
        if r.get("variant_id") == variant_id and int(r.get("evaluation_tier", 0)) >= 4
    ]
    scored = [r for r in relevant if str(r.get("status", "ok")) not in _UNSCORED_OOS_STATUSES]
    if scored:
        r = scored[0]
        raise PermissionError(
            f"variant {variant_id!r} already has a tier-{r.get('evaluation_tier')} "
            f"OOS result (run {r.get('run_id')!r}); refusing to re-read the holdout "
            f"(multiple-testing). Register a new variant id to test again."
        )
    attempts = [r for r in relevant if str(r.get("status", "ok")) == "oos_read_attempt"]
    if attempts and not allow_failed_rescore:
        r = attempts[0]
        raise PermissionError(
            f"variant {variant_id!r} has an unscored tier-{r.get('evaluation_tier')} OOS "
            f"read on record (status {r.get('status')!r}, run {r.get('run_id')!r}); the "
            "holdout may already have been read. Pass --allow-failed-rescore to retry."
        )


def assert_oos_budget(n_variants: int, budget: int) -> None:
    """Refuse a tier>=4 launch whose variant count exceeds the explicit OOS-read budget.

    Selecting the best of N grid cells *on the holdout* is the multiple-comparisons
    failure the tier system exists to prevent; large-N reads must be deliberate.
    """
    if n_variants > budget:
        raise PermissionError(
            f"this launch would read the OOS holdout for {n_variants} variant(s), "
            f"exceeding the OOS budget of {budget}. Selecting among many variants on "
            "the holdout is multiple testing; promote a single pre-registered variant "
            "instead, or raise --oos-budget explicitly if this is deliberate."
        )
