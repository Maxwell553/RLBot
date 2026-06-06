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


def assert_no_repeat_oos(
    records: Iterable[Mapping],
    variant_id: str,
) -> None:
    """Multiple-testing guard: refuse a second OOS (tier>=4) read for the same variant."""
    for r in records:
        if r.get("variant_id") == variant_id and int(r.get("evaluation_tier", 0)) >= 4:
            raise PermissionError(
                f"variant {variant_id!r} already has a tier-{r.get('evaluation_tier')} "
                f"OOS result (run {r.get('run_id')!r}); refusing to re-read the holdout "
                f"(multiple-testing). Register a new variant id to test again."
            )
