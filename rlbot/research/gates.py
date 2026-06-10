"""Evaluation tiers + OOS firewall for the auto-research loop.

The holdout windows are read **once** per pre-registered candidate. Tiers T1–T3 are
freely runnable by an agent; T4 (final OOS) requires explicit promotion and is refused
for a variant already scored at T4 (multiple-testing guard).
"""

from __future__ import annotations

import statistics
from typing import Iterable, Mapping

# tier -> (label, touches_oos, needs_promotion)
TIERS: dict[int, tuple[str, bool, bool]] = {
    0: ("static tests + leakage checks", False, False),
    1: ("smoke train (tiny budget, no OOS)", False, False),
    2: ("short dev train, in-training eval only", False, False),
    3: ("multi-seed / multi-window, no final OOS", False, False),
    4: ("pre-registered full train, OOS read once", True, True),
    # Tier 5 reads NO holdout (forward shadow data only) but starting it is still
    # a human promotion decision.
    5: ("paper / shadow trading (forward data; no holdout read)", False, True),
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


# ── Success-gate engine: pre-registered decision rules, evaluated at collect ──

# Gate keys a spec may declare. Anything else raises — a typo'd pre-registered
# gate that silently never evaluates would defeat the point of pre-registration.
SUPPORTED_SUCCESS_GATES = {
    "min_seeds",               # scored distinct seeds required, else inconclusive
    "eval_nav_mean_min",       # mean best_eval_nav across seeds >= x
    "eval_nav_median_min",     # median best_eval_nav across seeds >= x
    "eval_nav_spread_max_frac",  # (max-min)/|median| of best_eval_nav <= x (seed stability)
    "oos_sharpe_min",          # tier>=4 rows only
    "oos_max_drawdown_floor",  # tier>=4: max_drawdown >= x (x negative, e.g. -0.25)
    "deflated_sharpe_min",     # tier>=4: selection-aware significance bar
}

_OOS_GATES = {"oos_sharpe_min", "oos_max_drawdown_floor", "deflated_sharpe_min"}


def evaluate_success_gates(success_gates: Mapping, rows: Iterable[Mapping]) -> dict:
    """Evaluate one seed-group's scored records against its spec's success_gates.

    Returns ``{"verdict": "pass"|"fail"|"inconclusive", "checks": {...}}``.
    ``inconclusive`` = not enough evidence (too few seeds, or an OOS gate declared
    with no tier>=4 record yet); ``fail`` = a threshold was checked and missed.
    """
    gates = dict(success_gates or {})
    unknown = set(gates) - SUPPORTED_SUCCESS_GATES
    if unknown:
        raise ValueError(
            f"unknown success_gates key(s) {sorted(unknown)}; "
            f"supported: {sorted(SUPPORTED_SUCCESS_GATES)}"
        )
    rows = [r for r in rows if str(r.get("status", "ok")) == "ok"]
    checks: dict[str, dict] = {}
    verdict = "pass"
    from rlbot.research.report import dedupe_scored_by_run  # shared: never inline this

    def _check(name: str, ok: bool | None, observed, threshold) -> None:
        nonlocal verdict
        state = "inconclusive" if ok is None else ("pass" if ok else "fail")
        checks[name] = {"state": state, "observed": observed, "threshold": threshold}
        if state == "fail":
            verdict = "fail"
        elif state == "inconclusive" and verdict != "fail":
            verdict = "inconclusive"

    rows = dedupe_scored_by_run(rows)
    # Tier-1 rows are smoke/screen evidence (tiny budgets) — never promotion
    # evidence. A group with ONLY tier-1 rows has no decision evidence at all:
    # every threshold check below goes inconclusive on the empty set.
    rows = [r for r in rows if int(r.get("evaluation_tier", 0)) >= 2]

    navs = [float(r["best_eval_nav"]) for r in rows if r.get("best_eval_nav") is not None]
    seeds = {r.get("seed") for r in rows if r.get("seed") is not None}
    n_seeds = len(seeds) if seeds else len(rows)
    if "min_seeds" in gates:
        # Too few seeds is missing evidence, not a failed threshold.
        _check("min_seeds", True if n_seeds >= int(gates["min_seeds"]) else None,
               n_seeds, int(gates["min_seeds"]))
    if "eval_nav_mean_min" in gates:
        obs = sum(navs) / len(navs) if navs else None
        _check("eval_nav_mean_min", None if obs is None else obs >= float(gates["eval_nav_mean_min"]),
               obs, float(gates["eval_nav_mean_min"]))
    if "eval_nav_median_min" in gates:
        obs = float(statistics.median(navs)) if navs else None
        _check("eval_nav_median_min", None if obs is None else obs >= float(gates["eval_nav_median_min"]),
               obs, float(gates["eval_nav_median_min"]))
    if "eval_nav_spread_max_frac" in gates:
        if len(navs) >= 2:
            med = float(statistics.median(navs))
            obs = (max(navs) - min(navs)) / max(abs(med), 1e-9)
            _check("eval_nav_spread_max_frac", obs <= float(gates["eval_nav_spread_max_frac"]),
                   obs, float(gates["eval_nav_spread_max_frac"]))
        else:
            _check("eval_nav_spread_max_frac", None, None, float(gates["eval_nav_spread_max_frac"]))
    oos_rows = [r for r in rows if int(r.get("evaluation_tier", 0)) >= 4]
    for key, field in (
        ("oos_sharpe_min", "oos_sharpe"),
        ("oos_max_drawdown_floor", "oos_max_drawdown"),  # floor: dd (negative) >= x
        ("deflated_sharpe_min", "oos_deflated_sharpe"),
    ):
        if key not in gates:
            continue
        vals = [float(r[field]) for r in oos_rows if r.get(field) is not None]
        if not vals:
            _check(key, None, None, float(gates[key]))
            continue
        obs = float(statistics.median(vals))
        _check(key, obs >= float(gates[key]), obs, float(gates[key]))
    return {"verdict": verdict, "checks": checks}
