"""Explicit validity / comparability labels for walk-forward runs.

Documentation alone historically buried contamination (``623``–``625``) and
resumed long-budget cells (``W3``–``W5_619``). The audit report and dashboard
read these rules so non-comparable runs surface automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

# Contaminated by the ``vol_penalty_scale: 300`` × ``reward_scale`` bug.
# ``622`` carried the knob but is analyzed separately (see docs/RESEARCH.md):
# it is *not* bundled with 623–625 as identically invalid for every claim.
INVALID_VOL_PENALTY_COHORTS = frozenset({"623", "624", "625"})
# Same scale bug present, but treat as its own flagged cohort (not "bundled invalid").
VOL_PENALTY_BUG_PRESENT_COHORTS = frozenset({"622"}) | INVALID_VOL_PENALTY_COHORTS

# Resumed past the nominal 50M budget (~82–84M cumulative on W3–W5).
RESUMED_LONG_BUDGET_RUNS = frozenset({"W3_619", "W4_619", "W5_619"})

_RUN_ID_RE = re.compile(r"^W(\d+)_(\d+)(?:_([a-z0-9]+))?$", re.IGNORECASE)


@dataclass(frozen=True)
class RunLabel:
    """Comparability annotation for one run id."""

    run_id: str
    window: int | None
    cohort: str | None
    labels: tuple[str, ...]
    notes: tuple[str, ...]
    comparable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "window": self.window,
            "cohort": self.cohort,
            "labels": list(self.labels),
            "notes": list(self.notes),
            "comparable": self.comparable,
        }


def parse_run_id(run_id: str) -> tuple[int | None, str | None]:
    """Return ``(window, cohort_id)`` for ``W{n}_{cohort}`` ids, else ``(None, None)``."""
    m = _RUN_ID_RE.match(str(run_id).strip())
    if not m:
        return None, None
    return int(m.group(1)), str(m.group(2))


def label_run(
    run_id: str,
    *,
    manifest: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> RunLabel:
    """Attach explicit labels from id conventions + optional manifest/config evidence."""
    window, cohort = parse_run_id(run_id)
    labels: list[str] = []
    notes: list[str] = []
    comparable = True

    args = (manifest or {}).get("args") or {}
    reward = (config or {}).get("reward") or {}
    vol_pen = reward.get("vol_penalty_scale")

    if cohort in INVALID_VOL_PENALTY_COHORTS:
        labels.append("invalid_vol_penalty_scale")
        notes.append(
            "Cohorts 623–625: vol_penalty_scale≈300 × reward_scale dominated the "
            "reward. Exclude from method comparisons."
        )
        comparable = False
    elif cohort == "622":
        # Separate from 623–625: same bug present, but do not bundle as identical.
        labels.append("vol_penalty_bug_present")
        notes.append(
            "Cohort 622 carried the pre-fix vol_penalty_scale; treat separately from "
            "623–625 and from clean grids — do not bundle as identically invalid."
        )
        comparable = False
    elif isinstance(vol_pen, (int, float)) and float(vol_pen) >= 10.0:
        labels.append("invalid_vol_penalty_scale")
        notes.append(
            f"vol_penalty_scale={vol_pen} dominates reward (≥10 × reward_scale). "
            "Exclude from method comparisons."
        )
        comparable = False

    if run_id in RESUMED_LONG_BUDGET_RUNS:
        labels.append("resumed_long_budget")
        notes.append(
            "W3–W5_619 were crash-resumed past the nominal 50M budget "
            "(~82–84M cumulative steps). Not comparable to single-pass 50M cells."
        )
        comparable = False

    resume = str(args.get("resume") or "").strip()
    if resume and "resumed" not in labels and "resumed_long_budget" not in labels:
        labels.append("resumed")
        notes.append(f"Training resumed from checkpoint: {resume}")

    if str(args.get("finetune") or "").strip():
        labels.append("finetuned")
        notes.append("Training used --finetune (curriculum/entropy callbacks skipped).")

    status = (manifest or {}).get("training_status")
    if status == "interrupted":
        labels.append("interrupted")
        notes.append("training_status=interrupted")

    if (manifest or {}).get("git_dirty") is True:
        labels.append("dirty_source")
        notes.append("git_dirty=true at training provenance stamp")

    if not labels:
        labels.append("clean")

    return RunLabel(
        run_id=run_id,
        window=window,
        cohort=cohort,
        labels=tuple(labels),
        notes=tuple(notes),
        comparable=comparable,
    )
