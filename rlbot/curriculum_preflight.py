"""Curriculum / schedule preflight: milestones, LR/entropy values, early-stop reachability.

Pure helpers (torch-free) so schedule edits can be unit-tested before a long train.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from rlbot.rl_config import (
    CurriculumConfig,
    EntropyScheduleConfig,
    RLConfig,
    resolve_best_model_min_step,
    trade_curriculum_milestones,
)
from rlbot.training_progress import churn_scale_at_step, lr_at_step


@dataclass(frozen=True)
class ScheduleMilestone:
    name: str
    step: int
    fee_override: float | None  # None = full fees (no override)
    churn_scale: float
    ent_coef: float
    learning_rate: float
    dr_phase: str  # pinned | widening | full
    notes: str = ""


@dataclass(frozen=True)
class CurriculumPreflight:
    budget: int
    fee_free_until: int
    fee_ramp_end: int
    dr_widen_end: int
    best_model_min_step: int
    entropy_decay_start: int
    early_floor_end: int
    dr_lock_end: int
    lr_floor: float
    lr_initial: float
    lr_schedule: str
    lr_hold_until_step: int
    explore_ent: float
    final_ent: float
    early_stop_patience: int
    stationary_full_dr_steps: int
    early_stop_reachable: bool
    milestones: tuple[ScheduleMilestone, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["milestones"] = [asdict(m) for m in self.milestones]
        return d


def dr_widen_end_for_budget(budget: int, cur: CurriculumConfig) -> int:
    """Mirror ``scripts.train.dr_widen_end_milestone`` without importing train.py."""
    lb = max(1, int(budget))
    _, fee_ramp_end = trade_curriculum_milestones(lb, cur=cur)
    if lb <= cur.budget_short:
        span = int(cur.dr_widen_span_fraction * lb)
    elif lb >= cur.budget_long:
        span = int(cur.dr_widen_span_long)
    else:
        short_span = int(cur.dr_widen_span_fraction * cur.budget_short)
        t = (lb - cur.budget_short) / (cur.budget_long - cur.budget_short)
        span = int(short_span + t * (cur.dr_widen_span_long - short_span))
    return min(lb, fee_ramp_end + max(span, 0))


def _entropy_at_step(
    step: int,
    *,
    budget: int,
    ent: EntropyScheduleConfig,
) -> float:
    decay_start = int(ent.decay_start_fraction * budget)
    early_floor_end = int(ent.early_floor_fraction * budget)
    dr_lock = int(ent.dr_lock_fraction * budget)
    if step < decay_start:
        e = float(ent.explore_ent)
        if step < dr_lock:
            e = max(e, float(ent.early_floor_high))
        elif step < early_floor_end:
            e = max(e, float(ent.early_floor))
        return e
    span = max(budget - decay_start, 1)
    t = min(1.0, max(0.0, float(step - decay_start) / float(span)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(ent.final_ent + (ent.explore_ent - ent.final_ent) * cosine)


def _fee_override_at(step: int, fee_free: int, fee_ramp: int) -> float | None:
    if step < fee_free:
        return 0.0
    if step >= fee_ramp:
        return None
    span = max(fee_ramp - fee_free, 1)
    return float(step - fee_free) / float(span)


def _dr_phase(step: int, fee_ramp: int, dr_end: int) -> str:
    if step < fee_ramp:
        return "pinned"
    if step < dr_end:
        return "widening"
    return "full"


def build_curriculum_preflight(cfg: RLConfig, *, budget: int | None = None) -> CurriculumPreflight:
    """Compute schedule milestones and reachability warnings for a config."""
    cur = cfg.curriculum
    ent = cfg.entropy_schedule
    hp = cfg.hyperparameters
    tr = cfg.training
    lb = max(1, int(budget if budget is not None else tr.timesteps))

    fee_free, fee_ramp = trade_curriculum_milestones(lb, cur=cur)
    dr_end = dr_widen_end_for_budget(lb, cur)
    gate = resolve_best_model_min_step(lb, cur=cur)
    decay_start = int(ent.decay_start_fraction * lb)
    early_floor_end = int(ent.early_floor_fraction * lb)
    dr_lock = int(ent.dr_lock_fraction * lb)
    stationary = max(0, lb - dr_end)
    patience = int(tr.early_stop_patience)
    # Early stop needs curriculum complete *and* room for ``patience`` post-gate evals.
    # With default eval_freq 500k, need roughly patience * eval_freq residual steps.
    eval_freq = max(1, int(tr.eval_freq_steps))
    early_stop_reachable = (
        patience > 0 and stationary > 0 and (lb - dr_end) >= patience * eval_freq
    )

    lr_hold_until = (
        dr_end if str(getattr(hp, "lr_schedule", "cosine")).lower() == "phase_aware" else 0
    )

    warnings: list[str] = []
    if stationary <= 0:
        warnings.append(
            f"No stationary full-DR phase: dr_widen_end={dr_end} equals budget={lb}."
        )
    if patience > 0 and not early_stop_reachable:
        warnings.append(
            f"early_stop_patience={patience} cannot fire: curriculum ends at "
            f"{dr_end} with only {stationary} steps of residual budget "
            f"(need ≥{patience * eval_freq} for {patience} post-curriculum evals)."
        )
    if gate >= lb:
        warnings.append(f"best-model gate ({gate}) is at/after budget — best/ never opens.")

    named = [
        ("start", 0),
        ("fee_free_end", fee_free),
        ("fee_ramp_end / checkpoint_gate", fee_ramp),
        ("entropy_decay_start", decay_start),
        ("dr_lock_end", dr_lock),
        ("early_floor_end", early_floor_end),
        ("full_DR_start", dr_end),
        ("budget_end", lb),
    ]
    # Deduplicate identical steps while preserving order.
    seen: set[int] = set()
    milestones: list[ScheduleMilestone] = []
    for name, step in named:
        step_i = int(max(0, min(step, lb)))
        key = (name, step_i)
        if key in seen:
            continue
        seen.add(key)
        milestones.append(
            ScheduleMilestone(
                name=name,
                step=step_i,
                fee_override=_fee_override_at(step_i, fee_free, fee_ramp),
                churn_scale=churn_scale_at_step(
                    step_i,
                    fee_free_until=fee_free,
                    fee_ramp_end=fee_ramp,
                    churn_ramp_floor=cur.churn_ramp_floor,
                ),
                ent_coef=_entropy_at_step(step_i, budget=lb, ent=ent),
                learning_rate=lr_at_step(
                    step_i,
                    initial_lr=float(hp.learning_rate),
                    floor_lr=float(hp.learning_rate_floor),
                    budget=lb,
                    hold_until_step=lr_hold_until,
                ),
                dr_phase=_dr_phase(step_i, fee_ramp, dr_end),
            )
        )

    return CurriculumPreflight(
        budget=lb,
        fee_free_until=fee_free,
        fee_ramp_end=fee_ramp,
        dr_widen_end=dr_end,
        best_model_min_step=gate,
        entropy_decay_start=decay_start,
        early_floor_end=early_floor_end,
        dr_lock_end=dr_lock,
        lr_floor=float(hp.learning_rate_floor),
        lr_initial=float(hp.learning_rate),
        lr_schedule=str(getattr(hp, "lr_schedule", "cosine")).lower(),
        lr_hold_until_step=int(lr_hold_until),
        explore_ent=float(ent.explore_ent),
        final_ent=float(ent.final_ent),
        early_stop_patience=patience,
        stationary_full_dr_steps=stationary,
        early_stop_reachable=early_stop_reachable,
        milestones=tuple(milestones),
        warnings=tuple(warnings),
    )


def format_preflight_text(pf: CurriculumPreflight) -> str:
    """Human-readable preflight table for the CLI."""
    lines = [
        f"Curriculum preflight (budget={pf.budget:,} steps)",
        f"  fee_free_until={pf.fee_free_until:,}",
        f"  fee_ramp_end / best-gate={pf.fee_ramp_end:,} / {pf.best_model_min_step:,}",
        f"  dr_widen_end (full DR)={pf.dr_widen_end:,}",
        f"  stationary full-DR steps={pf.stationary_full_dr_steps:,}",
        f"  entropy decay start={pf.entropy_decay_start:,}",
        (
            f"  lr schedule={pf.lr_schedule}"
            + (
                f" (hold {pf.lr_initial:g} until {pf.lr_hold_until_step:,}, then cosine)"
                if pf.lr_schedule == "phase_aware"
                else " (global cosine)"
            )
        ),
        f"  early_stop_patience={pf.early_stop_patience} "
        f"(reachable={pf.early_stop_reachable})",
        "",
        f"{'milestone':<36} {'step':>12} {'fee':>8} {'churn':>7} {'ent':>8} {'lr':>10} {'DR':>9}",
    ]
    for m in pf.milestones:
        fee = "full" if m.fee_override is None else f"{m.fee_override:.3f}"
        lines.append(
            f"{m.name:<36} {m.step:>12,} {fee:>8} {m.churn_scale:>7.3f} "
            f"{m.ent_coef:>8.5f} {m.learning_rate:>10.3e} {m.dr_phase:>9}"
        )
    if pf.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w in pf.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)

