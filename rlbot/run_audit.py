"""Scan ``Runs/`` and build a compact audit report.

Surfaces provenance, elapsed steps, resume lineage, dirty-source status, OOS metrics,
benchmark excess, DSR / trial counts, best-checkpoint diagnostics, reward-decomp
history, and explicit non-comparability labels — so historical ambiguities do not
require detective work.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from rlbot.cohort_labels import label_run
from rlbot.curriculum_preflight import build_curriculum_preflight, dr_widen_end_for_budget
from rlbot.rl_config import load_config, trade_curriculum_milestones
from rlbot.run_artifacts import PROJECT_ROOT, RunPaths, read_run_manifest

_RUN_DIR_RE = re.compile(r"^W\d+_\d+")


@dataclass
class RunAuditRecord:
    run_id: str
    labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    comparable: bool = True
    training_status: str | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    source_patch: str | None = None
    source_tree_hash: str | None = None
    nominal_timesteps: int | None = None
    elapsed_timesteps: int | None = None
    resume_parent: str | None = None
    resume_parent_step: int | None = None
    best_eval_step: int | None = None
    best_eval_score: float | None = None
    early_stop_reason: str | None = None
    curriculum_stage_at_best: str | None = None
    oos_return: float | None = None
    oos_sharpe: float | None = None
    oos_max_dd: float | None = None
    oos_deflated_sharpe: float | None = None
    oos_deflated_sharpe_excess: float | None = None
    oos_trials: int | None = None
    oos_trials_conservative: int | None = None
    ew_excess_return: float | None = None
    reward_decomp_n_snapshots: int = 0
    reward_decomp_final_abs_share: dict[str, float] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_resume_parent_step(resume_path: str) -> int | None:
    """Extract step from ``.../ppo_{steps}_steps.zip`` checkpoint names."""
    m = re.search(r"ppo_(\d+)_steps", resume_path)
    if not m:
        return None
    return int(m.group(1))


def _curriculum_stage(step: int | None, budget: int, fee_free: int, fee_ramp: int, dr_end: int) -> str | None:
    if step is None:
        return None
    if step < fee_free:
        return "fee_free"
    if step < fee_ramp:
        return "fee_ramp"
    if step < dr_end:
        return "dr_widening"
    if step < budget:
        return "full_dr"
    return "at_or_past_budget"


def _infer_elapsed_timesteps(paths: RunPaths, manifest: Mapping[str, Any]) -> int | None:
    """Best-effort actual elapsed steps (not the nominal budget)."""
    # Prefer explicit stamp (new runs).
    for key in ("elapsed_timesteps", "num_timesteps", "cumulative_timesteps"):
        v = manifest.get(key)
        if isinstance(v, (int, float)) and int(v) > 0:
            return int(v)
    # Eval history last timestep.
    nav = paths.eval_nav_history
    if nav.is_file():
        try:
            import numpy as np

            z = np.load(nav, allow_pickle=False)
            steps = np.asarray(z["timesteps"], dtype=np.int64)
            if steps.size:
                return int(steps[-1])
        except (OSError, ValueError, KeyError):
            pass
    # Checkpoint filenames.
    ckpt_dir = paths.models_dir / "checkpoints"
    if ckpt_dir.is_dir():
        best = 0
        for p in ckpt_dir.glob("ppo_*_steps.zip"):
            m = re.search(r"ppo_(\d+)_steps", p.name)
            if m:
                best = max(best, int(m.group(1)))
        if best > 0:
            return best
    # Resume parent + nominal remaining is unknowable; fall back to best_eval_step.
    bes = manifest.get("best_eval_step")
    if isinstance(bes, (int, float)):
        return int(bes)
    return None


def _reward_decomp_history(paths: RunPaths) -> tuple[int, dict[str, float] | None]:
    jsonl = paths.eval_log_dir / "reward_decomp.jsonl"
    n = 0
    last_share: dict[str, float] | None = None
    if jsonl.is_file():
        try:
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n += 1
                share = rec.get("abs_share")
                if isinstance(share, dict):
                    last_share = {k: float(v) for k, v in share.items()}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    if last_share is None:
        legacy = _load_json(paths.eval_log_dir / "reward_decomp.json")
        if legacy and isinstance(legacy.get("abs_share"), dict):
            last_share = {k: float(v) for k, v in legacy["abs_share"].items()}
            n = max(n, 1)
    return n, last_share


def _conservative_trial_count(window: str | None, distinct_runs: int) -> int:
    """Distinct runs plus a checkpoint-candidate pad (best vs final/latest)."""
    if not window:
        return max(1, distinct_runs)
    # Conservative: each distinct model may have contributed best + final candidates.
    return max(1, int(distinct_runs) * 2)


def audit_run(run_id: str, *, root: Path = PROJECT_ROOT) -> RunAuditRecord:
    paths = RunPaths(run_id=run_id, root=root)
    manifest = read_run_manifest(run_id) or {}
    config = _load_yaml(paths.config_snapshot) or {}
    label = label_run(run_id, manifest=manifest, config=config)
    rec = RunAuditRecord(
        run_id=run_id,
        labels=list(label.labels),
        notes=list(label.notes),
        comparable=label.comparable,
        training_status=manifest.get("training_status"),
        started_at_utc=manifest.get("started_at_utc"),
        finished_at_utc=manifest.get("finished_at_utc"),
        git_commit=manifest.get("git_commit"),
        git_dirty=manifest.get("git_dirty"),
        source_patch=(manifest.get("provenance") or {}).get("source_patch")
        if isinstance(manifest.get("provenance"), dict)
        else manifest.get("source_patch"),
        source_tree_hash=(
            ((manifest.get("provenance") or {}).get("source_hashes") or {}).get("__tree__")
            if isinstance(manifest.get("provenance"), dict)
            else None
        ),
        best_eval_step=manifest.get("best_eval_step")
        if isinstance(manifest.get("best_eval_step"), (int, float))
        else None,
        best_eval_score=manifest.get("best_eval_score")
        if isinstance(manifest.get("best_eval_score"), (int, float))
        else None,
        early_stop_reason=manifest.get("early_stop_reason"),
    )

    args = manifest.get("args") or {}
    if isinstance(args.get("timesteps"), (int, float)):
        rec.nominal_timesteps = int(args["timesteps"])
    resume = str(args.get("resume") or "").strip() or None
    rec.resume_parent = resume
    if resume:
        rec.resume_parent_step = _parse_resume_parent_step(resume)

    rec.elapsed_timesteps = _infer_elapsed_timesteps(paths, manifest)
    if (
        rec.resume_parent_step is not None
        and rec.elapsed_timesteps is not None
        and rec.elapsed_timesteps < rec.resume_parent_step
    ):
        # Elapsed from this session alone; report cumulative when we can.
        rec.warnings.append(
            f"elapsed_timesteps ({rec.elapsed_timesteps}) < resume parent step "
            f"({rec.resume_parent_step}); prefer cumulative stamp when present."
        )

    budget = rec.nominal_timesteps or 50_000_000
    try:
        cfg = load_config(paths.config_snapshot) if paths.config_snapshot.is_file() else load_config()
        fee_free, fee_ramp = trade_curriculum_milestones(budget, cur=cfg.curriculum)
        dr_end = dr_widen_end_for_budget(budget, cfg.curriculum)
        rec.curriculum_stage_at_best = _curriculum_stage(
            rec.best_eval_step, budget, fee_free, fee_ramp, dr_end
        )
        pf = build_curriculum_preflight(cfg, budget=budget)
        if not pf.early_stop_reachable and int(cfg.training.early_stop_patience) > 0:
            rec.warnings.append("early_stop_unreachable")
        if pf.stationary_full_dr_steps <= 0:
            rec.warnings.append("no_stationary_full_dr_phase")
    except Exception as exc:  # noqa: BLE001 — audit must not crash on one bad config
        rec.warnings.append(f"curriculum_preflight_failed: {exc}")

    bt = _load_json(paths.run_meta_dir / "backtest_summary.json")
    if bt:
        rec.oos_return = bt.get("total_return")
        rec.oos_sharpe = bt.get("sharpe")
        rec.oos_max_dd = bt.get("max_drawdown")
        rec.oos_deflated_sharpe = bt.get("deflated_sharpe")
        rec.oos_deflated_sharpe_excess = bt.get("deflated_sharpe_excess")
        rec.oos_trials = bt.get("oos_trials_for_window")
        rec.oos_trials_conservative = bt.get("oos_trials_conservative")
        if rec.oos_trials_conservative is None and isinstance(rec.oos_trials, int):
            window = bt.get("oos_window")
            rec.oos_trials_conservative = _conservative_trial_count(window, rec.oos_trials)
        # Benchmark excess vs equal-weight if present in detailed / diagnostics.
        ew = (bt.get("detailed") or {}).get("equal_weight_daily_return") if isinstance(bt.get("detailed"), dict) else None
        if ew is None:
            ew = bt.get("equal_weight_daily_return")
        if isinstance(rec.oos_return, (int, float)) and isinstance(ew, (int, float)):
            rec.ew_excess_return = float(rec.oos_return) - float(ew)

    n_snap, share = _reward_decomp_history(paths)
    rec.reward_decomp_n_snapshots = n_snap
    rec.reward_decomp_final_abs_share = share

    if rec.git_dirty and not rec.source_patch:
        # Old runs: dirty but no patch persisted.
        if (paths.run_meta_dir / "provenance" / "git.diff").is_file():
            rec.source_patch = "provenance/git.diff"
        else:
            rec.warnings.append("dirty_source_without_persisted_patch")

    return rec


def discover_audit_run_ids(*, root: Path = PROJECT_ROOT, prefix: str = "") -> list[str]:
    runs = root / "Runs"
    if not runs.is_dir():
        return []
    out: list[str] = []
    for p in sorted(runs.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if prefix and not p.name.startswith(prefix):
            continue
        if (p / "manifest.json").is_file() or _RUN_DIR_RE.match(p.name):
            out.append(p.name)
    return out


def audit_runs(
    run_ids: Iterable[str] | None = None,
    *,
    root: Path = PROJECT_ROOT,
    prefix: str = "",
) -> list[RunAuditRecord]:
    ids = list(run_ids) if run_ids is not None else discover_audit_run_ids(root=root, prefix=prefix)
    return [audit_run(rid, root=root) for rid in ids]


def format_audit_text(records: list[RunAuditRecord], *, comparable_only: bool = False) -> str:
    rows = [r for r in records if r.comparable] if comparable_only else records
    lines = [
        f"Run audit ({len(rows)} runs"
        + (f", filtered from {len(records)}" if comparable_only else "")
        + ")",
        "",
        f"{'run_id':<12} {'labels':<28} {'elapsed':>10} {'best@':>10} "
        f"{'OOS%':>7} {'Sh':>5} {'DSR':>5} {'status':<12}",
    ]
    for r in rows:
        lab = ",".join(x for x in r.labels if x != "clean")[:28] or "clean"
        elapsed = f"{r.elapsed_timesteps:,}" if r.elapsed_timesteps else "-"
        best = f"{r.best_eval_step:,}" if r.best_eval_step else "-"
        oos = f"{100 * r.oos_return:.1f}" if isinstance(r.oos_return, float) else "-"
        sh = f"{r.oos_sharpe:.2f}" if isinstance(r.oos_sharpe, float) else "-"
        dsr = f"{r.oos_deflated_sharpe:.2f}" if isinstance(r.oos_deflated_sharpe, float) else "-"
        lines.append(
            f"{r.run_id:<12} {lab:<28} {elapsed:>10} {best:>10} "
            f"{oos:>7} {sh:>5} {dsr:>5} {str(r.training_status or '-'):<12}"
        )
    flagged = [r for r in records if not r.comparable]
    if flagged:
        lines.append("")
        lines.append(f"Non-comparable ({len(flagged)}):")
        for r in flagged:
            note = r.notes[0] if r.notes else ""
            lines.append(f"  {r.run_id}: {','.join(r.labels)} — {note}")
    return "\n".join(lines)


def write_audit_json(records: list[RunAuditRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_runs": len(records),
        "n_comparable": sum(1 for r in records if r.comparable),
        "runs": [r.to_dict() for r in records],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
