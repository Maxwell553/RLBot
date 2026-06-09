"""Auto-research orchestrator: plan → launch → collect → report → promote.

Shells out to the canonical ``scripts/train.py`` / ``scripts/backtest.py`` CLIs (no
training-stack rewrite) and records every run in ``Runs/<cohort>/registry.jsonl``.
Enforces the OOS firewall: tiers 1–3 train + in-training eval only; tier ≥ 4 reads the
holdout once per variant and requires ``--promote``.

Usage:
    python scripts/research.py plan    specs/feature_split_ab.yaml
    python scripts/research.py launch  specs/feature_split_ab.yaml [--backend local] [--dry-run]
    python scripts/research.py collect <cohort>
    python scripts/research.py report  <cohort>
    python scripts/research.py promote specs/feature_split_ab.yaml --variant <id> --promote
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from rlbot.research import gates, registry, report  # noqa: E402
from rlbot.research.spec import (  # noqa: E402
    ExperimentSpec,
    build_variant_config_dict,
    load_spec,
    resolve_variants,
)
from rlbot.rl_config import load_config  # noqa: E402
from rlbot.run_artifacts import PROJECT_ROOT, RunPaths, read_run_manifest  # noqa: E402

REPO = PROJECT_ROOT
RUNS = REPO / "Runs"


def _cohort_dir(cohort: str) -> Path:
    return RUNS / cohort


def _registry_path(cohort: str) -> Path:
    return _cohort_dir(cohort) / "registry.jsonl"


def _read_json(path: Path) -> dict | None:
    """Read an optional JSON artifact. Missing → None; present-but-malformed → warn (not silent)."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"[research] WARNING: could not read {path}: {e}", file=sys.stderr)
        return None


def _materialize(spec: ExperimentSpec) -> dict:
    """Write per-variant config files + a cohort manifest. Returns the cohort manifest."""
    cohort = spec.id
    cdir = _cohort_dir(cohort)
    (cdir / "configs").mkdir(parents=True, exist_ok=True)
    base_path = (REPO / spec.base_config) if not Path(spec.base_config).is_absolute() else Path(spec.base_config)
    base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    base_dict = load_config(base_path).to_dict()

    entries = []
    for v in resolve_variants(spec):
        cfg_dict = build_variant_config_dict(base_dict, v.concrete_patch)
        cfg_path = cdir / "configs" / f"{v.variant_id}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
        load_config(cfg_path)  # validate eagerly (raises on a bad patch value)
        entries.append(
            {
                "variant_id": v.variant_id,
                "run_id": v.variant_id,
                "config_path": str(cfg_path),
                "seed": v.seed,
                "window": v.window,
                "patch": v.concrete_patch,
                "evaluation_tier": spec.evaluation_tier,
            }
        )
    cohort_manifest = {
        "cohort": cohort,
        "hypothesis": spec.hypothesis,
        "parent": spec.parent,
        "checkpoint_rule": spec.checkpoint_rule,
        "evaluation_tier": spec.evaluation_tier,
        "success_gates": spec.success_gates,
        "budget": spec.budget,
        "timesteps": spec.timesteps,
        "base_config": spec.base_config,
        "base_config_sha256": base_sha,
        "variants": entries,
    }
    (cdir / "cohort.json").write_text(json.dumps(cohort_manifest, indent=2), encoding="utf-8")
    return cohort_manifest


def _train_cmd(entry: dict, spec: ExperimentSpec) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "train.py"),
        "--config",
        entry["config_path"],
        "--run-id",
        entry["run_id"],
        "--seed",
        str(entry["seed"]),
        "--no-viz",
    ]
    if spec.timesteps:
        cmd += ["--timesteps", str(spec.timesteps)]
    w = entry.get("window") or {}
    for flag, key in (("--train-end", "train_end"), ("--holdout-start", "holdout_start"),
                      ("--holdout-end", "holdout_end")):
        if w.get(key):
            cmd += [flag, str(w[key])]
    return cmd


def _backtest_cmd(entry: dict) -> list[str]:
    return [
        sys.executable,
        str(REPO / "scripts" / "backtest.py"),
        "--run-id",
        entry["run_id"],
        "--checkpoint",
        "best",
        "--detailed",
    ]


# ── commands ──────────────────────────────────────────────────────────────
def cmd_plan(args: argparse.Namespace) -> None:
    spec = load_spec(args.spec)
    cm = _materialize(spec)
    print(f"Cohort '{cm['cohort']}': {len(cm['variants'])} variant(s), tier {cm['evaluation_tier']} "
          f"({gates.tier_label(cm['evaluation_tier'])})")
    for e in cm["variants"]:
        print(f"  {e['variant_id']}  seed={e['seed']}  patch={e['patch']}")
    print(f"Configs materialized under {_cohort_dir(cm['cohort']) / 'configs'}")


def _collect_one(cm: dict, entry: dict, *, tier: int, status: str = "ok",
                 failure: str | None = None) -> dict:
    """Build one registry record for a variant from its run artifacts (may be sparse)."""
    run_id = entry["run_id"]
    manifest = read_run_manifest(run_id)
    rp = RunPaths(run_id)
    training_summary = _read_json(rp.run_meta_dir / "training_summary.json")
    backtest_summary = _read_json(rp.run_meta_dir / "backtest_summary.json")
    return registry.build_record(
        cohort=cm["cohort"],
        variant_id=entry["variant_id"],
        hypothesis=cm.get("hypothesis", ""),
        run_id=run_id,
        evaluation_tier=tier,
        manifest=manifest,
        training_summary=training_summary,
        backtest_summary=backtest_summary,
        status=status,
        failure=failure,
    )


def _scored_keys(records: list[dict]) -> set[tuple[str, int]]:
    """(run_id, tier) pairs that already have a scored ('ok') registry record."""
    return {
        (str(r.get("run_id")), int(r.get("evaluation_tier", 0)))
        for r in records
        if str(r.get("status", "ok")) == "ok"
    }


def cmd_launch(args: argparse.Namespace) -> None:
    spec = load_spec(args.spec)
    gates.assert_tier_allowed(spec.evaluation_tier, promoted=args.promote)
    cm = _materialize(spec)
    reg = _registry_path(spec.id)
    touches_oos = gates.tier_touches_oos(spec.evaluation_tier)
    tier = int(cm["evaluation_tier"])
    variants = cm["variants"]
    if spec.success_gates or spec.budget:
        print("[research] note: spec success_gates/budget are recorded in cohort.json "
              "but not enforced by the orchestrator.")
    if touches_oos:
        gates.assert_oos_budget(len(variants), args.oos_budget)
        print(f"[research] WARNING: this launch will read the OOS holdout for "
              f"{len(variants)} variant(s) (budget {args.oos_budget}).")
    failures: list[tuple[str, str]] = []
    cohort_t0 = time.perf_counter()
    for n, e in enumerate(variants, 1):
        # Re-read per iteration: records appended during this sweep (or by a concurrent
        # launch) must count. The gate stays a hard fail — never caught.
        existing = registry.read_records(reg)
        if (e["run_id"], tier) in _scored_keys(existing):
            print(f"[research] [{n}/{len(variants)}] {e['run_id']} already collected at "
                  f"tier {tier}; skipping (resume).")
            continue
        if touches_oos:
            gates.assert_no_repeat_oos(existing, e["variant_id"])
        train = _train_cmd(e, spec)
        bt = _backtest_cmd(e) if touches_oos else None
        if args.dry_run:
            print("DRY-RUN train:", " ".join(train))
            if bt:
                print("DRY-RUN backtest:", " ".join(bt))
            continue
        # Per-variant resilience: a failed run is logged + recorded, the sweep continues.
        t0 = time.perf_counter()
        print(f"[research] [{n}/{len(variants)}] training {e['run_id']} (cwd={REPO}) ...")
        try:
            subprocess.run(train, check=True, cwd=str(REPO))
            if bt:
                # Record the OOS read BEFORE it happens, so a crash between backtest and
                # collect can never leave an unaccounted holdout read.
                registry.append_record(
                    reg, _collect_one(cm, e, tier=tier, status="oos_read_attempt")
                )
                print(f"[research] [{n}/{len(variants)}] backtest (OOS) {e['run_id']} ...")
                subprocess.run(bt, check=True, cwd=str(REPO))
            registry.append_record(reg, _collect_one(cm, e, tier=tier))
            print(f"[research] [{n}/{len(variants)}] {e['run_id']} done "
                  f"({time.perf_counter() - t0:.0f}s)")
        except subprocess.CalledProcessError as exc:
            msg = f"exit {exc.returncode} from {' '.join(exc.cmd[:3])}..."
            print(f"[research] ERROR: variant {e['run_id']} failed: {msg}", file=sys.stderr)
            registry.append_record(
                reg, _collect_one(cm, e, tier=tier, status="failed", failure=msg)
            )
            failures.append((e["run_id"], msg))
    if not args.dry_run:
        print(f"[research] cohort {spec.id} ran {len(variants)} variant(s) in "
              f"{time.perf_counter() - cohort_t0:.0f}s, {len(failures)} failure(s)")
        if failures:
            for rid, msg in failures:
                print(f"[research]   FAILED {rid}: {msg}", file=sys.stderr)
            raise SystemExit(f"{len(failures)} variant(s) failed; see log above.")


def cmd_collect(args: argparse.Namespace) -> None:
    cohort = getattr(args, "cohort", None) or load_spec(args.spec).id
    cm = _read_json(_cohort_dir(cohort) / "cohort.json")
    if not cm:
        raise SystemExit(f"No cohort.json for {cohort!r}; run `plan`/`launch` first.")
    reg = _registry_path(cohort)
    tier = int(cm.get("evaluation_tier", 1))
    seen = _scored_keys(registry.read_records(reg))
    n_new = 0
    for e in cm["variants"]:
        run_id = e["run_id"]
        if (run_id, tier) in seen:
            continue
        if read_run_manifest(run_id) is None:
            continue  # not trained yet
        registry.append_record(reg, _collect_one(cm, e, tier=tier))
        n_new += 1
    print(f"[research] collected {n_new} new record(s) into {reg}")


def cmd_report(args: argparse.Namespace) -> None:
    cohort = args.cohort
    records = registry.read_records(_registry_path(cohort))
    out = _cohort_dir(cohort) / "report.md"
    report.write_report(records, out, title=f"Research cohort: {cohort}")
    print(f"[research] wrote {out} ({len(records)} records)")


def cmd_promote(args: argparse.Namespace) -> None:
    """Tier-4 OOS read for a single promoted variant (requires --promote)."""
    spec = load_spec(args.spec)
    if not args.promote:
        raise SystemExit("promote requires --promote (it reads the OOS holdout).")
    cm = _materialize(spec)
    entry = next((e for e in cm["variants"] if e["variant_id"] == args.variant), None)
    if entry is None:
        raise SystemExit(f"variant {args.variant!r} not in cohort {spec.id!r}")
    reg = _registry_path(spec.id)
    promote_tier = max(4, int(cm.get("evaluation_tier", 1)))
    existing = registry.read_records(reg)
    gates.assert_no_repeat_oos(
        existing, entry["variant_id"], allow_failed_rescore=args.allow_failed_rescore
    )
    bt = _backtest_cmd(entry)
    if args.dry_run:
        print("DRY-RUN promote backtest:", " ".join(bt))
        return
    # Record the OOS read BEFORE it happens: a crash below leaves an attempt record,
    # so the no-repeat gate fails closed instead of allowing silent re-reads.
    registry.append_record(
        reg, _collect_one(cm, entry, tier=promote_tier, status="oos_read_attempt")
    )
    try:
        subprocess.run(bt, check=True, cwd=str(REPO))
    except subprocess.CalledProcessError as exc:
        msg = f"exit {exc.returncode} from backtest"
        registry.append_record(
            reg, _collect_one(cm, entry, tier=promote_tier, status="failed", failure=msg)
        )
        raise SystemExit(f"promotion backtest failed for {entry['run_id']}: {msg}")
    record = _collect_one(cm, entry, tier=promote_tier)
    registry.append_record(reg, record)
    print(f"[research] promoted {entry['run_id']} at tier {promote_tier}: "
          f"OOS return={record.get('oos_total_return')} sharpe={record.get('oos_sharpe')} "
          f"maxDD={record.get('oos_max_drawdown')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("plan"); sp.add_argument("spec"); sp.set_defaults(func=cmd_plan)
    sl = sub.add_parser("launch")
    sl.add_argument("spec")
    sl.add_argument("--backend", default="local", choices=("local",),
                    help="execution backend (only 'local' is implemented)")
    sl.add_argument("--promote", action="store_true")
    sl.add_argument("--dry-run", action="store_true")
    sl.add_argument(
        "--oos-budget", type=int, default=1,
        help="max holdout reads a tier>=4 launch may perform (default 1; raising this "
             "is an explicit multiple-testing decision)",
    )
    sl.set_defaults(func=cmd_launch)
    sc = sub.add_parser("collect"); sc.add_argument("cohort"); sc.set_defaults(func=cmd_collect, spec=None)
    sr = sub.add_parser("report"); sr.add_argument("cohort"); sr.set_defaults(func=cmd_report)
    spm = sub.add_parser("promote")
    spm.add_argument("spec")
    spm.add_argument("--variant", required=True)
    spm.add_argument("--promote", action="store_true")
    spm.add_argument("--dry-run", action="store_true")
    spm.add_argument(
        "--allow-failed-rescore", action="store_true",
        help="retry the holdout read for a variant whose previous tier-4 read crashed "
             "before producing a score (refused by default, fail-closed)",
    )
    spm.set_defaults(func=cmd_promote)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
