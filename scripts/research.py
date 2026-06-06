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


def cmd_launch(args: argparse.Namespace) -> None:
    spec = load_spec(args.spec)
    gates.assert_tier_allowed(spec.evaluation_tier, promoted=args.promote)
    cm = _materialize(spec)
    existing = registry.read_records(_registry_path(spec.id))
    touches_oos = gates.tier_touches_oos(spec.evaluation_tier)
    variants = cm["variants"]
    failures: list[tuple[str, str]] = []
    cohort_t0 = time.perf_counter()
    for n, e in enumerate(variants, 1):
        if touches_oos:
            gates.assert_no_repeat_oos(existing, e["variant_id"])  # firewall: hard-fail, never caught
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
                print(f"[research] [{n}/{len(variants)}] backtest (OOS) {e['run_id']} ...")
                subprocess.run(bt, check=True, cwd=str(REPO))
            print(f"[research] [{n}/{len(variants)}] {e['run_id']} done "
                  f"({time.perf_counter() - t0:.0f}s)")
        except subprocess.CalledProcessError as exc:
            msg = f"exit {exc.returncode} from {' '.join(exc.cmd[:3])}..."
            print(f"[research] ERROR: variant {e['run_id']} failed: {msg}", file=sys.stderr)
            failures.append((e["run_id"], msg))
    if not args.dry_run:
        print(f"[research] cohort {spec.id} ran {len(variants)} variant(s) in "
              f"{time.perf_counter() - cohort_t0:.0f}s, {len(failures)} failure(s)")
        cmd_collect(args)
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
    seen = {r.get("run_id") for r in registry.read_records(reg)}
    n_new = 0
    for e in cm["variants"]:
        run_id = e["run_id"]
        if run_id in seen:
            continue
        manifest = read_run_manifest(run_id)
        if manifest is None:
            continue  # not trained yet
        rp = RunPaths(run_id)
        training_summary = _read_json(rp.run_meta_dir / "training_summary.json")
        backtest_summary = _read_json(rp.run_meta_dir / "backtest_summary.json")
        record = registry.build_record(
            cohort=cohort,
            variant_id=e["variant_id"],
            hypothesis=cm.get("hypothesis", ""),
            run_id=run_id,
            evaluation_tier=cm.get("evaluation_tier", 1),
            manifest=manifest,
            training_summary=training_summary,
            backtest_summary=backtest_summary,
        )
        registry.append_record(reg, record)
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
    existing = registry.read_records(_registry_path(spec.id))
    gates.assert_no_repeat_oos(existing, entry["variant_id"])
    bt = _backtest_cmd(entry)
    if args.dry_run:
        print("DRY-RUN promote backtest:", " ".join(bt))
        return
    subprocess.run(bt, check=True, cwd=str(REPO))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("plan"); sp.add_argument("spec"); sp.set_defaults(func=cmd_plan)
    sl = sub.add_parser("launch")
    sl.add_argument("spec")
    sl.add_argument("--backend", default="local", choices=("local", "modal"))
    sl.add_argument("--promote", action="store_true")
    sl.add_argument("--dry-run", action="store_true")
    sl.set_defaults(func=cmd_launch)
    sc = sub.add_parser("collect"); sc.add_argument("cohort"); sc.set_defaults(func=cmd_collect, spec=None)
    sr = sub.add_parser("report"); sr.add_argument("cohort"); sr.set_defaults(func=cmd_report)
    spm = sub.add_parser("promote")
    spm.add_argument("spec")
    spm.add_argument("--variant", required=True)
    spm.add_argument("--promote", action="store_true")
    spm.add_argument("--dry-run", action="store_true")
    spm.set_defaults(func=cmd_promote)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
