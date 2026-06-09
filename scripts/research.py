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
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from rlbot.research import gates, oos_ledger, registry, report  # noqa: E402
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
    """Write per-variant config files + a cohort manifest. Returns the cohort manifest.

    Refuses to overwrite a cohort whose registry already holds records when the spec
    file has changed since the original materialization — re-planning an edited spec
    over trained runs would silently relabel the plan of record (and refresh the
    spec_sha256 that promote's edit guard checks). Register a new spec id instead.
    """
    cohort = spec.id
    cdir = _cohort_dir(cohort)
    prior = _read_json(cdir / "cohort.json")
    if prior:
        prior_sha = prior.get("spec_sha256")
        if (
            prior_sha
            and spec.source_sha256
            and prior_sha != spec.source_sha256
            and registry.read_records(_registry_path(cohort))
        ):
            raise SystemExit(
                f"Cohort {cohort!r} already has registry records trained under a "
                f"different spec (sha {str(prior_sha)[:12]} != {str(spec.source_sha256)[:12]}). "
                "Re-planning an edited spec over trained runs would relabel the plan of "
                "record — register a NEW spec id for the changed experiment."
            )
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
                "group_id": v.group_id,
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
        "spec_sha256": getattr(spec, "source_sha256", None),
        "variants": entries,
    }
    (cdir / "cohort.json").write_text(json.dumps(cohort_manifest, indent=2), encoding="utf-8")
    return cohort_manifest


def _train_cmd(
    entry: dict,
    spec: ExperimentSpec,
    overwrite_run: bool = False,
    backend: str = "local",
    modal_gpu: str | None = None,
    timesteps_override: int | None = None,
) -> list[str]:
    cfg_path = Path(entry["config_path"])
    # Repo-relative config path: subprocesses run with cwd=REPO locally, and the
    # remote container mounts the runs volume at the same relative layout — an
    # absolute local path would not exist inside the container.
    if cfg_path.is_absolute():
        try:
            cfg_path = cfg_path.relative_to(REPO)
        except ValueError:
            pass  # outside the repo: leave absolute (local backend only)
    train_flags = [
        "--config",
        str(cfg_path),
        "--run-id",
        entry["run_id"],
        "--seed",
        str(entry["seed"]),
        "--no-viz",
    ]
    if overwrite_run:
        # Deliberate retry of a crashed/failed (never-scored) variant: train.py's
        # run-dir guard would otherwise refuse the deterministic run id forever.
        train_flags.append("--overwrite-run")
    ts = timesteps_override if timesteps_override is not None else spec.timesteps
    if ts:
        train_flags += ["--timesteps", str(ts)]
    w = entry.get("window") or {}
    for flag, key in (("--train-end", "train_end"), ("--holdout-start", "holdout_start"),
                      ("--holdout-end", "holdout_end")):
        if w.get(key):
            train_flags += [flag, str(w[key])]
    if backend == "modal":
        from rlbot.modal_cloud import modal_cli

        gpu_flags = ["--modal-gpu", modal_gpu] if modal_gpu else []
        # modal_app.py registers several local entrypoints; `modal run file.py` cannot
        # dispatch among them — the ::train entrypoint must be named explicitly.
        return [*modal_cli(), "run",
                str(REPO / "scripts" / "modal_app.py") + "::train", "--",
                *gpu_flags, *train_flags]
    return [sys.executable, str(REPO / "scripts" / "train.py"), *train_flags]


def _modal_pre_train(entry: dict) -> None:
    """Push the variant config to the Modal runs volume (the remote train reads the
    same repo-relative Runs/<cohort>/configs/... path off the mounted volume)."""
    from rlbot.modal_cloud import VOLUME_RUNS

    cfg = Path(entry["config_path"]).resolve()
    rel = cfg.relative_to((REPO / "Runs").resolve())
    from rlbot.modal_cloud import modal_cli

    subprocess.run(
        [*modal_cli(), "volume", "put", VOLUME_RUNS, str(cfg), str(rel), "--force"],
        check=True,
        cwd=str(REPO),
    )


def _modal_post_train(entry: dict) -> None:
    """Pull the finished run tree from the Modal volume so collect/backtest see it.

    Verifies the pull actually produced the run's manifest + training summary —
    a hollow sync followed by collect would append a scored-looking record with
    null metrics and permanently block the relaunch."""
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "modal_app.py"), "sync",
         "--run-id", entry["run_id"], "--pull-all"],
        check=True,
        cwd=str(REPO),
    )
    rp = RunPaths(entry["run_id"])
    missing = [
        str(p) for p in (rp.manifest_path, rp.run_meta_dir / "training_summary.json")
        if not p.is_file()
    ]
    if missing:
        raise subprocess.CalledProcessError(
            1, "modal-sync-verify", output=f"pulled run is missing {missing}"
        )


def _oos_env(cohort: str, window_budget: int | None = None) -> dict:
    """Subprocess env for gated backtests: stamps the ledger context + budget so the
    backtest re-checks the cumulative window budget atomically at read time."""
    env = dict(os.environ)
    env["RLBOT_OOS_CONTEXT"] = f"research:{cohort}"
    if window_budget is not None:
        env["RLBOT_WINDOW_BUDGET"] = str(int(window_budget))
    return env


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
    # A pre-read attempt record must never look scored: a stale backtest_summary.json
    # (e.g. a hand-run backtest) would otherwise flatten OOS metrics into a record
    # whose status says the read has not happened yet.
    backtest_summary = (
        None if status == "oos_read_attempt"
        else _read_json(rp.run_meta_dir / "backtest_summary.json")
    )
    return registry.build_record(
        cohort=cm["cohort"],
        variant_id=entry["variant_id"],
        group_id=entry.get("group_id") or None,
        patch=entry.get("patch") or None,
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
    if (spec.budget or {}).get("max_modal_hours"):
        print(f"[research] note: budget.max_modal_hours={spec.budget['max_modal_hours']} "
              "is enforced as a per-variant wall-clock cap (local kill; on modal the "
              "remote app must be stopped manually if the client dies).")
    if touches_oos:
        windowless = [
            e["variant_id"] for e in variants
            if not ((e.get("window") or {}).get("holdout_start"))
        ]
        if windowless:
            raise SystemExit(
                f"tier-{tier} spec has variant(s) without a canonical window: "
                f"{windowless}. OOS reads must name W1–W5 explicitly — the config-"
                "default calendar-tail holdout would bypass per-window budgets and "
                "can overlap the embargoed W6 range."
            )
        already = _scored_keys(registry.read_records(reg, on_corrupt="raise"))
        pending = [e for e in variants if (e["run_id"], tier) not in already]
        gates.assert_oos_budget(len(pending), args.oos_budget)
        # Cumulative per-window burn budget (global ledger, across all cohorts).
        by_window: dict[str, list[str]] = {}
        for e in pending:
            w = e.get("window") or {}
            if w.get("holdout_start") and w.get("holdout_end"):
                wkey = oos_ledger.window_key(w["holdout_start"], w["holdout_end"])
                by_window.setdefault(wkey, []).append(e["run_id"])
        ledger_records = oos_ledger.read_ledger(on_corrupt="raise")
        for wkey, rids in by_window.items():
            oos_ledger.assert_window_budget(
                ledger_records, wkey, rids,
                budget=(args.window_budget if getattr(args, "window_budget", None) is not None else oos_ledger.DEFAULT_WINDOW_BUDGET),
            )
        print(f"[research] WARNING: this launch will read the OOS holdout for "
              f"{len(pending)} variant(s) (budget {args.oos_budget}; cumulative "
              "per-window budgets enforced from Runs/oos_ledger.jsonl).")
    failures: list[tuple[str, str]] = []
    cohort_t0 = time.perf_counter()
    for n, e in enumerate(variants, 1):
        # Re-read per iteration: records appended during this sweep (or by a concurrent
        # launch) must count. Gate reads are STRICT (fail closed on corruption) — never caught.
        existing = registry.read_records(reg, on_corrupt="raise" if touches_oos else "skip")
        if (e["run_id"], tier) in _scored_keys(existing):
            print(f"[research] [{n}/{len(variants)}] {e['run_id']} already collected at "
                  f"tier {tier}; skipping (resume).")
            continue
        if touches_oos:
            gates.assert_no_repeat_oos(existing, e["variant_id"])
        # A leftover run dir here means a prior attempt crashed or failed (scored
        # variants were skipped above) — retry must overwrite, not brick the relaunch.
        backend = getattr(args, "backend", "local")
        # Local: a leftover manifest means a crashed/failed (never-scored) attempt.
        # Modal: the stale dir lives on the VOLUME where we cannot cheaply look —
        # any unscored variant retry must overwrite (train.py only acts on the flag
        # when a manifest actually exists).
        stale_run_dir = RunPaths(e["run_id"]).manifest_path.is_file() or backend == "modal"
        if stale_run_dir and backend != "modal":
            print(f"[research] [{n}/{len(variants)}] {e['run_id']}: stale unscored run dir "
                  "from a previous attempt; retraining with --overwrite-run.")
        train = _train_cmd(
            e, spec, overwrite_run=stale_run_dir, backend=backend,
            modal_gpu=getattr(args, "modal_gpu", None),
        )
        bt = _backtest_cmd(e) if touches_oos else None
        if args.dry_run:
            print("DRY-RUN train:", " ".join(train))
            if bt:
                print("DRY-RUN backtest:", " ".join(bt))
            continue
        # Per-variant resilience: a failed run is logged + recorded, the sweep continues.
        t0 = time.perf_counter()
        print(f"[research] [{n}/{len(variants)}] training {e['run_id']} "
              f"(backend={backend}, cwd={REPO}) ...")
        # spec.budget.max_modal_hours = per-variant wall-clock cap (enforced on both
        # backends; a hung remote/local train must not stall the whole cohort).
        timeout_s = None
        max_h = (spec.budget or {}).get("max_modal_hours")
        if max_h:
            timeout_s = float(max_h) * 3600.0
        try:
            if backend == "modal":
                _modal_pre_train(e)
            subprocess.run(train, check=True, cwd=str(REPO), timeout=timeout_s)
            if backend == "modal":
                _modal_post_train(e)
            if bt:
                # Record the OOS read BEFORE it happens, so a crash between backtest and
                # collect can never leave an unaccounted holdout read. Gate + append are
                # atomic under the registry lock (concurrent launches/promotes).
                with registry.registry_lock(reg):
                    gates.assert_no_repeat_oos(
                        registry.read_records(reg, on_corrupt="raise"), e["variant_id"]
                    )
                    registry.append_record(
                        reg, _collect_one(cm, e, tier=tier, status="oos_read_attempt")
                    )
                print(f"[research] [{n}/{len(variants)}] backtest (OOS) {e['run_id']} ...")
                subprocess.run(bt, check=True, cwd=str(REPO), env=_oos_env(cm["cohort"], getattr(args, "window_budget", None)))
            registry.append_record(reg, _collect_one(cm, e, tier=tier))
            print(f"[research] [{n}/{len(variants)}] {e['run_id']} done "
                  f"({time.perf_counter() - t0:.0f}s)")
        except subprocess.TimeoutExpired:
            msg = f"wall-clock cap exceeded ({max_h}h, spec budget.max_modal_hours)"
            print(f"[research] ERROR: variant {e['run_id']} timed out: {msg}", file=sys.stderr)
            if backend == "modal":
                print(
                    "[research] WARNING: the timeout killed only the LOCAL modal "
                    "client — the remote app may still be running (and billing). "
                    "Check `modal app list` and stop it manually.",
                    file=sys.stderr,
                )
            registry.append_record(
                reg, _collect_one(cm, e, tier=tier, status="failed", failure=msg)
            )
            failures.append((e["run_id"], msg))
            continue
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


def _evaluate_cohort_gates(cm: dict, records: list[dict]) -> dict:
    """Per-seed-group success_gates verdicts; written to Runs/<cohort>/verdicts.json."""
    success_gates = (cm.get("spec") or {}).get("success_gates") or cm.get("success_gates") or {}
    if not success_gates:
        return {}
    by_group: dict[str, list[dict]] = {}
    for r in records:
        gid = r.get("group_id") or str(r.get("variant_id"))
        by_group.setdefault(str(gid), []).append(r)
    verdicts = {
        gid: gates.evaluate_success_gates(success_gates, rows)
        for gid, rows in sorted(by_group.items())
    }
    out = _cohort_dir(cm["cohort"]) / "verdicts.json"
    out.write_text(json.dumps({"success_gates": dict(success_gates),
                               "verdicts": verdicts}, indent=2), encoding="utf-8")
    for gid, v in verdicts.items():
        print(f"[research] gate verdict {gid}: {v['verdict'].upper()} "
              + ", ".join(f"{k}={c['state']}" for k, c in v["checks"].items()))
    print(f"[research] wrote {out}")
    return verdicts


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
    _evaluate_cohort_gates(cm, registry.read_records(reg))


def cmd_report(args: argparse.Namespace) -> None:
    if getattr(args, "all", False):
        if getattr(args, "cohort", ""):
            raise SystemExit("report takes a cohort id OR --all, not both")
        return cmd_report_all(args)
    if not args.cohort:
        raise SystemExit("report needs a cohort id or --all")
    cohort = args.cohort
    records = registry.read_records(_registry_path(cohort))
    out = _cohort_dir(cohort) / "report.md"
    report.write_report(records, out, title=f"Research cohort: {cohort}")
    print(f"[research] wrote {out} ({len(records)} records)")


def cmd_report_all(args: argparse.Namespace) -> None:
    """Cross-cohort memory: every Runs/*/registry.jsonl in one report, with parent
    lineage and per-knob sensitivity — the input a hypothesis proposer reads."""
    runs_root = PROJECT_ROOT / "Runs"
    cohorts: dict[str, list[dict]] = {}
    cohort_meta: dict[str, dict] = {}
    for reg in sorted(runs_root.glob("*/registry.jsonl")):
        cohort = reg.parent.name
        cohorts[cohort] = registry.read_records(reg)
        cohort_meta[cohort] = _read_json(reg.parent / "cohort.json") or {}
    if not cohorts:
        print(f"[research] no registries under {runs_root}")
        return
    out = report.write_global_report(cohorts, cohort_meta, runs_root / "research_report_all.md")
    n = sum(len(v) for v in cohorts.values())
    print(f"[research] wrote {out} ({len(cohorts)} cohort(s), {n} record(s))")


def cmd_promote(args: argparse.Namespace) -> None:
    """Tier-4 OOS read for a single promoted variant (requires --promote)."""
    spec = load_spec(args.spec)
    if not args.promote:
        raise SystemExit("promote requires --promote (it reads the OOS holdout).")
    # Promote must NOT re-materialize: rewriting cohort.json + variant configs from
    # the *current* spec/config would silently clobber the plan of record the runs
    # actually trained under. Load the launch-time cohort manifest and verify the
    # spec file is unchanged.
    cm = _read_json(_cohort_dir(spec.id) / "cohort.json")
    if not cm:
        raise SystemExit(f"No cohort.json for {spec.id!r}; run `plan`/`launch` first.")
    spec_now = hashlib.sha256(Path(args.spec).read_bytes()).hexdigest()
    spec_then = cm.get("spec_sha256")
    if spec_then and spec_now != spec_then:
        raise SystemExit(
            f"Spec file {args.spec} changed since the cohort was materialized "
            f"(sha {spec_now[:12]} != {spec_then[:12]}). Promoting under an edited "
            "spec would mislabel the result; re-plan as a NEW cohort instead."
        )
    entry = next((e for e in cm["variants"] if e["variant_id"] == args.variant), None)
    if entry is None:
        raise SystemExit(f"variant {args.variant!r} not in cohort {spec.id!r}")
    reg = _registry_path(spec.id)
    promote_tier = max(4, int(cm.get("evaluation_tier", 1)))

    # Pre-registered promotion rule: when the spec declares success_gates, the
    # variant's seed-group must PASS them on in-training evidence before its one
    # holdout read is spent. --force-gates overrides with a loud trail.
    if not ((entry.get("window") or {}).get("holdout_start")):
        raise SystemExit(
            f"variant {entry['variant_id']!r} has no canonical window; tier-4 promotion "
            "requires an explicit W1–W5 window (config-tail holdouts bypass per-window "
            "budgets and can overlap the embargoed W6 range)."
        )
    success_gates = cm.get("success_gates") or {}
    if success_gates:
        rows = [
            r for r in registry.read_records(reg, on_corrupt="raise")
            if (r.get("group_id") or r.get("variant_id")) == (entry.get("group_id") or entry["variant_id"])
        ]
        verdict = gates.evaluate_success_gates(success_gates, rows)
        print(f"[research] promote gate verdict for {entry.get('group_id') or entry['variant_id']}: "
              f"{verdict['verdict'].upper()}")
        if verdict["verdict"] != "pass":
            if not getattr(args, "force_gates", False):
                raise SystemExit(
                    f"Promotion gate verdict is {verdict['verdict']!r} "
                    f"({json.dumps(verdict['checks'], default=str)}). The holdout read "
                    "is spent forever — fix the evidence (more seeds / better eval NAV) "
                    "or pass --force-gates to spend it anyway (recorded)."
                )
            print("[research] WARNING: promoting despite gate verdict "
                  f"{verdict['verdict']!r} (--force-gates).")

    # Cumulative per-window burn budget from the global OOS ledger.
    w = entry.get("window") or {}
    if w.get("holdout_start") and w.get("holdout_end"):
        wkey = oos_ledger.window_key(w["holdout_start"], w["holdout_end"])
        oos_ledger.assert_window_budget(
            oos_ledger.read_ledger(on_corrupt="raise"), wkey, [entry["run_id"]],
            budget=(args.window_budget if getattr(args, "window_budget", None) is not None else oos_ledger.DEFAULT_WINDOW_BUDGET),
        )
    bt = _backtest_cmd(entry)
    if args.dry_run:
        existing = registry.read_records(reg, on_corrupt="raise")
        gates.assert_no_repeat_oos(
            existing, entry["variant_id"], allow_failed_rescore=args.allow_failed_rescore
        )
        print("DRY-RUN promote backtest:", " ".join(bt))
        return
    # Gate + attempt-append are atomic under the registry lock (two concurrent
    # promotes must not both pass), and the gate read fails CLOSED on corruption.
    with registry.registry_lock(reg):
        existing = registry.read_records(reg, on_corrupt="raise")
        gates.assert_no_repeat_oos(
            existing, entry["variant_id"], allow_failed_rescore=args.allow_failed_rescore
        )
        # Record the OOS read BEFORE it happens: a crash below leaves an attempt
        # record, so the no-repeat gate fails closed instead of allowing re-reads.
        registry.append_record(
            reg, _collect_one(cm, entry, tier=promote_tier, status="oos_read_attempt")
        )
    try:
        subprocess.run(bt, check=True, cwd=str(REPO), env=_oos_env(cm["cohort"], getattr(args, "window_budget", None)))
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


def _queue_move(src: Path, dest_dir: Path) -> None:
    """Move a queue spec without silently clobbering a same-named earlier outcome."""
    dest = dest_dir / src.name
    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = dest_dir / f"{src.stem}.{stamp}{src.suffix}"
    shutil.move(str(src), dest)


def cmd_run_queue(args: argparse.Namespace) -> None:
    """Process queued specs (Runs/queue/*.yaml) sequentially: launch → report, then
    move each spec to done/ or failed/. The queue is the substrate an autonomous
    proposer schedules onto; tier ≥ 4 specs are REFUSED here — promotion stays a
    human action (run `promote` directly).

    Single-drainer by design: run ONE run-queue process at a time (no claim locks;
    two drains would double-launch). A spec moved to failed/ may have partially
    succeeded — re-queueing it resumes, since launch skips already-scored variants."""
    qdir = Path(args.queue_dir)
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "done").mkdir(exist_ok=True)
    (qdir / "failed").mkdir(exist_ok=True)
    pending = sorted(p for p in qdir.glob("*.yaml") if p.is_file())
    if not pending:
        print(f"[research] queue empty: {qdir}")
        return
    for spec_path in pending:
        print(f"\n[research] ===== queue: {spec_path.name} =====")
        try:
            spec = load_spec(spec_path)
            if gates.tier_touches_oos(spec.evaluation_tier):
                raise PermissionError(
                    f"tier {spec.evaluation_tier} touches the OOS holdout; the queue "
                    "never promotes. Run `research.py promote` by hand."
                )
            sub = argparse.Namespace(
                spec=str(spec_path), promote=False, dry_run=False,
                oos_budget=1, window_budget=getattr(args, "window_budget", None),
                backend=getattr(args, "backend", "local"),
                modal_gpu=getattr(args, "modal_gpu", None),
            )
            cmd_launch(sub)
            cmd_report(argparse.Namespace(cohort=spec.id))
        except SystemExit as exc:
            # cmd_launch exits non-zero on per-variant failures after finishing the
            # sweep — record and keep draining the queue.
            print(f"[research] queue: {spec_path.name} finished with failures: {exc}",
                  file=sys.stderr)
            _queue_move(spec_path, qdir / "failed")
            continue
        except (PermissionError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            # Machine-written specs fail in machine ways (bad YAML, wrong-typed
            # fields) — a single bad file must not crash-loop the whole drain.
            print(f"[research] queue: {spec_path.name} rejected: {exc}", file=sys.stderr)
            _queue_move(spec_path, qdir / "failed")
            continue
        _queue_move(spec_path, qdir / "done")
        print(f"[research] queue: {spec_path.name} → done/")


def screen_ranking(records: list[dict], keep_top: float) -> tuple[list[tuple[str, float]], list[str]]:
    """Rank seed-groups by median best_eval_nav (descending); return the ranking
    and the advancing top fraction (always at least one group when any ranked)."""
    import statistics as _st

    by_group: dict[str, list[dict]] = {}
    for r in records:
        by_group.setdefault(str(r.get("group_id") or r.get("variant_id")), []).append(r)
    ranked = sorted(
        (
            (gid, _st.median([float(r["best_eval_nav"]) for r in rows
                              if r.get("best_eval_nav") is not None] or [float("-inf")]))
            for gid, rows in by_group.items()
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )
    keep_n = max(1, int(round(len(ranked) * float(keep_top)))) if ranked else 0
    return ranked, [g for g, _ in ranked[:keep_n]]


def cmd_screen(args: argparse.Namespace) -> None:
    """Successive-halving screen: run EVERY grid combo at tier 1 with a tiny budget,
    rank seed-groups by median best_eval_nav, and write screen_ranking.json naming
    the top fraction to advance to a full-tier launch. Never touches the holdout."""
    spec = load_spec(args.spec)
    if not spec.grid:
        raise SystemExit("screen requires a spec with a grid (nothing to halve).")
    cm = _materialize(spec)
    reg = _registry_path(spec.id)
    screen_tier = 1
    ts = int(args.screen_timesteps)
    max_h = (spec.budget or {}).get("max_modal_hours")
    timeout_s = float(max_h) * 3600.0 if max_h else None
    print(f"[research] screening {len(cm['variants'])} variant(s) at tier {screen_tier} "
          f"({ts:,} timesteps each)")
    failures = []
    for n, e in enumerate(cm["variants"], 1):
        # Screen runs live under their own run ids: training the SAME id at a tiny
        # budget would overwrite full-budget artifacts that registry records still
        # point at, and a later `collect` would stamp screen runs at the full tier.
        e = {**e, "run_id": e["run_id"] + "__screen"}
        existing = registry.read_records(reg)
        if (e["run_id"], screen_tier) in _scored_keys(existing):
            print(f"[research] [{n}/{len(cm['variants'])}] {e['run_id']} already screened; skipping.")
            continue
        backend = getattr(args, "backend", "local")
        stale = RunPaths(e["run_id"]).manifest_path.is_file() or backend == "modal"
        train = _train_cmd(
            e, spec, overwrite_run=stale, backend=backend,
            modal_gpu=getattr(args, "modal_gpu", None), timesteps_override=ts,
        )
        print(f"[research] [{n}/{len(cm['variants'])}] screen-train {e['run_id']} ...")
        try:
            if backend == "modal":
                _modal_pre_train(e)
            subprocess.run(train, check=True, cwd=str(REPO), timeout=timeout_s)
            if backend == "modal":
                _modal_post_train(e)
            registry.append_record(reg, _collect_one(cm, e, tier=screen_tier))
        except subprocess.TimeoutExpired:
            msg = f"wall-clock cap exceeded ({max_h}h)"
            registry.append_record(
                reg, _collect_one(cm, e, tier=screen_tier, status="failed", failure=msg)
            )
            failures.append((e["run_id"], msg))
        except subprocess.CalledProcessError as exc:
            msg = f"exit {exc.returncode}"
            registry.append_record(
                reg, _collect_one(cm, e, tier=screen_tier, status="failed", failure=msg)
            )
            failures.append((e["run_id"], msg))
    records = [
        r for r in registry.read_records(reg)
        if int(r.get("evaluation_tier", 0)) == screen_tier
        and str(r.get("status", "ok")) == "ok"
    ]
    ranked, advance = screen_ranking(records, float(args.keep_top))
    keep_n = len(advance)
    out = {
        "screen_timesteps": ts,
        "keep_top": float(args.keep_top),
        "ranking": [{"group_id": g, "median_best_eval_nav": v} for g, v in ranked],
        "advance": advance,
    }
    out_path = _cohort_dir(spec.id) / "screen_ranking.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[research] screen ranking ({len(ranked)} group(s); advancing top {keep_n}):")
    for g, v in ranked:
        marker = "→ ADVANCE" if g in out["advance"] else ""
        print(f"  {v:>14,.0f}  {g}  {marker}")
    print(f"[research] wrote {out_path}")
    print("[research] next: restrict the spec's grid to the advancing combos in a NEW "
          "spec id and launch at the full tier/budget.")
    if failures:
        raise SystemExit(f"{len(failures)} screen variant(s) failed.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("plan"); sp.add_argument("spec"); sp.set_defaults(func=cmd_plan)
    sl = sub.add_parser("launch")
    sl.add_argument("spec")
    sl.add_argument("--backend", default="local", choices=("local", "modal"),
                    help="execution backend: local subprocess or Modal cloud "
                    "(pushes the variant config to the runs volume, trains remotely, "
                    "pulls the run tree back before collect)")
    sl.add_argument("--modal-gpu", default=None,
                    help="GPU profile for --backend modal (e.g. A10G, H100)")
    sl.add_argument("--promote", action="store_true")
    sl.add_argument("--dry-run", action="store_true")
    sl.add_argument(
        "--window-budget", type=int, default=None,
        help="Cumulative distinct-model budget per holdout window (global ledger); "
        "default rlbot.research.oos_ledger.DEFAULT_WINDOW_BUDGET.")
    sl.add_argument(
        "--oos-budget", type=int, default=1,
        help="max holdout reads a tier>=4 launch may perform (default 1; raising this "
             "is an explicit multiple-testing decision)",
    )
    sl.set_defaults(func=cmd_launch)
    sc = sub.add_parser("collect"); sc.add_argument("cohort"); sc.set_defaults(func=cmd_collect, spec=None)
    sq = sub.add_parser("run-queue", help="drain Runs/queue/*.yaml: launch+report each, move to done/failed")
    sq.add_argument("--queue-dir", default=str(PROJECT_ROOT / "Runs" / "queue"))
    sq.add_argument("--backend", default="local", choices=("local", "modal"))
    sq.add_argument("--modal-gpu", default=None)
    sq.add_argument("--window-budget", type=int, default=None)
    sq.set_defaults(func=cmd_run_queue)
    ss = sub.add_parser("screen", help="tier-1 successive-halving screen over a grid spec")
    ss.add_argument("spec")
    ss.add_argument("--screen-timesteps", type=int, default=2_000_000)
    ss.add_argument("--keep-top", type=float, default=0.25)
    ss.add_argument("--backend", default="local", choices=("local", "modal"))
    ss.add_argument("--modal-gpu", default=None)
    ss.set_defaults(func=cmd_screen)
    sr = sub.add_parser("report")
    sr.add_argument("cohort", nargs="?", default="")
    sr.add_argument("--all", action="store_true",
                    help="aggregate every Runs/*/registry.jsonl: lineage + knob sensitivity")
    sr.set_defaults(func=cmd_report)
    spm = sub.add_parser("promote")
    spm.add_argument("spec")
    spm.add_argument("--variant", required=True)
    spm.add_argument("--promote", action="store_true")
    spm.add_argument("--dry-run", action="store_true")
    spm.add_argument(
        "--window-budget", type=int, default=None,
        help="Cumulative distinct-model budget per holdout window (global ledger).")
    spm.add_argument(
        "--force-gates", action="store_true",
        help="Promote even when the pre-registered success_gates verdict is not 'pass' "
        "(spends the holdout read anyway; use deliberately).")
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
