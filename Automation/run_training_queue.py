#!/usr/bin/env python3
"""Sequential training automation for the three-week ablation campaign.

This runner intentionally does not call backtest.py. It materializes variant
configs from ``hypotheses.yaml`` and trains each variant on W1-W5, one process
at a time, writing logs and queue state under ``Automation/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


AUTOMATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUTOMATION_DIR.parent
DEFAULT_SPEC = AUTOMATION_DIR / "hypotheses.yaml"
GENERATED_CONFIG_DIR = AUTOMATION_DIR / "generated_configs"
LOG_DIR = AUTOMATION_DIR / "logs"
STATE_PATH = AUTOMATION_DIR / "queue_state.jsonl"
QUEUE_PATH = AUTOMATION_DIR / "queue_manifest.json"
LOCK_PATH = AUTOMATION_DIR / "queue.lock"
RUN_ID_MAX_LEN = 96


@dataclass(frozen=True)
class Job:
    ordinal: int
    variant_id: str
    window: int
    seed: int
    run_id: str
    config_path: Path
    patch: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str, *, max_len: int = 64) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    out = re.sub(r"_+", "_", out)
    return out[:max_len].strip("_") or "variant"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if not parts or any(not p for p in parts):
        raise ValueError(f"invalid patch key {dotted!r}")
    cur: dict[str, Any] = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            raise KeyError(f"patch key {dotted!r} cannot descend through {part!r}")
        cur = nxt
    if parts[-1] not in cur:
        raise KeyError(f"patch key {dotted!r} does not exist in base config")
    cur[parts[-1]] = value


def _apply_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        _set_dotted(out, key, value)
    return out


def _complete_run(run_id: str) -> bool:
    manifest = REPO_ROOT / "Runs" / run_id / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("training_status") == "completed"


def _run_exists(run_id: str) -> bool:
    return (REPO_ROOT / "Runs" / run_id / "manifest.json").is_file()


def _run_dir_exists(run_id: str) -> bool:
    return (REPO_ROOT / "Runs" / run_id).exists() or (REPO_ROOT / "runs" / run_id).exists()


def _job_key(variant_id: str, window: int, seed: int, patch: dict[str, Any]) -> str:
    return json.dumps(
        {
            "variant_id": variant_id,
            "window": int(window),
            "seed": int(seed),
            "patch": patch,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _existing_manifest_run_ids() -> dict[str, str]:
    """Reuse previously assigned default-style run ids when rerunning the queue."""
    if not QUEUE_PATH.is_file():
        return {}
    try:
        data = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[str, str] = {}
    for row in data.get("jobs", []):
        try:
            run_id = str(row["run_id"])
            window = int(row["window"])
            if not re.match(rf"^W{window}_\d{{3,4}}(?:_[a-z]|\_\d+)?$", run_id):
                continue
            key = _job_key(
                str(row["variant_id"]),
                window,
                int(row["seed"]),
                dict(row.get("patch") or {}),
            )
            out[key] = run_id
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _allocate_default_run_id(window: int, allocated: set[str]) -> str:
    now = datetime.now(timezone.utc)
    base = f"W{int(window)}_{now.month}{now.day:02d}"
    candidate = base
    suffix = 0
    while candidate in allocated or _run_dir_exists(candidate):
        suffix += 1
        letter = "abcdefghijklmnopqrstuvwxyz"[suffix - 1] if suffix <= 26 else str(suffix)
        candidate = f"{base}_{letter}"
    allocated.add(candidate)
    return candidate


def _latest_checkpoint(run_id: str) -> Path | None:
    ckpt_dir = REPO_ROOT / "Runs" / run_id / "models" / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    pairs: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("ppo_*_steps.zip"):
        m = re.search(r"ppo_(\d+)_steps\.zip$", path.name)
        if m:
            pairs.append((int(m.group(1)), path))
    if not pairs:
        return None
    return max(pairs, key=lambda x: x[0])[1]


def _load_state() -> dict[str, str]:
    state: dict[str, str] = {}
    if not STATE_PATH.is_file():
        return state
    with STATE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = rec.get("run_id")
            status = rec.get("status")
            if run_id and status:
                state[str(run_id)] = str(status)
    return state


def _append_state(job: Job, status: str, **extra: Any) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "time_utc": _utc_now(),
        "status": status,
        "ordinal": job.ordinal,
        "variant_id": job.variant_id,
        "window": job.window,
        "seed": job.seed,
        "run_id": job.run_id,
        "config_path": str(job.config_path.relative_to(REPO_ROOT)),
        **extra,
    }
    with STATE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def _materialize_jobs(spec_path: Path) -> list[Job]:
    spec = _load_yaml(spec_path)
    base_config_path = REPO_ROOT / str(spec.get("base_config", "config/config.yaml"))
    base = _load_yaml(base_config_path)
    windows = [int(w) for w in spec.get("windows", [1, 2, 3, 4, 5])]
    variants = spec.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("hypotheses.yaml must define a non-empty variants list")

    jobs: list[Job] = []
    manifest_rows: list[dict[str, Any]] = []
    ordinal = 0
    previous_run_ids = _existing_manifest_run_ids()
    allocated_run_ids = set(previous_run_ids.values())
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError(f"variant must be a mapping: {variant!r}")
        variant_id = _slug(variant["id"], max_len=56)
        seed = int(variant.get("seed", base.get("training", {}).get("seed", 0)))
        patch = dict(variant.get("patch") or {})
        patched = _apply_patch(base, patch)
        patched.setdefault("training", {})["seed"] = seed
        cfg_path = GENERATED_CONFIG_DIR / f"{variant_id}.yaml"
        _write_yaml(cfg_path, patched)

        for window in windows:
            ordinal += 1
            key = _job_key(variant_id, window, seed, patch)
            run_id = previous_run_ids.get(key)
            if run_id is None:
                run_id = _allocate_default_run_id(window, allocated_run_ids)
            else:
                allocated_run_ids.add(run_id)
            job = Job(
                ordinal=ordinal,
                variant_id=variant_id,
                window=window,
                seed=seed,
                run_id=run_id,
                config_path=cfg_path,
                patch=patch,
            )
            jobs.append(job)
            manifest_rows.append(
                {
                    "ordinal": ordinal,
                    "run_id": run_id,
                    "variant_id": variant_id,
                    "window": window,
                    "seed": seed,
                    "config_path": str(cfg_path.relative_to(REPO_ROOT)),
                    "patch": patch,
                }
            )

    QUEUE_PATH.write_text(
        json.dumps(
            {
                "generated_at_utc": _utc_now(),
                "spec_path": str(spec_path.relative_to(REPO_ROOT)),
                "base_config": str(base_config_path.relative_to(REPO_ROOT)),
                "n_jobs": len(manifest_rows),
                "jobs": manifest_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return jobs


def _train_command(job: Job, python: str, extra_args: list[str], *, resume: Path | None) -> list[str]:
    cmd = [
        python,
        "scripts/train.py",
        "--config",
        str(job.config_path),
        "--window",
        str(job.window),
        "--run-id",
        job.run_id,
        "--seed",
        str(job.seed),
    ]
    if resume is not None:
        cmd.extend(["--resume", str(resume)])
    cmd.extend(extra_args)
    return cmd


def _acquire_lock(force: bool) -> None:
    if LOCK_PATH.exists() and not force:
        raise SystemExit(
            f"{LOCK_PATH} exists. Another automation run may be active. "
            "Remove it or pass --force-lock if you are sure."
        )
    LOCK_PATH.write_text(
        json.dumps({"pid": os.getpid(), "started_at_utc": _utc_now()}) + "\n",
        encoding="utf-8",
    )


def _release_lock() -> None:
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def run_queue(args: argparse.Namespace) -> int:
    jobs = _materialize_jobs(Path(args.spec).resolve())
    if args.limit is not None:
        jobs = jobs[: int(args.limit)]
    if args.only_window is not None:
        windows = {int(w) for w in args.only_window}
        jobs = [j for j in jobs if j.window in windows]
    if args.only_variant:
        allowed = {_slug(v, max_len=56) for v in args.only_variant}
        jobs = [j for j in jobs if j.variant_id in allowed]

    extra_args: list[str] = []
    if args.no_viz:
        extra_args.append("--no-viz")
    if args.refresh_data:
        extra_args.append("--refresh-data")
    if args.timesteps is not None:
        extra_args.extend(["--timesteps", str(int(args.timesteps))])

    print(f"[automation] jobs selected: {len(jobs)}")
    print(f"[automation] queue manifest: {QUEUE_PATH.relative_to(REPO_ROOT)}")
    if args.dry_run:
        for job in jobs:
            resume = _latest_checkpoint(job.run_id) if args.resume_incomplete else None
            print(" ".join(_train_command(job, args.python, extra_args, resume=resume)))
        return 0

    _acquire_lock(force=bool(args.force_lock))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for job in jobs:
            if _complete_run(job.run_id):
                print(f"[automation] skip completed {job.run_id}")
                _append_state(job, "skipped_completed")
                continue
            resume: Path | None = None
            if _run_exists(job.run_id):
                if not args.resume_incomplete:
                    print(f"[automation] skip existing incomplete {job.run_id}")
                    _append_state(job, "skipped_existing_incomplete")
                    continue
                resume = _latest_checkpoint(job.run_id)
                if resume is None:
                    print(f"[automation] skip incomplete without checkpoint {job.run_id}")
                    _append_state(job, "skipped_no_checkpoint")
                    continue

            log_path = LOG_DIR / f"{job.ordinal:03d}_{job.run_id}.log"
            cmd = _train_command(job, args.python, extra_args, resume=resume)
            print(f"[automation] start {job.ordinal}/{len(jobs)} {job.run_id}")
            print(f"[automation] log: {log_path.relative_to(REPO_ROOT)}")
            _append_state(job, "started", command=cmd, log_path=str(log_path.relative_to(REPO_ROOT)))
            started = time.time()
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[automation] {_utc_now()} command: {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - started
            status = "completed" if proc.returncode == 0 and _complete_run(job.run_id) else "failed"
            _append_state(job, status, returncode=proc.returncode, elapsed_s=elapsed)
            print(
                f"[automation] {status} {job.run_id} "
                f"returncode={proc.returncode} elapsed_h={elapsed / 3600:.2f}"
            )
            if status == "failed" and not args.keep_going:
                return proc.returncode or 1
    finally:
        _release_lock()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC), help="Hypothesis YAML path")
    parser.add_argument("--python", default=sys.executable, help="Python executable for scripts/train.py")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected jobs")
    parser.add_argument("--only-window", type=int, action="append", help="Restrict to one window; repeatable")
    parser.add_argument("--only-variant", action="append", help="Restrict to one variant id; repeatable")
    parser.add_argument("--timesteps", type=int, default=None, help="Override training timesteps for smoke tests")
    parser.add_argument("--no-viz", action="store_true", help="Pass --no-viz to train.py")
    parser.add_argument("--refresh-data", action="store_true", help="Pass --refresh-data to every train.py call")
    parser.add_argument("--resume-incomplete", action="store_true", help="Resume existing incomplete runs from latest checkpoint")
    parser.add_argument("--keep-going", action="store_true", help="Continue after failed jobs")
    parser.add_argument("--force-lock", action="store_true", help="Overwrite stale Automation/queue.lock")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_queue(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
