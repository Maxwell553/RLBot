#!/usr/bin/env python3
"""
Move legacy per-run artifacts into ``Runs/<run_id>/``.

Consolidates:
  runs/<id>/     → Runs/<id>/          (manifest, config, eval_logs, data_cache.npz)
  models/<id>/   → Runs/<id>/models/
  plots/<id>/    → Runs/<id>/plots/
  logs/<id>/     → Runs/<id>/logs/
  tb_logs/<id>/  → Runs/<id>/tb_logs/

Safe to re-run: skips destinations that already exist (prints a warning).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rlbot.run_artifacts import (
    PROJECT_ROOT,
    RUNS_ROOT,
    _LEGACY_RUNS_META_ROOT,
    RunPaths,
)


def _collect_run_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for meta_root in (_LEGACY_RUNS_META_ROOT, root / "models"):
        if not meta_root.is_dir():
            continue
        for p in meta_root.iterdir():
            if p.is_dir():
                ids.add(p.name)
    if RUNS_ROOT.is_dir():
        for p in RUNS_ROOT.iterdir():
            if p.is_dir():
                ids.add(p.name)
    return ids


def _move_dir(src: Path, dest: Path, *, dry_run: bool) -> None:
    if not src.is_dir():
        return
    if dest.exists():
        print(f"  skip (exists): {dest}")
        return
    print(f"  {src} → {dest}")
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))


def _move_file(src: Path, dest: Path, *, dry_run: bool) -> None:
    if not src.is_file():
        return
    if dest.exists():
        print(f"  skip (exists): {dest}")
        return
    print(f"  {src} → {dest}")
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))


def migrate_run(run_id: str, *, dry_run: bool) -> None:
    paths = RunPaths(run_id)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    legacy_meta = PROJECT_ROOT / "runs" / run_id
    if legacy_meta.is_dir():
        for name in ("manifest.json", "config.yaml", "data_cache.npz"):
            _move_file(legacy_meta / name, paths.run_dir / name, dry_run=dry_run)
        _move_dir(legacy_meta / "eval_logs", paths.run_dir / "eval_logs", dry_run=dry_run)
        if legacy_meta.is_dir() and not any(legacy_meta.iterdir()):
            print(f"  remove empty: {legacy_meta}")
            if not dry_run:
                legacy_meta.rmdir()

    _move_dir(PROJECT_ROOT / "models" / run_id, paths.run_dir / "models", dry_run=dry_run)
    _move_dir(PROJECT_ROOT / "plots" / run_id, paths.run_dir / "plots", dry_run=dry_run)
    _move_dir(PROJECT_ROOT / "logs" / run_id, paths.run_dir / "logs", dry_run=dry_run)
    _move_dir(PROJECT_ROOT / "tb_logs" / run_id, paths.run_dir / "tb_logs", dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="",
        help="Migrate one run id only (default: all discovered ids).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print moves without applying.")
    args = parser.parse_args()

    if args.run_id.strip():
        run_ids = [args.run_id.strip()]
    else:
        run_ids = sorted(_collect_run_ids(PROJECT_ROOT))

    if not run_ids:
        print("No run ids found under runs/, models/, or Runs/.")
        return

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    for rid in run_ids:
        print(f"\n=== {rid} ===")
        migrate_run(rid, dry_run=args.dry_run)

    print("\nDone. New training runs write only under Runs/<run_id>/.")


if __name__ == "__main__":
    main()
