#!/usr/bin/env python3
"""Scan ``Runs/`` and print / write a compact run audit report."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path as _Path

_bootstrap_path = _Path(__file__).resolve().parent / "_bootstrap.py"
_bootstrap_spec = importlib.util.spec_from_file_location("_rlbot_repo_bootstrap", _bootstrap_path)
assert _bootstrap_spec is not None and _bootstrap_spec.loader is not None
_bootstrap_mod = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap_mod)

from pathlib import Path

from rlbot.run_artifacts import PROJECT_ROOT, RUNS_ROOT
from rlbot.run_audit import audit_runs, format_audit_text, write_audit_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", default="", help="Only runs whose id starts with this prefix")
    p.add_argument("--run-ids", default="", help="Comma-separated run ids (default: all under Runs/)")
    p.add_argument("--comparable-only", action="store_true")
    p.add_argument(
        "--json",
        default="",
        help="Write full JSON report (default: Runs/audit_report.json when --write)",
    )
    p.add_argument("--write", action="store_true", help="Write Runs/audit_report.json")
    args = p.parse_args()

    run_ids = [x.strip() for x in args.run_ids.split(",") if x.strip()] or None
    records = audit_runs(run_ids, root=PROJECT_ROOT, prefix=args.prefix)
    print(format_audit_text(records, comparable_only=args.comparable_only))

    out = (args.json or "").strip()
    if args.write or out:
        path = Path(out) if out else RUNS_ROOT / "audit_report.json"
        write_audit_json(records, path)
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
