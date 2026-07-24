#!/usr/bin/env python3
"""Print curriculum / schedule milestones before launching a training job."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path as _Path

_bootstrap_path = _Path(__file__).resolve().parent / "_bootstrap.py"
_bootstrap_spec = importlib.util.spec_from_file_location("_rlbot_repo_bootstrap", _bootstrap_path)
assert _bootstrap_spec is not None and _bootstrap_spec.loader is not None
_bootstrap_mod = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap_mod)

import json
from pathlib import Path

from rlbot.curriculum_preflight import build_curriculum_preflight, format_preflight_text
from rlbot.rl_config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="", help="Config YAML (default: config/config.yaml)")
    p.add_argument("--budget", type=int, default=0, help="Override training.timesteps")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 when schedule warnings are present (e.g. early-stop unreachable)",
    )
    args = p.parse_args()

    cfg_path = Path(args.config) if args.config.strip() else None
    cfg = load_config(cfg_path)
    budget = int(args.budget) if args.budget > 0 else None
    pf = build_curriculum_preflight(cfg, budget=budget)
    if args.json:
        print(json.dumps(pf.to_dict(), indent=2))
    else:
        print(format_preflight_text(pf))
    if pf.warnings and args.strict:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
