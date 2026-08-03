#!/usr/bin/env python3
"""Paper / forward companion for CrestDay.

Usage:
    python scripts/paper_crest_day.py run-day
    python scripts/paper_crest_day.py run-day --refresh-data
    python scripts/paper_crest_day.py status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlbot.pack_crestday import PAPER_RUN_ID, STRATEGY_ID  # noqa: E402
from rlbot.paper_crest_day import (  # noqa: E402
    ledger_path,
    load_state,
    run_paper_day,
)
from rlbot.forward_mark import set_active_forward_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="CrestDay paper / forward companion")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run-day", help="Run pack NAV + live intents → mark")
    run_p.add_argument("--refresh-data", action="store_true", help="Bypass NAV cache")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument(
        "--set-active",
        action="store_true",
        help="Make CREST_DAY the active forward pointer (usually leave off)",
    )
    run_p.add_argument("--aum", type=float, default=100_000.0)

    sub.add_parser("status", help="Show paper state / ledger tip")
    act = sub.add_parser("activate", help="Point /ops/forward at CREST_DAY (rare)")
    del act

    args = ap.parse_args()
    if args.cmd == "run-day":
        out = run_paper_day(
            force_refresh=bool(args.refresh_data),
            set_active=bool(args.set_active),
            initial_cash=float(args.aum),
            dry_run=bool(args.dry_run),
        )
        print(json.dumps(out, indent=2, default=str))
        return 0
    if args.cmd == "activate":
        set_active_forward_run(PAPER_RUN_ID)
        print(json.dumps({"active": PAPER_RUN_ID}, indent=2))
        return 0
    if args.cmd == "status":
        st = load_state()
        tip = None
        path = ledger_path()
        if path.is_file():
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                tip = json.loads(lines[-1])
        print(
            json.dumps(
                {
                    "strategy_id": STRATEGY_ID,
                    "run_id": PAPER_RUN_ID,
                    "state": st,
                    "ledger_tip": tip,
                },
                indent=2,
                default=str,
            )
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
