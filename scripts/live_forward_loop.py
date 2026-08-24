#!/usr/bin/env python3
"""Headless forward collector — 5m marks + paper/shadow logs without the UI.

Yahoo 5m MTM, CoreEquity / CrestDay paper state, and the RLModel shadow
ledger used to refresh only when ``/ops/forward`` was open. This process writes
the same ``execution/`` caches on a timer.

Examples::

    python scripts/live_forward_loop.py --once
    python scripts/live_forward_loop.py --interval 300
    bash scripts/install_live_forward_launchd.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlbot.forward_loop import (  # noqa: E402
    DEFAULT_INTERVAL_S,
    read_status,
    run_loop,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single tick and exit (still takes the collector lock)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_S,
        help=f"Seconds between ticks (default {DEFAULT_INTERVAL_S})",
    )
    parser.add_argument("--run-id", default="", help="Forward mark run id (default: active pointer)")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print execution/forward_loop_status.json and exit",
    )
    parser.add_argument(
        "--no-paper",
        action="store_true",
        help="Skip CoreEquity / CrestDay paper-day ticks",
    )
    parser.add_argument(
        "--no-shadow",
        action="store_true",
        help="Skip the post-close RLModel shadow record",
    )
    parser.add_argument(
        "--no-prices",
        action="store_true",
        help="Skip the Yahoo 5m refresh (parent collector already wrote the mark)",
    )
    parser.add_argument(
        "--skip-lock",
        action="store_true",
        help="Do not take forward_loop.lock (nested tick from the lite collector)",
    )
    args = parser.parse_args()

    if args.status:
        print(json.dumps(read_status(), indent=2, default=str))
        return 0

    result = run_loop(
        interval_s=max(30, int(args.interval)),
        run_id=(args.run_id or "").strip() or None,
        once=bool(args.once),
        # Long-running KeepAlive waits if another collector holds the lock.
        # --once fails fast so a manual tick does not sit behind launchd.
        exit_if_locked=bool(args.once) and not bool(args.skip_lock),
        refresh_prices=not bool(args.no_prices),
        run_paper=not bool(args.no_paper),
        run_rl_shadow=not bool(args.no_shadow),
        take_lock=not bool(args.skip_lock),
    )
    if args.once and result is not None:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
