#!/usr/bin/env python3
"""Paper / forward loop for the locked PIT S&P momentum strategy (FINALMODEL).

Computes month-end target weights from prices + PIT membership, places paper
orders on the next session (exec lag = 1), and writes the shadow ledger +
forward mark so ``/ops/forward`` shows the stock book.

Examples::

    # Bootstrap current book + set active forward run
    python scripts/paper_pit_momentum.py run-day --refresh-data

    # Signal / trade for a specific session
    python scripts/paper_pit_momentum.py run-day --as-of 2026-06-30

    # Dry-run (no writes)
    python scripts/paper_pit_momentum.py run-day --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlbot.paper_pit_momentum import (  # noqa: E402
    PAPER_DIR,
    run_paper_day,
)
from rlbot.pit_momentum import PAPER_RUN_ID, STRATEGY_ID  # noqa: E402
from rlbot.forward_mark import (  # noqa: E402
    load_forward_mark,
    resolve_active_forward_run_id,
    set_active_forward_run,
)


def _parse_date(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    return date.fromisoformat(s[:10])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run-day", help="Signal / trade / hold for one session")
    p_run.add_argument("--as-of", default="", help="YYYY-MM-DD (default: last price bar)")
    p_run.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force yfinance refresh of the daily price cache",
    )
    p_run.add_argument(
        "--no-activate",
        action="store_true",
        help="Do not point execution/forward_active.json at FINALMODEL",
    )
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--initial-cash", type=float, default=100_000.0)

    p_status = sub.add_parser("status", help="Show paper state + active forward pointer")
    p_activate = sub.add_parser(
        "activate", help="Point /ops/forward at FINALMODEL (existing mark)"
    )

    args = parser.parse_args()

    if args.cmd == "activate":
        set_active_forward_run(PAPER_RUN_ID)
        print(f"[paper_pit] active forward → {PAPER_RUN_ID}")
        return

    if args.cmd == "status":
        from rlbot.paper_pit_momentum import STATE_PATH, load_state

        st = load_state()
        mark = load_forward_mark(PAPER_RUN_ID)
        tw = st.get("target_weights") or {}
        risky = {k: v for k, v in tw.items() if str(k).upper() != "CASH"}
        print(
            json.dumps(
                {
                    "strategy_id": STRATEGY_ID,
                    "run_id": PAPER_RUN_ID,
                    "active_forward": resolve_active_forward_run_id(),
                    "state_path": str(STATE_PATH),
                    "paper_dir": str(PAPER_DIR),
                    "last_signal_date": st.get("last_signal_date"),
                    "last_trade_date": st.get("last_trade_date"),
                    "equity": st.get("equity"),
                    "cash": st.get("cash"),
                    "n_positions": len(st.get("positions") or {}),
                    "n_target_names": len(risky),
                    "top_weights": dict(
                        sorted(risky.items(), key=lambda kv: -float(kv[1]))[:10]
                    ),
                    "has_forward_mark": mark is not None,
                    "mark_n_bars": (mark or {}).get("n_bars"),
                },
                indent=2,
                default=str,
            )
        )
        return

    # run-day
    result = run_paper_day(
        as_of=_parse_date(args.as_of),
        force_refresh=bool(args.refresh_data),
        set_active=not bool(args.no_activate),
        initial_cash=float(args.initial_cash),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, indent=2, default=str))
    tw = result.get("target_weights") or {}
    risky = [(k, v) for k, v in tw.items() if str(k).upper() != "CASH"]
    risky.sort(key=lambda kv: -float(kv[1]))
    if risky:
        print(f"[paper_pit] book ({len(risky)} names, cash={tw.get('CASH', 0):.1%}):")
        for k, v in risky[:15]:
            print(f"  {k:8s} {float(v):6.2%}")
        if len(risky) > 15:
            print(f"  … +{len(risky) - 15} more")
    print(
        f"[paper_pit] forward: /ops/forward (run_id={result.get('run_id')}) "
        f"actions={result.get('actions')}"
    )


if __name__ == "__main__":
    main()
