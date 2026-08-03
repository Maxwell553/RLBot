#!/usr/bin/env python3
"""Paper / forward loop for GeneralEquity1 (prod_return_alpha_v3 pack).

Examples::

    python scripts/paper_prod_return_alpha.py run-day --refresh-data
    python scripts/paper_prod_return_alpha.py run-day --as-of 2026-06-30
    python scripts/paper_prod_return_alpha.py run-day --dry-run
    python scripts/paper_prod_return_alpha.py activate
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlbot.forward_mark import (  # noqa: E402
    load_forward_mark,
    resolve_active_forward_run_id,
    set_active_forward_run,
)
from rlbot.paper_prod_return_alpha import (  # noqa: E402
    PAPER_DIR,
    STATE_PATH,
    load_state,
    run_paper_day,
)
from rlbot.pack_general_equity1 import PAPER_RUN_ID, STRATEGY_ID  # noqa: E402


def _parse_date(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    return date.fromisoformat(s[:10])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run-day", help="Signal / rebalance / hold for one session")
    p_run.add_argument("--as-of", default="", help="YYYY-MM-DD (default: last price bar)")
    p_run.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force yfinance refresh of the daily OHLC cache",
    )
    p_run.add_argument(
        "--no-activate",
        action="store_true",
        help="Do not point execution/forward_active.json at GENERAL_EQUITY1",
    )
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--initial-cash", type=float, default=100_000.0)

    sub.add_parser("status", help="Show paper state + active forward pointer")
    sub.add_parser(
        "activate", help="Point /ops/forward at GENERAL_EQUITY1 (existing mark)"
    )

    args = parser.parse_args()

    if args.cmd == "activate":
        set_active_forward_run(PAPER_RUN_ID)
        print(f"[paper_ge] active forward → {PAPER_RUN_ID}")
        return

    if args.cmd == "status":
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
                    "flat_a": st.get("flat_a"),
                    "n_positions": len(st.get("positions") or {}),
                    "target_weights": dict(
                        sorted(risky.items(), key=lambda kv: -float(kv[1]))
                    ),
                    "has_forward_mark": mark is not None,
                    "mark_n_bars": (mark or {}).get("n_bars"),
                },
                indent=2,
                default=str,
            )
        )
        return

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
        print(f"[paper_ge] book (cash={tw.get('CASH', 0):.1%}):")
        for k, v in risky:
            print(f"  {k:8s} {float(v):6.2%}")
    print(
        f"[paper_ge] forward: /ops/forward (run_id={result.get('run_id')}) "
        f"actions={result.get('actions')}"
    )


if __name__ == "__main__":
    main()
