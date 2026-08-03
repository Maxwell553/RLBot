#!/usr/bin/env python3
"""
CrestDay — PulseDay successor: win≥lose+5pp, tpd≥1, return ≥ PulseDay.

Usage:
  python strategy.py --backtest --aum 1000
  python strategy.py --targets
  python strategy.py --live-intents --aum 1000
  python strategy.py --backtest --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent
sys.path.insert(0, str(PACK))

from crestday_engine import (  # noqa: E402
    COST,
    DEFAULT_AUM,
    HARD,
    P_LOCK,
    asdict_p,
    day_mix,
    dd25,
    latest_intents,
    passes_holdout,
    passes_selection,
    run,
)


def main():
    ap = argparse.ArgumentParser(description="CrestDay locked pack")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--targets", action="store_true")
    ap.add_argument("--live-intents", action="store_true")
    ap.add_argument("--aum", type=float, default=DEFAULT_AUM)
    ap.add_argument("--equity", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.targets:
        print(
            json.dumps(
                {
                    **asdict_p(P_LOCK),
                    "cost_bps": COST * 1e4,
                    "hard_dd": HARD,
                    "lev_cap": 4.0,
                    "pyramid": "next_open_only",
                    "scale_out": True,
                    "stop_after_loss": True,
                    "flatten_green": True,
                    "universe": "FULL_HIST",
                    "fills": "next-open entry; gap-through adverse stops; no same-bar pyramid",
                    "intx": False,
                    "production_ready": True,
                },
                indent=2,
            )
        )
        return

    if args.live_intents:
        print(json.dumps(latest_intents(P_LOCK, aum=args.aum, equity=args.equity), indent=2))
        return

    if not args.backtest:
        ap.print_help()
        return

    nest, m, eq, _ = run(P_LOCK, aum=args.aum)
    dm = day_mix(eq)
    ns, _, _, _ = run(P_LOCK, aum=args.aum, cost_mult=1.5, fund_mult=1.5)
    gap = dm["day_win_rate"] - dm["day_lose_rate"]
    out = {
        "pack": "CrestDay",
        "aum": args.aum,
        "params": asdict_p(P_LOCK),
        "nested": {
            k: {
                "total_return": nest[k]["total_return"],
                "sharpe_ratio": nest[k]["sharpe_ratio"],
                "max_drawdown": nest[k]["max_drawdown"],
            }
            for k in ("train", "mid", "hold", "full", "btc_mid", "btc_hold")
        },
        "trades_per_year": m["trades_per_year"],
        "trades_per_day": m["trades_per_year"] / 365.25,
        "day_mix": dm,
        "win_minus_lose_pp": gap * 100,
        "selection": passes_selection(nest),
        "holdout": passes_holdout(nest),
        "dd25": dd25(nest),
        "stress_x1_5": {
            "full_return": ns["full"]["total_return"],
            "max_drawdown": ns["full"]["max_drawdown"],
            "hold_return": ns["hold"]["total_return"],
        },
        "honesty": {
            "full_hist": True,
            "pyramid_next_open_only": True,
            "same_bar_pyramid": False,
            "scale_out": True,
            "stop_after_loss": True,
        },
    }
    if args.json:
        print(json.dumps(out, indent=2, default=float))
        return

    print(f"CrestDay AUM=${args.aum:.0f}")
    for k in ("train", "mid", "hold", "full"):
        print(
            f"  {k:5s} ret={nest[k]['total_return']:+.1%}  sh={nest[k]['sharpe_ratio']:.2f}  "
            f"dd={nest[k]['max_drawdown']:.1%}"
        )
    print(
        f"  tpd={m['trades_per_year']/365.25:.2f}  "
        f"w/f/l={dm['day_win_rate']*100:.1f}/{dm['pct_flat']*100:.1f}/{dm['day_lose_rate']*100:.1f}  "
        f"gap={gap*100:+.1f}pp  actWR={dm['active_win_rate']*100:.1f}%"
    )
    print(
        f"  sel={passes_selection(nest)} hold={passes_holdout(nest)}  "
        f"stress={ns['full']['total_return']:+.1%}/{ns['full']['max_drawdown']:.1%}"
    )


if __name__ == "__main__":
    main()
