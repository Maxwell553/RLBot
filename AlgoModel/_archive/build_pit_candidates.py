#!/usr/bin/env python3
"""Build point-in-time pair candidate schedules (no full-sample durable leak)."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    edge = ROOT / "logs" / "walkforward_coint_edge_20260729_205814.json"
    d = json.loads(edge.read_text())
    cycles = d["cycles"]

    # Map trade_start -> pairs selected that cycle (formation used only past data)
    rows = []
    seen_before: dict[str, list[str]] = {}
    counts: dict[str, int] = defaultdict(int)
    all_prior: list[tuple[str, str]] = []

    train_end = date(2017, 12, 29)
    train_frozen: set[tuple[str, str]] = set()

    for c in cycles:
        ts = date.fromisoformat(c["trade_start"])
        # Candidates available AT ts = pairs selected in strictly earlier cycles
        cands = sorted(set(all_prior))
        seen_before[str(ts)] = [f"{a}-{b}" for a, b in cands]
        for s in c.get("selected") or []:
            key = s["key"]
            a, b = key.split("-", 1)
            pair = (a, b)
            counts[key] += 1
            all_prior.append(pair)
            if ts <= train_end:
                train_frozen.add(pair)
        rows.append(
            {
                "trade_start": str(ts),
                "trade_end": c["trade_end"],
                "n_pass": c["n_pass"],
                "n_pit_candidates": len(cands),
                "selected": ",".join(s["key"] for s in c.get("selected") or []),
            }
        )

    out_dir = ROOT / "logs"
    pd.DataFrame(rows).to_csv(out_dir / "pit_candidate_schedule.csv", index=False)

    # Flatten schedule: for each trade_start, list of candidate pairs
    sched = []
    for ts, keys in seen_before.items():
        for key in keys:
            a, b = key.split("-", 1)
            sched.append({"asof": ts, "symbol_a": a, "symbol_b": b, "pair": key})
    pd.DataFrame(sched).to_csv(out_dir / "pit_candidates_long.csv", index=False)

    frozen = pd.DataFrame(
        [
            {"pair": f"{a}-{b}", "symbol_a": a, "symbol_b": b, "sector": ""}
            for a, b in sorted(train_frozen)
        ]
    )
    frozen.to_csv(out_dir / "train_frozen_selected_universe.csv", index=False)

    summary = {
        "source": str(edge.name),
        "n_cycles": len(cycles),
        "train_frozen_n": len(train_frozen),
        "note": (
            "PIT candidates at T = union of walk-forward SELECTED pairs from cycles "
            "with trade_start < T. Not the full-sample durable≥3 list (that leaked)."
        ),
    }
    (out_dir / "pit_candidates_meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out_dir / "pit_candidates_long.csv")
    print("wrote", out_dir / "train_frozen_selected_universe.csv", "n=", len(frozen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
