#!/usr/bin/env python3
"""Reproduce locked honest PIT momentum (config/honest_alpha.locked.yaml)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_honest_alpha import (  # noqa: E402
    BT_END,
    BT_START,
    OOS_START,
    START_EQ,
    TRAIN_END,
    blend,
    build_cache,
    evaluate,
    run_pit_momentum,
    vol_scale,
    window_slice,
    metrics_eq,
)


def spy_m(spy, dates, start, end):
    i0, i1 = window_slice(dates, start, end)
    s = spy[i0 : i1 + 1]
    eq = START_EQ * (s / s[0]) * (1 - 0.0009)
    return metrics_eq(eq, list(dates[i0 : i1 + 1]), start, end)


def main() -> int:
    lock = json.loads((ROOT / "config" / "honest_alpha.locked.yaml").read_text())
    p = lock["params"]
    data = build_cache(False)
    dates, spy = data["dates"], data["spy"]
    st = spy_m(spy, dates, BT_START, TRAIN_END)
    so = spy_m(spy, dates, OOS_START, BT_END)
    sf = spy_m(spy, dates, BT_START, BT_END)

    base = run_pit_momentum(
        data,
        top_n=int(p["top_n"]),
        lookback=int(p["lookback"]),
        skip=int(p.get("skip", 21)),
        crash_mode=str(p["crash_mode"]),
        exec_lag=int(p.get("exec_lag", 1)),
    )
    ov = str(p.get("overlay", "raw"))
    knob = p.get("knob")
    if ov.startswith("vol") and "_b" in ov:
        vt, w = knob
        eq = blend(vol_scale(base, dates, float(vt)), dates, float(w))
    elif ov.startswith("vol"):
        eq = vol_scale(base, dates, float(knob))
    elif ov.startswith("blend"):
        eq = blend(base, dates, float(knob))
    else:
        eq = base

    row = evaluate(eq, dates, spy, lock["strategy"], p, st, so, sf)
    i0, i1 = window_slice(dates, BT_START, BT_END)
    out = ROOT / "logs" / "honest_alpha_run_curve.csv"
    with open(out, "w") as f:
        f.write("date,equity\n")
        for i in range(i0, i1 + 1):
            f.write(f"{dates[i].isoformat()},{eq[i]:.6f}\n")
    print(json.dumps({"params": p, "metrics": {
        "train_excess": row["train_excess"],
        "oos_excess": row["oos_excess"],
        "full_excess": row["full_excess"],
        "full_dd": row["full_dd"],
        "full_cagr": row["full_cagr"],
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
