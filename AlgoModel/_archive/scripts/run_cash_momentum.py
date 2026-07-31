#!/usr/bin/env python3
"""
Reproduce locked cash long-only momentum (no margin / no shorts).

Reads config/cash_momentum.locked.yaml and writes equity curve + metrics.
Uses /tmp/statarb_mom.db if present (copy of data/statarb.db), else data/statarb.db.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from find_mom_blend_numpy import (  # noqa: E402
    BT_END,
    BT_START,
    OOS_START,
    START_EQ,
    TRAIN_END,
    blend,
    load_cache,
    metrics_from_eq,
    run_base,
    vol_scale,
    window_slice,
)


def spy_metrics(spy, dates, start: date, end: date) -> dict:
    i0, i1 = window_slice(dates, start, end)
    s = spy[i0 : i1 + 1]
    eq = START_EQ * (s / s[0]) * (1 - 0.0009)
    total = float(eq[-1] / eq[0] - 1)
    years = max((end - start).days / 365.25, 1e-6)
    r = np.diff(eq) / eq[:-1]
    sharpe = float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = float(np.min(eq / peak - 1))
    cagr = float((eq[-1] / eq[0]) ** (1 / years) - 1)
    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe_ratio": sharpe,
        "max_drawdown": dd,
    }


def ensure_db_copy() -> None:
    src = ROOT / "data" / "statarb.db"
    dst = Path("/tmp/statarb_mom.db")
    if src.exists() and (not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime):
        print(f"copying {src} -> {dst}", flush=True)
        shutil.copy2(src, dst)


def main() -> int:
    ensure_db_copy()
    lock_path = ROOT / "config" / "cash_momentum.locked.yaml"
    lock = yaml.safe_load(lock_path.read_text())
    bp = lock["base_params"]
    ov = lock["risk_overlay"]

    data = load_cache()
    dates = data["dates"]
    spy = data["spy"]

    base = run_base(
        data,
        top_n=int(bp["top_n"]),
        lookback=int(bp["lookback"]),
        skip=int(bp.get("skip", 21)),
        crash_mode=str(bp["crash_mode"]),
        exec_lag=int(bp.get("exec_lag_days", bp.get("exec_lag", 1))),
    )
    kind = ov.get("kind", "vol")
    if kind in ("vol", "vol_then_blend"):
        eq = vol_scale(
            base,
            dates,
            float(ov.get("vol_target", ov.get("knob", 0.14))),
            lookback=int(ov.get("vol_lookback", 63)),
        )
        if kind == "vol_then_blend":
            eq = blend(eq, dates, float(ov["blend_weight"]))
    elif kind == "blend":
        eq = blend(base, dates, float(ov.get("blend_weight", ov.get("knob", 1.0))))
    else:
        raise SystemExit(f"unknown overlay kind {kind}")

    st = spy_metrics(spy, dates, BT_START, TRAIN_END)
    so = spy_metrics(spy, dates, OOS_START, BT_END)
    sf = spy_metrics(spy, dates, BT_START, BT_END)
    tr = metrics_from_eq(eq, dates, BT_START, TRAIN_END)
    oos = metrics_from_eq(eq, dates, OOS_START, BT_END)
    full = metrics_from_eq(eq, dates, BT_START, BT_END)

    out = {
        "train": {**tr, "excess": tr["total_return"] - st["total_return"], "spy": st},
        "oos": {**oos, "excess": oos["total_return"] - so["total_return"], "spy": so},
        "full": {**full, "excess": full["total_return"] - sf["total_return"], "spy": sf},
        "params": lock,
    }
    i0, i1 = window_slice(dates, BT_START, BT_END)
    curve = ROOT / "logs" / "cash_momentum_winner_curve.csv"
    with open(curve, "w") as f:
        f.write("date,equity\n")
        for i in range(i0, i1 + 1):
            f.write(f"{dates[i].isoformat()},{eq[i]:.6f}\n")
    (ROOT / "logs" / "cash_momentum_run.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(
        f"beats train/oos/full: "
        f"{out['train']['excess']>0}/{out['oos']['excess']>0}/{out['full']['excess']>0} "
        f"dd={full['max_drawdown']:.1%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
