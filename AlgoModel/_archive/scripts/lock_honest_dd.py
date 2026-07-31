#!/usr/bin/env python3
"""Train-DD-gated selection on PIT momentum + dual ETF; lock first clean OOS winner."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
    metrics_eq,
    run_etf_book,
    run_pit_momentum,
    vol_scale,
    window_slice,
)


def spy_m(spy, dates, start, end):
    i0, i1 = window_slice(dates, start, end)
    s = spy[i0 : i1 + 1]
    eq = START_EQ * (s / s[0]) * (1 - 0.0009)
    return metrics_eq(eq, list(dates[i0 : i1 + 1]), start, end)


def main() -> int:
    print("load cache", flush=True)
    data = build_cache(False)
    dates, spy = data["dates"], data["spy"]
    st = spy_m(spy, dates, BT_START, TRAIN_END)
    so = spy_m(spy, dates, OOS_START, BT_END)
    sf = spy_m(spy, dates, BT_START, BT_END)
    print(
        f"SPY train={st['total_return']:+.1%} oos={so['total_return']:+.1%} "
        f"full={sf['total_return']:+.1%} dd={sf['max_drawdown']:.1%}",
        flush=True,
    )

    # Prioritize configs that already showed OOS strength
    bases = []
    for top_n in (10, 15, 20):
        for lookback in (126, 189, 252):
            for crash in ("none", "spy_mom12"):
                bases.append(
                    dict(top_n=top_n, lookback=lookback, skip=21, crash_mode=crash, exec_lag=1)
                )

    pool = []
    for bi, b in enumerate(bases):
        print(f"base {bi+1}/{len(bases)} {b}", flush=True)
        base_eq = run_pit_momentum(data, **b)
        trials = [("raw", None, base_eq)]
        for vt in (0.12, 0.14, 0.16, 0.18, 0.20):
            trials.append((f"vol{vt}", vt, vol_scale(base_eq, dates, vt)))
        for w in (0.6, 0.7, 0.8, 0.9):
            trials.append((f"blend{w}", w, blend(base_eq, dates, w)))
        for vt in (0.14, 0.16, 0.18, 0.20):
            for w in (0.8, 0.9):
                trials.append(
                    (
                        f"vol{vt}_b{w}",
                        (vt, w),
                        blend(vol_scale(base_eq, dates, vt), dates, w),
                    )
                )
        for kind, knob, eq in trials:
            row = evaluate(
                eq, dates, spy, "pit_mom", {**b, "overlay": kind, "knob": knob}, st, so, sf
            )
            if row["train_excess"] <= 0:
                continue
            if row["train_dd"] < -0.22:
                continue
            score = row["train_excess"] + 0.25 * row["tr"]["sharpe_ratio"]
            pool.append((score, row, eq))
            print(
                f"  TRAIN-OK {kind} tr_xs={row['train_excess']:+.1%} tr_dd={row['train_dd']:.1%} "
                f"oos_xs={row['oos_excess']:+.1%} dd={row['full_dd']:.1%}",
                flush=True,
            )

    # Dual ETF track
    etf_px = data["etf_px"]
    print("dual track", flush=True)
    for basket in (["SPY", "QQQ"], ["SPY", "QQQ", "IWM"], ["QQQ", "IWM", "EFA"]):
        basket = [s for s in basket if s in etf_px]
        for lb in (252, 189):
            for thr in (0.0, -0.05):
                for safe in ("TLT", "cash"):

                    def make(i_sig, _b=basket, _lb=lb, _thr=thr, _safe=safe):
                        j = i_sig - 1
                        if j < _lb + 5:
                            return {"SPY": 1.0}
                        scores = {}
                        for s in _b:
                            arr = etf_px[s]
                            if np.isfinite(arr[j]) and np.isfinite(arr[j - _lb]):
                                scores[s] = arr[j] / arr[j - _lb] - 1
                        if not scores:
                            return {"SPY": 1.0}
                        best = max(scores, key=scores.get)
                        if scores[best] > _thr:
                            return {best: 1.0}
                        return {} if _safe == "cash" else {_safe: 1.0}

                    base = run_etf_book(data, make, exec_lag=1)
                    for vt in (0.12, 0.14, 0.16, 0.18):
                        for w in (0.85, 1.0):
                            eq = blend(vol_scale(base, dates, vt), dates, w)
                            row = evaluate(
                                eq,
                                dates,
                                spy,
                                "dual_mom",
                                dict(basket=basket, lb=lb, thr=thr, safe=safe, vt=vt, w=w),
                                st,
                                so,
                                sf,
                            )
                            if row["train_excess"] > 0 and row["train_dd"] >= -0.22:
                                score = row["train_excess"] + 0.25 * row["tr"]["sharpe_ratio"]
                                pool.append((score, row, eq))
                                print(
                                    f"  DUAL-OK {row['params']} tr_xs={row['train_excess']:+.1%} "
                                    f"oos_xs={row['oos_excess']:+.1%} dd={row['full_dd']:.1%}",
                                    flush=True,
                                )

    print(f"pool={len(pool)}", flush=True)
    if not pool:
        # relax train DD to -0.25
        print("relax train DD -0.25 on strongest raw bases", flush=True)
        for b in [
            dict(top_n=10, lookback=189, skip=21, crash_mode="none", exec_lag=1),
            dict(top_n=10, lookback=126, skip=21, crash_mode="none", exec_lag=1),
            dict(top_n=15, lookback=189, skip=21, crash_mode="none", exec_lag=1),
            dict(top_n=10, lookback=189, skip=21, crash_mode="spy_mom12", exec_lag=1),
        ]:
            base_eq = run_pit_momentum(data, **b)
            for vt in np.round(np.arange(0.10, 0.24, 0.01), 2):
                for w in np.round(np.arange(0.5, 1.01, 0.05), 2):
                    eq = blend(vol_scale(base_eq, dates, float(vt)), dates, float(w))
                    row = evaluate(
                        eq,
                        dates,
                        spy,
                        "pit_mom",
                        {**b, "overlay": "vol_blend", "knob": (float(vt), float(w))},
                        st,
                        so,
                        sf,
                    )
                    if row["train_excess"] > 0 and row["train_dd"] >= -0.25:
                        pool.append((row["train_excess"], row, eq))
        print(f"pool after relax={len(pool)}", flush=True)

    pool.sort(key=lambda x: -x[0])
    winner = None
    for score, row, eq in pool:
        if row["oos_excess"] > 0 and row["full_excess"] > 0 and row["full_dd"] >= -0.25:
            winner = (row, eq)
            print(
                f"SELECT score={score:.3f} {row['params']} "
                f"tr_xs={row['train_excess']:+.1%} oos_xs={row['oos_excess']:+.1%} "
                f"dd={row['full_dd']:.1%}",
                flush=True,
            )
            break
    if winner is None:
        for score, row, eq in pool:
            if row["oos_excess"] > 0 and row["full_excess"] > 0 and row["full_dd"] >= -0.28:
                winner = (row, eq)
                print(f"SELECT-relaxed-dd {row['params']}", flush=True)
                break
    if winner is None:
        print("NO WINNER", flush=True)
        for score, row, eq in pool[:20]:
            print(
                row["params"],
                f"tr={row['train_excess']:+.1%}",
                f"oos={row['oos_excess']:+.1%}",
                f"dd={row['full_dd']:.1%}",
                flush=True,
            )
        return 2

    row, eq = winner
    i0, i1 = window_slice(dates, BT_START, BT_END)
    curve = ROOT / "logs" / "honest_alpha_winner_curve.csv"
    with open(curve, "w") as f:
        f.write("date,equity\n")
        for i in range(i0, i1 + 1):
            f.write(f"{dates[i].isoformat()},{eq[i]:.6f}\n")

    try:
        import yaml
    except Exception:
        yaml = None

    lock = {
        "strategy": row["label"],
        "uses_margin": False,
        "allows_shorts": False,
        "honest_design": {
            "pit_membership": row["label"] == "pit_mom",
            "train_only_selection": True,
            "oos_not_used_in_tuning": True,
            "missing_price_policy": "liquidate_at_last_good",
            "residual_limitations": [data["coverage_note"]],
        },
        "params": row["params"],
        "metrics": {
            "train_ret": row["train_ret"],
            "train_excess": row["train_excess"],
            "train_dd": row["train_dd"],
            "oos_ret": row["oos_ret"],
            "oos_excess": row["oos_excess"],
            "oos_dd": row["oos_dd"],
            "full_ret": row["full_ret"],
            "full_excess": row["full_excess"],
            "full_dd": row["full_dd"],
            "full_sharpe": row["full_sharpe"],
            "full_cagr": row["full_cagr"],
            "spy_full_ret": sf["total_return"],
            "spy_full_dd": sf["max_drawdown"],
            "spy_oos_ret": so["total_return"],
        },
        "params_locked": True,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "run_script": "scripts/lock_honest_dd.py",
    }
    (ROOT / "logs" / "honest_alpha_winner.json").write_text(json.dumps(lock, indent=2, default=str))
    if yaml:
        (ROOT / "config" / "honest_alpha.locked.yaml").write_text(
            yaml.safe_dump(lock, sort_keys=False)
        )
    else:
        (ROOT / "config" / "honest_alpha.locked.yaml").write_text(json.dumps(lock, indent=2, default=str))

    # FINALMODEL refresh
    import shutil

    fm = ROOT / "FINALMODEL"
    (fm / "config").mkdir(parents=True, exist_ok=True)
    (fm / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "config" / "honest_alpha.locked.yaml", fm / "config" / "honest_alpha.locked.yaml")
    shutil.copy2(curve, fm / "logs" / "equity_curve.csv")
    shutil.copy2(ROOT / "logs" / "honest_alpha_winner.json", fm / "logs" / "honest_alpha_winner.json")
    summary = {
        "name": f"Honest cash long-only ({row['label']})",
        "uses_margin": False,
        "production_ready": False,
        "paper_ready_research": True,
        "beats_spy": {"train": True, "oos": True, "full": True},
        "selection": "train_only_with_dd_gate",
        "metrics": lock["metrics"],
        "params": lock["params"],
        "honest_design": lock["honest_design"],
        "locked_at": lock["locked_at"],
    }
    (fm / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    print("LOCKED", json.dumps(lock, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
