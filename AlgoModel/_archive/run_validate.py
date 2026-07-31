#!/usr/bin/env python3
"""
Audited train/lock/OOS for cash-only long strategy.

Fixes vs prior FINALMODEL:
  - NO full-sample durable universe (look-ahead). Uses train-frozen walk-forward
    selected pairs, or expanding PIT schedule.
  - execution_lag_days=1 (signal t, trade t+1).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from statarb.data.bars_loader import load_bars_from_db
from statarb.data.storage import Storage
from statarb.data.yahoo import ensure_yahoo_history, fetch_yahoo_bars
from statarb.strategy.cash_long_only import (
    CashLongConfig,
    CashLongOnlyEngine,
    load_pit_schedule,
    make_fixed_candidate_fn,
    make_pit_candidate_fn,
)


def spy_bh(bars, start: date, end: date, start_eq: float = 100_000.0) -> dict:
    xs = [b for b in bars if start <= b.timestamp.date() <= end]
    if len(xs) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "ending_equity": start_eq,
        }
    px = pd.Series(
        [b.close for b in xs],
        index=pd.to_datetime([b.timestamp.replace(tzinfo=None) for b in xs]),
    )
    rt = 0.0009
    eq = start_eq * (px / px.iloc[0]) * (1 - rt)
    rets = eq.pct_change().dropna()
    years = max((end - start).days / 365.25, 1e-6)
    total = (float(eq.iloc[-1]) - start_eq) / start_eq
    cagr = (float(eq.iloc[-1]) / start_eq) ** (1 / years) - 1
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    dd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe_ratio": sharpe,
        "max_drawdown": dd,
        "ending_equity": float(eq.iloc[-1]),
    }


def run_cfg(cfg, cand_fn, sector, bars, start, end) -> dict:
    eng = CashLongOnlyEngine(cfg)
    return eng.run(cand_fn, bars, start, end, sector_of=sector).summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--universe",
        choices=["train_frozen", "pit"],
        default="train_frozen",
        help="train_frozen=pairs selected in WF cycles ≤2017; pit=expanding prior selections",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    # Ensure PIT artifacts exist
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_pit_candidates", ROOT / "scripts" / "build_pit_candidates.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main()

    train_end = date(2017, 12, 29)
    oos_start = date(2018, 1, 2)
    bt_start = date(2010, 1, 4)
    bt_end = date(2026, 7, 29)

    frozen = pd.read_csv(ROOT / "logs" / "train_frozen_selected_universe.csv")
    frozen_pairs = [(r.symbol_a, r.symbol_b) for r in frozen.itertuples()]
    # sector unknown in frozen file — leave blank
    sector = {s: "" for a, b in frozen_pairs for s in (a, b)}

    pit_sched = load_pit_schedule(ROOT / "logs" / "pit_candidates_long.csv")
    if args.universe == "train_frozen":
        cand_fn = make_fixed_candidate_fn(frozen_pairs)
        uni_note = f"train_frozen_selected n={len(frozen_pairs)} (WF selected ≤{train_end})"
    else:
        cand_fn = make_pit_candidate_fn(pit_sched)
        uni_note = "expanding PIT prior WF selected pairs"

    syms = sorted({s for a, b in frozen_pairs for s in (a, b)} | {"SPY"})
    # PIT may reference more symbols over time
    if args.universe == "pit":
        for pairs in pit_sched.values():
            for a, b in pairs:
                syms.extend([a, b])
        syms = sorted(set(syms) | {"SPY"})

    db = ROOT / "data" / "statarb.db"
    print(f"universe={args.universe} ({uni_note})", flush=True)
    print(f"loading {len(syms)} symbols…", flush=True)
    bars = load_bars_from_db(db, syms, start=date(2008, 1, 1), end=bt_end)
    if "SPY" not in bars:
        storage = Storage(str(db))
        got = ensure_yahoo_history(
            storage, ["SPY"], start=date(2008, 1, 1), end=bt_end, force=True
        )
        bars["SPY"] = got.get("SPY") or fetch_yahoo_bars("SPY", date(2008, 1, 1), bt_end)
    print(f"loaded {len(bars)}", flush=True)

    # Leakage contrast: old full-sample durable (for audit report only)
    from statarb.strategy.portable_alpha import load_durable_candidates

    leak_pairs, leak_sec = load_durable_candidates(
        ROOT / "logs" / "walkforward_coint_durable_universe.csv"
    )

    if args.fast:
        grid = [
            # baseline audited
            dict(top_n=8, open_z=1.25, position_pct=0.12, min_spy_pct=0.45, max_concurrent_longs=5, spy_ma_filter=0),
            dict(top_n=6, open_z=1.5, position_pct=0.10, min_spy_pct=0.55, max_concurrent_longs=4, spy_ma_filter=0),
            # drawdown-focused
            dict(top_n=6, open_z=1.5, position_pct=0.08, min_spy_pct=0.70, max_concurrent_longs=3, spy_ma_filter=200),
            dict(top_n=5, open_z=2.0, position_pct=0.08, min_spy_pct=0.65, max_concurrent_longs=3, spy_ma_filter=200),
            dict(top_n=8, open_z=1.5, position_pct=0.10, min_spy_pct=0.60, max_concurrent_longs=4, spy_ma_filter=200),
            dict(top_n=8, open_z=1.25, position_pct=0.15, min_spy_pct=0.40, max_concurrent_longs=6, spy_ma_filter=0),
        ]
    else:
        grid = []
        for top_n in (5, 6, 8):
            for open_z in (1.25, 1.5, 2.0):
                for pos in (0.08, 0.10, 0.12):
                    for min_spy in (0.45, 0.60, 0.70):
                        for ma in (0, 200):
                            grid.append(
                                dict(
                                    top_n=top_n,
                                    open_z=open_z,
                                    position_pct=pos,
                                    min_spy_pct=min_spy,
                                    max_concurrent_longs=min(top_n, 5),
                                    spy_ma_filter=ma,
                                )
                            )

    spy_train = spy_bh(bars["SPY"], bt_start, train_end)
    print(f"TRAIN {bt_start}→{train_end} SPY={spy_train['total_return']:+.2%} grid={len(grid)}", flush=True)
    rows = []
    for i, p in enumerate(grid):
        cfg = CashLongConfig(execution_lag_days=1, **p)
        print(f"  [{i+1}/{len(grid)}] {p}", flush=True)
        m = run_cfg(cfg, cand_fn, sector, bars, bt_start, train_end)
        if m.get("error"):
            print("   error", m, flush=True)
            continue
        excess = m["total_return"] - spy_train["total_return"]
        rows.append({**p, **m, "spy_return": spy_train["total_return"], "excess": excess})
        print(
            f"    ret={m['total_return']:+.2%} excess={excess:+.2%} "
            f"sharpe={m['sharpe_ratio']:.3f} dd={m['max_drawdown']:.2%}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "logs" / "cash_long_only_train_grid_audited.csv", index=False)

    # Primary: beat SPY; secondary prefer lower |DD| among beaters
    ok = df[(df.excess > 0) & (df.max_drawdown > -0.40)]
    if ok.empty:
        ok = df[df.excess > 0]
    if ok.empty:
        print("WARN: no config beat SPY on train", flush=True)
        ok = df.copy()
    ok = ok.assign(score=ok.excess + 0.15 * ok.sharpe_ratio + 0.5 * (ok.max_drawdown + 0.3))
    best = ok.sort_values(["excess", "max_drawdown", "sharpe_ratio"], ascending=[False, False, False]).iloc[0]

    # Best low-DD among those within 50% of best excess (if any beat)
    beaters = df[df.excess > 0]
    if not beaters.empty:
        low_dd = beaters.sort_values("max_drawdown", ascending=False).iloc[0]
    else:
        low_dd = best

    locked = {
        "universe_mode": args.universe,
        "universe_note": uni_note,
        "top_n": int(best.top_n),
        "open_z": float(best.open_z),
        "position_pct": float(best.position_pct),
        "min_spy_pct": float(best.min_spy_pct),
        "max_concurrent_longs": int(best.max_concurrent_longs),
        "spy_ma_filter": int(best.spy_ma_filter),
        "execution_lag_days": 1,
        "formation_days": 252,
        "trading_days": 63,
        "min_edge_cost_multiple": 2.0,
        "uses_margin": False,
        "allows_shorts": False,
        "params_locked": True,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "train_window": {"start": str(bt_start), "end": str(train_end)},
        "train_metrics": {
            "total_return": float(best.total_return),
            "excess_vs_spy": float(best.excess),
            "sharpe_ratio": float(best.sharpe_ratio),
            "max_drawdown": float(best.max_drawdown),
            "spy_return": float(best.spy_return),
        },
        "low_dd_alternative": {
            "top_n": int(low_dd.top_n),
            "open_z": float(low_dd.open_z),
            "position_pct": float(low_dd.position_pct),
            "min_spy_pct": float(low_dd.min_spy_pct),
            "max_concurrent_longs": int(low_dd.max_concurrent_longs),
            "spy_ma_filter": int(low_dd.spy_ma_filter),
            "train_return": float(low_dd.total_return),
            "train_excess": float(low_dd.excess),
            "train_max_drawdown": float(low_dd.max_drawdown),
            "train_sharpe": float(low_dd.sharpe_ratio),
        },
    }
    print("\nTRAIN best:", {k: locked[k] for k in ("top_n", "open_z", "position_pct", "min_spy_pct", "spy_ma_filter", "train_metrics")}, flush=True)

    lock_path = ROOT / "config" / "cash_long_only.locked.yaml"
    if args.lock:
        lock_path.write_text(yaml.safe_dump(locked, sort_keys=False))
        print("LOCKED →", lock_path, flush=True)

    def cfg_from(row_or_locked):
        return CashLongConfig(
            top_n=int(row_or_locked["top_n"]),
            open_z=float(row_or_locked["open_z"]),
            position_pct=float(row_or_locked["position_pct"]),
            min_spy_pct=float(row_or_locked["min_spy_pct"]),
            max_concurrent_longs=int(row_or_locked["max_concurrent_longs"]),
            spy_ma_filter=int(row_or_locked.get("spy_ma_filter", 0)),
            execution_lag_days=1,
        )

    cfg_oos = cfg_from(locked)
    spy_oos = spy_bh(bars["SPY"], oos_start, bt_end)
    m_oos = run_cfg(cfg_oos, cand_fn, sector, bars, oos_start, bt_end)
    excess_oos = m_oos["total_return"] - spy_oos["total_return"]

    m_full = run_cfg(cfg_oos, cand_fn, sector, bars, bt_start, bt_end)
    spy_full = spy_bh(bars["SPY"], bt_start, bt_end)

    # Audit: leaked durable universe with same locked params (contaminated)
    leak_fn = make_fixed_candidate_fn(leak_pairs)
    m_leak_oos = run_cfg(cfg_oos, leak_fn, leak_sec, bars, oos_start, bt_end)
    leak_excess = m_leak_oos["total_return"] - spy_oos["total_return"]

    # Low-DD alt OOS
    cfg_dd = cfg_from(locked["low_dd_alternative"])
    m_dd_oos = run_cfg(cfg_dd, cand_fn, sector, bars, oos_start, bt_end)
    m_dd_full = run_cfg(cfg_dd, cand_fn, sector, bars, bt_start, bt_end)

    eng = CashLongOnlyEngine(cfg_oos)
    res = eng.run(cand_fn, bars, bt_start, bt_end, sector_of=sector)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    curve = ROOT / "logs" / f"cash_long_only_audited_curve_{ts}.csv"
    pd.DataFrame(
        [(t.date().isoformat(), e) for t, e in res.equity_curve],
        columns=["date", "equity"],
    ).to_csv(curve, index=False)

    report = {
        "strategy": "cash_long_only_spy_core_rv_satellite",
        "audit": {
            "full_sample_durable_universe_leak": True,
            "fix": "Prior FINALMODEL used durable≥3 counted over 2010-2026 (look-ahead on candidates).",
            "remediation": uni_note,
            "execution_lag_days": 1,
            "leaked_universe_oos_excess_for_contrast": leak_excess,
            "clean_universe_oos_excess": excess_oos,
        },
        "uses_margin": False,
        "locked_params": locked,
        "train": locked["train_metrics"],
        "oos": {
            "start": str(oos_start),
            "end": str(bt_end),
            "strategy": {
                k: m_oos[k]
                for k in (
                    "total_return",
                    "cagr",
                    "sharpe_ratio",
                    "max_drawdown",
                    "ending_equity",
                    "number_of_fills",
                )
                if k in m_oos
            },
            "spy": spy_oos,
            "excess_vs_spy": excess_oos,
            "beats_spy": excess_oos > 0,
        },
        "full_sample": {
            "strategy": {
                k: m_full[k]
                for k in (
                    "total_return",
                    "cagr",
                    "sharpe_ratio",
                    "max_drawdown",
                    "ending_equity",
                    "number_of_fills",
                )
                if k in m_full
            },
            "spy": spy_full,
            "excess_vs_spy": m_full["total_return"] - spy_full["total_return"],
        },
        "low_dd_alternative_oos": {
            "params": locked["low_dd_alternative"],
            "strategy": {
                k: m_dd_oos[k]
                for k in ("total_return", "cagr", "sharpe_ratio", "max_drawdown")
                if k in m_dd_oos
            },
            "excess_vs_spy": m_dd_oos["total_return"] - spy_oos["total_return"],
        },
        "low_dd_alternative_full": {
            "strategy": {
                k: m_dd_full[k]
                for k in ("total_return", "cagr", "sharpe_ratio", "max_drawdown")
                if k in m_dd_full
            },
            "excess_vs_spy": m_dd_full["total_return"] - spy_full["total_return"],
        },
    }
    report_path = ROOT / "logs" / f"cash_long_only_audited_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2))
    (ROOT / "logs" / f"cash_long_only_audited_full_{ts}.json").write_text(
        json.dumps({"summary": res.summary}, indent=2)
    )

    print("\n=== AUDITED OOS ===", flush=True)
    print(
        f"Clean:  ret={m_oos['total_return']:+.2%} excess={excess_oos:+.2%} "
        f"sharpe={m_oos['sharpe_ratio']:.3f} dd={m_oos['max_drawdown']:.2%}",
        flush=True,
    )
    print(
        f"Leaked durable (contrast): OOS excess={leak_excess:+.2%}",
        flush=True,
    )
    print(
        f"FULL clean: ret={m_full['total_return']:+.2%} vs SPY {spy_full['total_return']:+.2%} "
        f"excess={m_full['total_return']-spy_full['total_return']:+.2%} dd={m_full['max_drawdown']:.2%}",
        flush=True,
    )
    print(
        f"LOW-DD alt OOS: ret={m_dd_oos['total_return']:+.2%} "
        f"excess={m_dd_oos['total_return']-spy_oos['total_return']:+.2%} "
        f"dd={m_dd_oos['max_drawdown']:.2%}",
        flush=True,
    )
    print("saved", report_path, curve, flush=True)
    return 0 if excess_oos > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
