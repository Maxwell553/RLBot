#!/usr/bin/env python3
"""Compare OOS metrics for earliest, latest step, and best checkpoints across Runs/.

This is an audit script, not a routine reporting command: each comparison runs
additional holdout backtests and can add OOS burn-ledger entries. Use it only
when intentionally spending those reads.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rlbot.run_artifacts import RunPaths, discover_run_ids_with_models, read_run_manifest


def _step_checkpoints(run_id: str) -> tuple[Path | None, Path | None]:
    ckpt_dir = RunPaths(run_id).models_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None, None
    earliest: Path | None = None
    latest: Path | None = None
    lo = hi = -1
    for p in ckpt_dir.glob("ppo_*_steps.zip"):
        m = re.search(r"ppo_(\d+)_steps\.zip$", p.name)
        if not m:
            continue
        step = int(m.group(1))
        if lo < 0 or step < lo:
            lo = step
            earliest = p
        if step > hi:
            hi = step
            latest = p
    return earliest, latest


@dataclass
class Row:
    run_id: str
    variant: str
    model_path: str
    total_return: float
    sharpe: float
    max_drawdown: float
    n_bars: int
    error: str = ""


def _winner(rows: list[Row], key: str) -> str:
    ok = [r for r in rows if not r.error]
    if not ok:
        return "none"
    if key == "sharpe":
        best = max(ok, key=lambda r: r.sharpe)
    else:
        best = max(ok, key=lambda r: r.total_return)
    return best.variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "Runs" / "checkpoint_comparison.json"),
        help="Write incremental JSON results here.",
    )
    parser.add_argument("--run-id", action="append", default=[], help="Limit to specific run(s).")
    parser.add_argument(
        "--i-understand-oos-burn",
        action="store_true",
        help="Required: acknowledge that this runs additional OOS backtests.",
    )
    args = parser.parse_args()
    if not args.i_understand_oos_burn:
        raise SystemExit(
            "Refusing to compare checkpoints without --i-understand-oos-burn. "
            "This script runs additional OOS backtests and may update the burn ledger."
        )

    # Import after argparse so --help is fast.
    from scripts.backtest import ensure_backtest_dependencies, run_oos_backtest

    ensure_backtest_dependencies()
    import argparse as ap

    run_ids = args.run_id or discover_run_ids_with_models()
    run_ids = sorted(run_ids)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[Row] = []
    per_run: dict[str, dict[str, object]] = {}
    t0 = time.perf_counter()

    for i, run_id in enumerate(run_ids, 1):
        if read_run_manifest(run_id) is None:
            print(f"[{i}/{len(run_ids)}] skip {run_id}: no manifest")
            continue

        rp = RunPaths(run_id)
        best = rp.best_model_dir / "best_model.zip"
        earliest, latest = _step_checkpoints(run_id)

        variants: list[tuple[str, Path | None]] = [
            ("earliest", earliest),
            ("latest", latest),
            ("best", best if best.is_file() else None),
        ]

        run_rows: list[Row] = []
        print(f"[{i}/{len(run_ids)}] {run_id}", flush=True)
        for variant, model in variants:
            if model is None:
                row = Row(run_id, variant, "", 0.0, 0.0, 0.0, 0, error="missing checkpoint")
                run_rows.append(row)
                print(f"  {variant}: missing", flush=True)
                continue
            bt_args = ap.Namespace(
                run_id=run_id,
                model=str(model),
                no_viz=True,
                no_progress=True,
                fast=True,
                stochastic_paths=0,
                detailed=False,
                allow_latest_checkpoint=False,
                reuse_panel=True,
                plot_tag=variant,
                until=None,
                train_end=None,
                holdout_start=None,
                holdout_end=None,
                holdout_days=None,
                obs_lag=None,
                data_cache="",
                vec_normalize="",
                use_current_config=False,
                allow_missing_vec_normalize=False,
                allow_raw_obs=False,
                device="cpu",
                full_policy_load=False,
                bootstrap_resamples=2000,
                bootstrap_avg_block=10,
            )
            try:
                result = run_oos_backtest(bt_args)
                row = Row(
                    run_id,
                    variant,
                    str(result.model_path),
                    result.total_return,
                    result.sharpe,
                    result.max_drawdown,
                    result.n_bars,
                )
                print(
                    f"  {variant}: ret={row.total_return * 100:.2f}% sharpe={row.sharpe:.2f} "
                    f"dd={row.max_drawdown * 100:.2f}%",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 — batch audit script
                row = Row(run_id, variant, str(model), 0.0, 0.0, 0.0, 0, error=str(exc))
                print(f"  {variant}: ERROR {exc}", flush=True)
            run_rows.append(row)

        all_rows.extend(run_rows)
        per_run[run_id] = {
            "rows": [asdict(r) for r in run_rows],
            "best_sharpe": _winner(run_rows, "sharpe"),
            "best_return": _winner(run_rows, "return"),
        }

        payload = {
            "elapsed_s": time.perf_counter() - t0,
            "n_runs": len(per_run),
            "per_run": per_run,
            "summary": _aggregate(per_run),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Aggregate (per-run winner counts) ===")
    summary = _aggregate(per_run)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nWrote {out_path} ({time.perf_counter() - t0:.0f}s)")


def _aggregate(per_run: dict[str, dict[str, object]]) -> dict[str, object]:
    sharpe_wins: dict[str, int] = {"earliest": 0, "latest": 0, "best": 0, "none": 0}
    return_wins: dict[str, int] = {"earliest": 0, "latest": 0, "best": 0, "none": 0}
    beats_earliest_sharpe = {"latest": 0, "best": 0, "tie": 0, "missing": 0}
    beats_earliest_return = {"latest": 0, "best": 0, "tie": 0, "missing": 0}

    sharpe_deltas: list[dict[str, float | str]] = []
    for run_id, info in per_run.items():
        rows = {r["variant"]: r for r in info["rows"]}  # type: ignore[index]
        ws = str(info["best_sharpe"])
        wr = str(info["best_return"])
        sharpe_wins[ws] = sharpe_wins.get(ws, 0) + 1
        return_wins[wr] = return_wins.get(wr, 0) + 1

        e = rows.get("earliest", {})
        if e.get("error"):
            beats_earliest_sharpe["missing"] += 1
            beats_earliest_return["missing"] += 1
            continue
        e_sh = float(e["sharpe"])
        e_ret = float(e["total_return"])
        delta_row: dict[str, float | str] = {"run_id": run_id}
        for variant in ("latest", "best"):
            v = rows.get(variant, {})
            if v.get("error"):
                continue
            v_sh = float(v["sharpe"])
            v_ret = float(v["total_return"])
            if v_sh > e_sh + 1e-9:
                beats_earliest_sharpe[variant] += 1
            elif abs(v_sh - e_sh) <= 1e-9:
                beats_earliest_sharpe["tie"] += 1
            if v_ret > e_ret + 1e-9:
                beats_earliest_return[variant] += 1
            elif abs(v_ret - e_ret) <= 1e-9:
                beats_earliest_return["tie"] += 1
            delta_row[f"{variant}_minus_earliest_sharpe"] = v_sh - e_sh
            delta_row[f"{variant}_minus_earliest_return"] = v_ret - e_ret
        if len(delta_row) > 1:
            sharpe_deltas.append(delta_row)

    mean_delta: dict[str, float] = {}
    if sharpe_deltas:
        for variant in ("latest", "best"):
            sh_key = f"{variant}_minus_earliest_sharpe"
            ret_key = f"{variant}_minus_earliest_return"
            sh_vals = [float(d[sh_key]) for d in sharpe_deltas if sh_key in d]
            ret_vals = [float(d[ret_key]) for d in sharpe_deltas if ret_key in d]
            if sh_vals:
                mean_delta[f"mean_{sh_key}"] = sum(sh_vals) / len(sh_vals)
            if ret_vals:
                mean_delta[f"mean_{ret_key}"] = sum(ret_vals) / len(ret_vals)

    return {
        "n_runs_compared": len(per_run),
        "per_run_sharpe_winner": sharpe_wins,
        "per_run_return_winner": return_wins,
        "vs_earliest_sharpe": beats_earliest_sharpe,
        "vs_earliest_return": beats_earliest_return,
        "mean_deltas_vs_earliest": mean_delta,
    }


if __name__ == "__main__":
    main()
