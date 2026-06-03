#!/usr/bin/env python3
"""
Run the same saved policy on multiple time slices, with **explicit** leakage
labels so you do not confuse in-sample path checks with OOS.

Universe must match the training run (manifest ``universe.tickers``) and VecNormalize stats.
You **cannot** swap tickers without retraining — observation shape is tied to the saved model.

- **HOLDOUT_OOS**: the trailing calendar holdout (same as scripts/train.py / scripts/backtest.py).
- **YEAR_IN_SAMPLE** / **FULL_TRAIN_IN_SAMPLE**: the policy and VecNormalize were fit
  on the trainable timeline, so these are not prospective OOS for the learned weights.
  Use them for stress / regime behavior, not to claim new generalization.

No look-ahead in features: same causal env + obs_lag as training; VecNormalize is
from the run (not refit per slice), matching how you would deploy the saved model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path as _Path

_bootstrap_path = _Path(__file__).resolve().parent / "_bootstrap.py"
_bootstrap_spec = importlib.util.spec_from_file_location("_rlbot_repo_bootstrap", _bootstrap_path)
assert _bootstrap_spec is not None and _bootstrap_spec.loader is not None
_bootstrap_mod = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap_mod)

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO

from scripts.backtest import (
    DATA_CACHE,
    _find_vec_normalize,
    _max_drawdown,
    _sharpe_ann_from_log_rets,
    _resolve_model_path,
    rollout_policy_on_slice,
)
from rlbot.data_utils import (
    benchmark_ohlcv_index,
    load_cache,
    resolve_panel_tickers,
    reserve_chronological_holdout,
)
from rlbot.run_artifacts import read_latest_run_id, read_run_manifest

_ROOT = Path(__file__).resolve().parent.parent
_PLOTS = _ROOT / "plots"


def _holdout_days(manifest: dict | None) -> int:
    if not manifest:
        return 365
    if manifest.get("chronological_holdout"):
        return int(manifest["chronological_holdout"]["holdout_days"])
    a = manifest.get("args", {})
    if a.get("holdout_days") is not None:
        return int(a["holdout_days"])
    return 365


def _weight_index(tickers: list[str], name: str) -> int | None:
    return tickers.index(name) if name in tickers else None


def _mean_weight(wmean: np.ndarray, idx: int | None) -> str | float:
    if idx is None or wmean.size <= idx:
        return ""
    return round(float(wmean[idx]), 3)


def _spy_metrics(
    ohlcv: np.ndarray,
    start_bar: int,
    nav0: float,
    navs_len: int,
    tickers: list[str],
) -> tuple[float, float, float]:
    close = ohlcv[:, benchmark_ohlcv_index(tickers), 3]
    i0 = int(start_bar)
    i1 = int(start_bar + navs_len - 1)
    s0 = max(float(close[i0]), 1e-12)
    path = (close[i0 : i1 + 1] / s0) * float(nav0)
    lr = np.diff(np.log(np.maximum(path, 1e-12)))
    if lr.size < 2:
        return float("nan"), float("nan"), float("nan")
    tot = float(path[-1] / path[0] - 1.0)
    sh = _sharpe_ann_from_log_rets(lr)
    mdd = _max_drawdown(path)
    return tot, sh, mdd


def _one_row(
    kind: str,
    label: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    ohlcv: np.ndarray,
    start_bar: int,
    navs: np.ndarray,
    wmean: np.ndarray,
    tickers: list[str],
) -> dict:
    log_rets = np.diff(np.log(np.maximum(navs, 1e-12)))
    n_steps = int(len(navs) - 1)
    n_bars = int(ohlcv.shape[0])
    tot = float(navs[-1] / max(navs[0], 1e-12) - 1.0)
    sh = _sharpe_ann_from_log_rets(log_rets) if log_rets.size else float("nan")
    mdd = _max_drawdown(navs) * 100.0
    spy_t, spy_sh, _ = _spy_metrics(ohlcv, start_bar, float(navs[0]), len(navs), tickers)
    i_sp = _weight_index(tickers, "SP500")
    i_nik = _weight_index(tickers, "NIKKEI")
    i_em = _weight_index(tickers, "EM")
    return {
        "kind": kind,
        "label": label,
        "t0": str(t0)[:10],
        "t1": str(t1)[:10],
        "n_bars": n_bars,
        "n_steps": n_steps,
        "ret_pct": round(tot * 100, 2),
        "sharpe": round(sh, 2),
        "max_dd_pct": round(mdd, 2),
        "spy_ret_pct": round(spy_t * 100, 2),
        "spy_sharpe": round(spy_sh, 2) if not np.isnan(spy_sh) else "",
        "w_cash": _mean_weight(wmean, 0),
        "w_SPY": _mean_weight(wmean, (i_sp + 1) if i_sp is not None else None),
        "w_N225": _mean_weight(wmean, (i_nik + 1) if i_nik is not None else None),
        "w_EEM": _mean_weight(wmean, (i_em + 1) if i_em is not None else None),
    }


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No rows.")
        return
    keys = list(rows[0].keys())
    w = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    line = " | ".join(f"{k:{w[k]}s}" for k in keys)
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(f"{str(r.get(k, '')):{w[k]}}" for k in keys))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="", help="e.g. 65M_4_20_26_a")
    p.add_argument("--min-bars", type=int, default=120, help="Skip year slices with fewer daily rows")
    p.add_argument("--obs-lag", type=int, default=1)
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="Write CSV to this path (default: plots/<run-id>/sweep_regimes.csv)",
    )
    p.add_argument(
        "--include-full-train",
        action="store_true",
        help="One LSTM episode over the entire trainable span (in-sample; can "
        "compound to unrealistic NAV; not comparable to calendar-year rows).",
    )
    args = p.parse_args()

    ap = argparse.Namespace(model="", run_id=args.run_id)
    if not ap.run_id.strip() and (lid := read_latest_run_id()):
        ap.run_id = lid
    model_path, run_hint = _resolve_model_path(ap)
    if not args.run_id.strip() and run_hint:
        args.run_id = run_hint
    if not str(args.run_id or "").strip():
        raise SystemExit("Pass --run-id (or have runs/LATEST.txt).")

    manifest = read_run_manifest(args.run_id) or None
    hd = _holdout_days(manifest)
    out_csv = (
        Path(args.csv).expanduser() if args.csv.strip() else _PLOTS / str(args.run_id) / "sweep_regimes.csv"
    )

    (
        idx,
        ohlcv,
        rsi,
        macd,
        macro,
        fracdiff,
        fracdiff_macro,
        trend,
        cache_tickers,
    ) = load_cache(str(DATA_CACHE))
    panel_tickers = resolve_panel_tickers(manifest, cache_tickers)
    n_actions = len(panel_tickers) + 1

    print(
        f"""
=== Regime / calendar sweep  (run: {args.run_id})  holdout_days={hd}  ===
Universe (must match training manifest / cache; not swappable for this .zip / vecnorm):
{", ".join(panel_tickers)}

HOLDOUT_OOS  =  strict future block excluded from PPO (only honest OOS for this .zip)
YEAR_* / FULL_TRAIN_*  =  in-sample path checks; NOT new evidence of generalization
""".strip()
        + "\n"
    )
    (
        (t_idx, t_oh, t_rsi, t_macd, t_m, t_fd, t_fdm, t_trend),
        (h_idx, h_oh, h_rsi, h_macd, h_m, h_fd, h_fdm, h_trend),
    ) = reserve_chronological_holdout(
        idx,
        ohlcv,
        rsi,
        macd,
        macro,
        fracdiff,
        fracdiff_macro,
        trend,
        holdout_days=hd,
    )

    model = RecurrentPPO.load(str(model_path), device="auto")
    vn = _find_vec_normalize(model_path, str(args.run_id), explicit=None)
    if not vn.is_file():
        raise SystemExit(f"Need VecNormalize at {vn}")

    rows: list[dict] = []
    ylist = sorted(np.unique(t_idx.year).tolist()) if len(t_idx) else []

    for y in ylist:
        m = t_idx.year == y
        if m.sum() < int(args.min_bars):
            continue
        sidx = t_idx[m]
        if len(sidx) < 10:
            continue
        nav, sb, _nr, wopt = rollout_policy_on_slice(
            model,
            test_idx=sidx,
            test_ohlcv=t_oh[m, ...],
            test_rsi=t_rsi[m, ...],
            test_macd=t_macd[m, ...],
            test_macro=t_m[m, ...],
            test_fd=t_fd[m, ...],
            test_fdm=t_fdm[m, ...],
            test_trend=t_trend[m, ...],
            obs_lag=int(args.obs_lag),
            vec_norm_path=vn,
            use_vec_norm=True,
            collect_weights=True,
        )
        wm = wopt.mean(axis=0) if wopt is not None and wopt.size else np.zeros(n_actions, dtype=np.float64)
        rows.append(
            _one_row(
                "YEAR_IN_SAMPLE", f"year_{y}", sidx[0], sidx[-1], t_oh[m, ...], sb, nav, wm, panel_tickers
            )
        )

    if args.include_full_train and len(t_idx) >= int(args.min_bars):
        nav, sb, _nr, wopt = rollout_policy_on_slice(
            model,
            test_idx=t_idx,
            test_ohlcv=t_oh,
            test_rsi=t_rsi,
            test_macd=t_macd,
            test_macro=t_m,
            test_fd=t_fd,
            test_fdm=t_fdm,
            test_trend=t_trend,
            obs_lag=int(args.obs_lag),
            vec_norm_path=vn,
            use_vec_norm=True,
            collect_weights=True,
        )
        wm = wopt.mean(axis=0) if wopt is not None and wopt.size else np.zeros(n_actions, dtype=np.float64)
        t_end = str(t_idx[-1].date()) if hasattr(t_idx[-1], "date") else str(t_idx[-1])
        rows.append(
            _one_row(
                "FULL_TRAIN_IN_SAMPLE",
                f"full_train_..{t_end}",
                t_idx[0],
                t_idx[-1],
                t_oh,
                sb,
                nav,
                wm,
                panel_tickers,
            )
        )

    if len(h_idx) >= 10:
        nav, sb, _nr, wopt = rollout_policy_on_slice(
            model,
            test_idx=h_idx,
            test_ohlcv=h_oh,
            test_rsi=h_rsi,
            test_macd=h_macd,
            test_macro=h_m,
            test_fd=h_fd,
            test_fdm=h_fdm,
            test_trend=h_trend,
            obs_lag=int(args.obs_lag),
            vec_norm_path=vn,
            use_vec_norm=True,
            collect_weights=True,
        )
        wm = wopt.mean(axis=0) if wopt is not None and wopt.size else np.zeros(n_actions, dtype=np.float64)
        rows.append(
            _one_row(
                "HOLDOUT_OOS", "official_holdout", h_idx[0], h_idx[-1], h_oh, sb, nav, wm, panel_tickers
            )
        )

    _k = {"YEAR_IN_SAMPLE": 0, "FULL_TRAIN_IN_SAMPLE": 1, "HOLDOUT_OOS": 2}
    rows.sort(key=lambda r: (_k.get(r["kind"], 9), r["t0"]))

    print("Results:")
    _print_table(rows)
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
