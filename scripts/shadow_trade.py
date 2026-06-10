"""Tier-5 shadow trading: forward evaluation that never burns a holdout.

A daily loop records the model's audited target weights forward in time and
reconciles them against realized next-bar returns — true walk-forward evidence
accumulating one bar per day, with full provenance and zero broker risk.

    python scripts/shadow_trade.py record    --run-id <RUN_ID> --refresh-data
    python scripts/shadow_trade.py reconcile --run-id <RUN_ID>
    python scripts/shadow_trade.py report    --run-id <RUN_ID>

``record`` (run after the close) refreshes the GLOBAL data cache when
``--refresh-data`` is passed (the run-local snapshot is frozen at training time
by design — recording from it would no-op forever), then runs the audited
inference path (scripts/infer_weights.py — frozen VecNormalize, manifest-checked
panel, OOS-ledger logged) against that cache and appends one row to
``execution/shadow_ledger_<RUN_ID>.jsonl`` (gitignored: live state never enters
the research tree). A staleness warning fires when the newest bar is old. An
observation-drift alarm fires when current features sit far outside the frozen
normalization stats.

The recorded book is keyed by its **decision bar** — the bar whose observation
actually produced the weights (the rollout env needs two later bars to execute
and mark a step, so the decision lags the cache tail; the ledger is honest about
this rather than pretending the weights are as-of the newest bar).

``reconcile`` is torch-free: once the cache holds the decision bar + 2, it fills
in the realized open→open portfolio return, NET of linear transaction costs
(turnover vs the previously recorded book × per-asset slippage+fee, plus daily
holding costs) alongside the cap-weighted benchmark (buy-and-hold: holding costs
only). No market-impact or capacity model — linear costs only.

The ledger is append-only; reconciliation writes a sibling ``*_reconciled.jsonl``
so raw records are never mutated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from rlbot.data_utils import load_cache, resolve_panel_tickers  # noqa: E402
from rlbot.rl_config import get_config, load_config, set_config  # noqa: E402
from rlbot.run_artifacts import (  # noqa: E402
    PROJECT_ROOT,
    RunPaths,
    read_run_manifest,
    resolve_data_cache,
)

EXECUTION_DIR = PROJECT_ROOT / "execution"
STALE_BAR_WARN_DAYS = 5  # warn when the newest cache bar is older than this


def ledger_path(run_id: str) -> Path:
    return EXECUTION_DIR / f"shadow_ledger_{run_id}.jsonl"


def reconciled_path(run_id: str) -> Path:
    return EXECUTION_DIR / f"shadow_ledger_{run_id}_reconciled.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _bind_run_config(run_id: str, use_current: bool) -> None:
    snap = RunPaths(run_id).config_snapshot
    if snap.is_file() and not use_current:
        set_config(load_config(snap))


# ── record (needs torch via the infer subprocess) ─────────────────────────


def obs_drift_alarm(z_scores: np.ndarray, threshold: float = 5.0, frac: float = 0.05) -> bool:
    """True when more than ``frac`` of features sit beyond ``threshold`` sigma of the
    frozen training normalization — the policy is seeing a regime its stats never
    covered, and its outputs should not be trusted blindly. The 5σ/5% defaults are
    HEURISTICS (override via --drift-sigma/--drift-frac); calibrate against the
    alarm rate observed over a quiet month."""
    z = np.abs(np.asarray(z_scores, dtype=np.float64).reshape(-1))
    return bool(np.mean(z > threshold) > frac)


def _refresh_global_cache() -> Path:
    """Fetch fresh bars and rebuild the GLOBAL cache (torch-free; same feature
    pipeline the trainer uses for the full-timeline cache)."""
    from rlbot.data_utils import _indicators_from_merged, fetch_aligned_daily, save_cache

    cfg = get_config()
    cache_path = resolve_data_cache()
    print(f"[shadow] refreshing global cache → {cache_path}")
    merged = fetch_aligned_daily(since=cfg.data.since, until=None)
    (idx, ohlcv, rsi, macd, macro, fd, fdm, trend, avol, mvol, live) = _indicators_from_merged(
        merged, list(cfg.universe.tickers), fracdiff_d=cfg.data.fracdiff_d
    )
    save_cache(
        str(cache_path), idx, ohlcv, rsi, macd, macro, fd, fdm, trend, avol, mvol,
        asset_live=live, fracdiff_d=cfg.data.fracdiff_d, tickers=list(cfg.universe.tickers),
    )
    return cache_path


def cmd_record(args: argparse.Namespace) -> None:
    run_id = args.run_id.strip()
    manifest = read_run_manifest(run_id)
    if manifest is None:
        raise SystemExit(f"Missing Runs/{run_id}/manifest.json")
    # Config context for the refresh (universe/since); inference itself binds the
    # run snapshot inside the infer_weights subprocess.
    _bind_run_config(run_id, args.use_current_config)

    if args.refresh_data:
        cache_path = _refresh_global_cache()
    else:
        # The run-local snapshot is frozen at training time — recording from it
        # would replay the same bar forever. Default to the refreshable global cache.
        cache_path = Path(args.data_cache) if args.data_cache.strip() else resolve_data_cache()
    if not Path(cache_path).is_file():
        raise SystemExit(
            f"No cache at {cache_path}; run with --refresh-data (or build it via "
            "scripts/train.py --refresh-data)."
        )

    import subprocess

    out_json = EXECUTION_DIR / f"_shadow_weights_{run_id}_{os.getpid()}.json"
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "infer_weights.py"),
        "--run-id", run_id, "--checkpoint", args.checkpoint,
        "--data-cache", str(cache_path), "--emit-observation",
        "--out", str(out_json),
    ]
    if args.use_current_config:
        cmd.append("--use-current-config")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        payload = json.loads(out_json.read_text(encoding="utf-8"))
    finally:
        out_json.unlink(missing_ok=True)

    prov = payload.get("provenance") or {}
    as_of = payload.get("as_of")
    decision_bar = payload.get("decision_bar") or as_of
    last_bar = prov.get("as_of_last_bar") or as_of
    import pandas as pd

    age = (pd.Timestamp.now().normalize() - pd.Timestamp(last_bar)).days
    if age > STALE_BAR_WARN_DAYS:
        print(
            f"[shadow] WARNING: newest cache bar {last_bar} is {age} days old — "
            "recording stale weights. Run with --refresh-data after the close."
        )

    dedupe_key = (str(decision_bar), str(args.checkpoint))
    existing = {
        (str(r.get("decision_bar") or r.get("as_of")), str(r.get("checkpoint")))
        for r in _read_jsonl(ledger_path(run_id))
    }
    if dedupe_key in existing:
        print(f"[shadow] {run_id}: decision bar {decision_bar} ({args.checkpoint}) "
              "already recorded; skipping.")
        return

    # Observation-drift alarm: infer_weights emits the final VecNormalize-normalized
    # observation (z-scores vs frozen training stats). NOTE: values are clipped at
    # clip_obs (10), so max_abs_z saturates there; the >sigma fraction is unaffected
    # for sigma < 10. over_sigma_features gives attribution (indices into the obs).
    drift = None
    z = payload.pop("observation_normalized", None)
    if z is not None:
        z = np.asarray(z, dtype=np.float64)
        over = np.flatnonzero(np.abs(z) > args.drift_sigma)
        drift = {
            "max_abs_z_clipped": float(np.max(np.abs(z))),
            "frac_over_sigma": float(np.mean(np.abs(z) > args.drift_sigma)),
            "sigma": float(args.drift_sigma),
            "over_sigma_features": over.tolist()[:32],
            "alarm": obs_drift_alarm(z, threshold=args.drift_sigma, frac=args.drift_frac),
        }
        if drift["alarm"]:
            print(
                f"[shadow] ALARM: {drift['frac_over_sigma']:.1%} of features "
                f">{args.drift_sigma:g}σ from training stats — a regime the model "
                "never saw; treat today's weights with suspicion."
            )

    rec = {
        "run_id": run_id,
        "as_of": as_of,
        "decision_bar": decision_bar,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint,
        "target_weights": payload.get("target_weights"),
        "cash_weight": payload.get("cash_weight"),
        "provenance": prov,
        "obs_drift": drift,
        "realized": None,  # filled by reconcile
    }
    _append_jsonl(ledger_path(run_id), rec)
    print(f"[shadow] recorded {run_id} decision bar {decision_bar} → {ledger_path(run_id)}")


# ── reconcile + report (torch-free) ──────────────────────────────────────


def realized_portfolio_return(
    weights: dict[str, float],
    tickers: list[str],
    ohlcv: np.ndarray,
    t: int,
) -> float:
    """GROSS open[t+1] → open[t+2] return of the recorded book, where ``t`` is the
    DECISION bar (the bar whose observation produced the weights; fill at the next
    open, measured at the open after — no same-bar close needed). Costs are netted
    by the caller. Cash (missing tickers) earns 0, matching the env."""
    w = np.array([float(weights.get(tk, 0.0)) for tk in tickers], dtype=np.float64)
    open_fill = ohlcv[t + 1, :, 0]
    open_next = ohlcv[t + 2, :, 0]
    rets = open_next / np.maximum(open_fill, 1e-12) - 1.0
    return float(np.dot(w, rets))


def linear_costs(
    weights: dict[str, float],
    prev_weights: dict[str, float] | None,
    tickers: list[str],
    slippage: np.ndarray,
    tx_fee: np.ndarray,
    daily_holding: np.ndarray,
) -> float:
    """Linear cost approximation for one shadow day: turnover vs the previously
    recorded book × (slippage + fee), plus daily holding cost on the held book.
    No market impact / capacity model."""
    w = np.array([float(weights.get(tk, 0.0)) for tk in tickers], dtype=np.float64)
    w_prev = (
        np.array([float(prev_weights.get(tk, 0.0)) for tk in tickers], dtype=np.float64)
        if prev_weights
        else np.zeros_like(w)
    )
    turnover_cost = float(np.dot(np.abs(w - w_prev), slippage + tx_fee))
    holding = float(np.dot(w, daily_holding))
    return turnover_cost + holding


def cmd_reconcile(args: argparse.Namespace) -> None:
    run_id = args.run_id.strip()
    manifest = read_run_manifest(run_id) or {}
    _bind_run_config(run_id, args.use_current_config)
    cache_path = Path(args.data_cache) if args.data_cache.strip() else resolve_data_cache()
    (idx, ohlcv, _rsi, _macd, _macro, _fd, _fdm, _trend, _avol, _mvol, _live, cache_tickers) = (
        load_cache(str(cache_path))
    )
    tickers = resolve_panel_tickers(manifest, cache_tickers)
    if len(tickers) != ohlcv.shape[1]:
        raise SystemExit(
            f"panel width {ohlcv.shape[1]} != {len(tickers)} manifest tickers — "
            "refresh the cache for this run's universe before reconciling."
        )
    cfg = get_config()
    bench_w = cfg.reward.benchmark_cap_weights_array()
    if len(bench_w) != len(tickers):
        raise SystemExit(
            f"benchmark_cap_weights has {len(bench_w)} entries but the run trades "
            f"{len(tickers)} tickers — bind the run's config snapshot (drop "
            "--use-current-config) so the benchmark book matches the universe."
        )
    tc = cfg.transaction_costs
    slip, fee = tc.slippage_array(), tc.tx_fee_array()
    hold = tc.daily_holding_cost_array()
    bench_book = dict(zip(tickers, bench_w))
    date_to_t = {str(d.date()): i for i, d in enumerate(idx)}

    rows = _read_jsonl(ledger_path(run_id))
    already = {
        (str(r.get("decision_bar") or r.get("as_of")), str(r.get("checkpoint")))
        for r in _read_jsonl(reconciled_path(run_id))
    }
    prev_weights: dict[str, float] | None = None
    n_done = 0
    for rec in rows:
        dbar = str(rec.get("decision_bar") or rec.get("as_of"))
        key = (dbar, str(rec.get("checkpoint")))
        if key in already or rec.get("target_weights") is None:
            prev_weights = rec.get("target_weights") or prev_weights
            continue
        t = date_to_t.get(dbar)
        if t is None or t + 2 >= len(idx):
            continue  # next bars not in the cache yet
        gross = realized_portfolio_return(rec["target_weights"], tickers, ohlcv, t)
        cost = linear_costs(rec["target_weights"], prev_weights, tickers, slip, fee, hold)
        bench_gross = realized_portfolio_return(bench_book, tickers, ohlcv, t)
        bench_cost = float(np.dot(bench_w, hold))  # buy & hold: holding costs only
        _append_jsonl(
            reconciled_path(run_id),
            {
                **rec,
                "realized": {
                    "fill_bar": str(idx[t + 1].date()),
                    "measure_bar": str(idx[t + 2].date()),
                    "model_return_gross": gross,
                    "model_linear_costs": cost,
                    "model_return_net": gross - cost,
                    "benchmark_return_net": bench_gross - bench_cost,
                    "excess_return_net": (gross - cost) - (bench_gross - bench_cost),
                },
            },
        )
        prev_weights = rec.get("target_weights")
        n_done += 1
    print(f"[shadow] reconciled {n_done} row(s) → {reconciled_path(run_id)}")


def cmd_report(args: argparse.Namespace) -> None:
    run_id = args.run_id.strip()
    rows = [r for r in _read_jsonl(reconciled_path(run_id)) if r.get("realized")]
    if not rows:
        print(f"[shadow] no reconciled rows for {run_id}")
        return
    rets = np.array([r["realized"]["model_return_net"] for r in rows])
    exc = np.array([r["realized"]["excess_return_net"] for r in rows])
    ann = np.sqrt(252)
    print(f"[shadow] {run_id}: {len(rows)} reconciled day(s) "
          "(net of linear costs; no market impact/capacity model)")
    print(f"  cum return: {float(np.prod(1 + rets) - 1):+.2%}")
    print(f"  daily mean {rets.mean():+.4%}  std {rets.std():.4%}  "
          f"Sharpe(ann) {rets.mean() / (rets.std() + 1e-12) * ann:.2f}")
    print(f"  vs benchmark: mean excess {exc.mean():+.4%}  "
          f"IR(ann) {exc.mean() / (exc.std() + 1e-12) * ann:.2f}")
    alarms = sum(1 for r in rows if (r.get("obs_drift") or {}).get("alarm"))
    if alarms:
        print(f"  WARNING: {alarms} day(s) had observation-drift alarms")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in (("record", cmd_record), ("reconcile", cmd_reconcile), ("report", cmd_report)):
        sp = sub.add_parser(name)
        sp.add_argument("--run-id", required=True)
        sp.add_argument("--checkpoint", default="best", choices=("best", "final"))
        sp.add_argument("--data-cache", default="", metavar="PATH",
                        help="cache to read (default: the refreshable GLOBAL cache, "
                        "never the frozen run snapshot)")
        sp.add_argument("--use-current-config", action="store_true")
        if name == "record":
            sp.add_argument("--refresh-data", action="store_true",
                            help="fetch fresh bars into the global cache first")
            sp.add_argument("--drift-sigma", type=float, default=5.0)
            sp.add_argument("--drift-frac", type=float, default=0.05)
        sp.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
