"""Tier-5 shadow trading: the only evaluation that never burns a holdout.

A daily loop records the model's audited target weights forward in time and
reconciles them against realized next-bar returns — true walk-forward evidence
accumulating one bar per day, with full provenance and zero broker risk.

    python scripts/shadow_trade.py record    --run-id <RUN_ID>   # after the close
    python scripts/shadow_trade.py reconcile --run-id <RUN_ID>   # any later day
    python scripts/shadow_trade.py report    --run-id <RUN_ID>

``record`` runs the audited inference path (scripts/infer_weights.py machinery —
frozen VecNormalize, manifest-checked panel, OOS-ledger logged) for the latest
cache bar and appends one row to ``execution/shadow_ledger_<RUN_ID>.jsonl``
(gitignored: live state never enters the research tree). It also raises an
observation-drift alarm when current features sit far outside the frozen
normalization stats — a cheap regime/staleness check.

``reconcile`` is torch-free: it fills in each pending row's realized open→open
portfolio return once the next bar exists in the cache, alongside the
cap-weighted benchmark for the same bar.

The ledger is append-only; reconciliation writes a sibling ``*_reconciled.jsonl``
so raw records are never mutated.
"""

from __future__ import annotations

import argparse
import json
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
    resolve_run_data_cache,
)

EXECUTION_DIR = PROJECT_ROOT / "execution"


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


# ── record (needs torch via the infer path) ───────────────────────────────


def obs_drift_alarm(z_scores: np.ndarray, threshold: float = 5.0, frac: float = 0.05) -> bool:
    """True when more than ``frac`` of features sit beyond ``threshold`` sigma of the
    frozen training normalization — the policy is seeing a regime its stats never
    covered, and its outputs should not be trusted blindly."""
    z = np.abs(np.asarray(z_scores, dtype=np.float64).reshape(-1))
    return bool(np.mean(z > threshold) > frac)


def cmd_record(args: argparse.Namespace) -> None:
    run_id = args.run_id.strip()
    manifest = read_run_manifest(run_id)
    if manifest is None:
        raise SystemExit(f"Missing Runs/{run_id}/manifest.json")
    _bind_run_config(run_id, args.use_current_config)

    # Reuse the audited inference path end-to-end (frozen VecNormalize, panel
    # compatibility assert, OOS-ledger record with context=infer_weights).
    import subprocess

    out_json = EXECUTION_DIR / f"_shadow_weights_{run_id}.json"
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "infer_weights.py"),
        "--run-id", run_id, "--checkpoint", args.checkpoint, "--out", str(out_json),
    ]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    as_of = payload.get("as_of") or (payload.get("provenance") or {}).get("as_of_last_bar")
    existing = {r.get("as_of") for r in _read_jsonl(ledger_path(run_id))}
    if as_of in existing:
        print(f"[shadow] {run_id}: as_of {as_of} already recorded; skipping.")
        return

    # Observation-drift alarm: infer_weights emits the final VecNormalize-normalized
    # observation (z-scores vs frozen training stats; clipped at clip_obs=10, so
    # the >5σ measure is unaffected by clipping).
    drift = None
    z = payload.get("observation_normalized")
    if z is not None:
        z = np.asarray(z, dtype=np.float64)
        drift = {
            "max_abs_z": float(np.max(np.abs(z))),
            "frac_over_5sigma": float(np.mean(np.abs(z) > 5.0)),
            "alarm": obs_drift_alarm(z),
        }
        if drift["alarm"]:
            print(
                f"[shadow] ALARM: {drift['frac_over_5sigma']:.1%} of features >5σ "
                "from training stats — a regime the model never saw; treat today's "
                "weights with suspicion."
            )

    rec = {
        "run_id": run_id,
        "as_of": as_of,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint,
        "target_weights": payload.get("target_weights"),
        "cash_weight": payload.get("cash_weight"),
        "provenance": payload.get("provenance"),
        "obs_drift": drift,
        "realized": None,  # filled by reconcile
    }
    _append_jsonl(ledger_path(run_id), rec)
    print(f"[shadow] recorded {run_id} @ {as_of} → {ledger_path(run_id)}")


# ── reconcile + report (torch-free) ──────────────────────────────────────


def realized_portfolio_return(
    weights: dict[str, float],
    tickers: list[str],
    ohlcv: np.ndarray,
    t: int,
) -> float:
    """Open[t+1] → open[t+2] return of the recorded target book (weights decided on
    bar t's close fill at the next open; the following open closes the measurement
    window without needing same-bar close data)."""
    w = np.array([float(weights.get(tk, 0.0)) for tk in tickers], dtype=np.float64)
    open_fill = ohlcv[t + 1, :, 0]
    open_next = ohlcv[t + 2, :, 0]
    rets = open_next / np.maximum(open_fill, 1e-12) - 1.0
    return float(np.dot(w, rets))


def cmd_reconcile(args: argparse.Namespace) -> None:
    run_id = args.run_id.strip()
    manifest = read_run_manifest(run_id) or {}
    _bind_run_config(run_id, args.use_current_config)
    cache_path = resolve_run_data_cache(run_id, args.data_cache, default=resolve_data_cache())
    (idx, ohlcv, _rsi, _macd, _macro, _fd, _fdm, _trend, _avol, _mvol, _live, cache_tickers) = (
        load_cache(str(cache_path))
    )
    tickers = resolve_panel_tickers(manifest, cache_tickers)
    bench_w = get_config().reward.benchmark_cap_weights_array()
    date_to_t = {str(d.date()): i for i, d in enumerate(idx)}

    rows = _read_jsonl(ledger_path(run_id))
    already = {r.get("as_of") for r in _read_jsonl(reconciled_path(run_id))}
    n_done = 0
    for rec in rows:
        as_of = rec.get("as_of")
        if as_of in already or rec.get("target_weights") is None:
            continue
        t = date_to_t.get(str(as_of))
        if t is None or t + 2 >= len(idx):
            continue  # next bars not in the cache yet
        model_ret = realized_portfolio_return(rec["target_weights"], tickers, ohlcv, t)
        bench_ret = realized_portfolio_return(
            dict(zip(tickers, bench_w)), tickers, ohlcv, t
        )
        _append_jsonl(
            reconciled_path(run_id),
            {
                **rec,
                "realized": {
                    "fill_bar": str(idx[t + 1].date()),
                    "measure_bar": str(idx[t + 2].date()),
                    "model_open_to_open_return": model_ret,
                    "benchmark_open_to_open_return": bench_ret,
                    "excess_return": model_ret - bench_ret,
                },
            },
        )
        n_done += 1
    print(f"[shadow] reconciled {n_done} row(s) → {reconciled_path(run_id)}")


def cmd_report(args: argparse.Namespace) -> None:
    run_id = args.run_id.strip()
    rows = [r for r in _read_jsonl(reconciled_path(run_id)) if r.get("realized")]
    if not rows:
        print(f"[shadow] no reconciled rows for {run_id}")
        return
    rets = np.array([r["realized"]["model_open_to_open_return"] for r in rows])
    exc = np.array([r["realized"]["excess_return"] for r in rows])
    ann = np.sqrt(252)
    print(f"[shadow] {run_id}: {len(rows)} reconciled day(s)")
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
        sp.add_argument("--data-cache", default="", metavar="PATH")
        sp.add_argument("--use-current-config", action="store_true")
        sp.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
