"""Append-only JSONL run registry. One record per train+backtest; aggregated from the
manifest / training_summary / backtest_summary already written by the pipeline (this is
an aggregator, not a new metric source)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def append_record(path: str | Path, record: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), default=str) + "\n")


def read_records(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def build_record(
    *,
    cohort: str,
    variant_id: str,
    hypothesis: str,
    run_id: str,
    evaluation_tier: int,
    manifest: Mapping[str, Any] | None = None,
    training_summary: Mapping[str, Any] | None = None,
    backtest_summary: Mapping[str, Any] | None = None,
    status: str = "ok",
    failure: str | None = None,
) -> dict:
    """Flatten the run's artifacts into a single registry record."""
    manifest = manifest or {}
    training_summary = training_summary or {}
    backtest_summary = backtest_summary or {}
    uni = manifest.get("universe") or {}
    ch = manifest.get("chronological_holdout") or {}
    detailed = backtest_summary.get("detailed") or {}
    boot = detailed.get("bootstrap_sharpe") or {}
    return {
        "cohort": cohort,
        "variant_id": variant_id,
        "hypothesis": hypothesis,
        "run_id": run_id,
        "evaluation_tier": int(evaluation_tier),
        "status": status,
        "failure": failure,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # provenance
        "git_commit": manifest.get("git_commit") or training_summary.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
        "config_hash": manifest.get("config_hash") or backtest_summary.get("config_hash"),
        "data_cache_hash": manifest.get("data_cache_hash")
        or backtest_summary.get("data_cache_hash"),
        "feature_split_mode": manifest.get("feature_split_mode")
        or training_summary.get("feature_split_mode"),
        # universe / split
        "n_assets": uni.get("n_assets"),
        "tickers": uni.get("tickers"),
        "train_end": ch.get("train_end"),
        "holdout_start": ch.get("holdout_start"),
        "holdout_end": ch.get("holdout_end"),
        "seed": (manifest.get("args") or {}).get("seed"),
        "timesteps": training_summary.get("timesteps"),
        # training selection signal
        "best_eval_nav": training_summary.get("best_eval_nav")
        or manifest.get("best_eval_nav"),
        "best_eval_step": training_summary.get("best_eval_step")
        or manifest.get("best_eval_step"),
        "early_stop_reason": training_summary.get("early_stop_reason"),
        # OOS metrics (only meaningful when the tier permits)
        "checkpoint_label": backtest_summary.get("checkpoint_label"),
        "oos_total_return": backtest_summary.get("total_return"),
        "oos_sharpe": backtest_summary.get("sharpe"),
        "oos_max_drawdown": backtest_summary.get("max_drawdown"),
        "oos_sharpe_ci": [boot.get("p2_5"), boot.get("p50"), boot.get("p97_5")]
        if boot
        else None,
        "benchmark_spy": (detailed.get("benchmark_spy") or {}).get("sharpe"),
    }
