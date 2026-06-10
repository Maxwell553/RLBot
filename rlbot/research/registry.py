"""Append-only JSONL run registry. One record per train+backtest; aggregated from the
manifest / training_summary / backtest_summary already written by the pipeline (this is
an aggregator, not a new metric source)."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

try:  # POSIX only; on other platforms locking degrades to best-effort appends
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None  # type: ignore[assignment]


@contextmanager
def registry_lock(path: str | Path) -> Iterator[None]:
    """Exclusive advisory lock for read-gate-append sequences.

    The per-write flock in ``append_record`` only serializes bytes; the no-repeat-OOS
    gate needs *check-then-append* atomicity (two concurrent promotes of the same
    variant must not both pass the gate). Hold this around read → assert → append.
    """
    p = Path(str(path) + ".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+") as lf:
        if _fcntl is not None:
            _fcntl.flock(lf.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            if _fcntl is not None:
                _fcntl.flock(lf.fileno(), _fcntl.LOCK_UN)


def append_record(path: str | Path, record: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), default=str) + "\n"
    with p.open("a", encoding="utf-8") as f:
        locked = False
        if _fcntl is not None:
            try:  # serialize the write so two appends cannot interleave bytes
                _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
                locked = True
            except OSError:
                pass
        try:
            f.write(line)
            f.flush()
        finally:
            if locked:
                _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)


def read_records(path: str | Path, *, on_corrupt: str = "skip") -> list[dict]:
    """Parse the JSONL registry.

    ``on_corrupt="skip"`` (default, for collect/report): torn/corrupt lines are
    skipped with a loud warning so a crash's half-written tail cannot brick
    reporting. ``on_corrupt="raise"`` (for OOS gates): any corrupt line raises —
    the no-repeat-holdout gate must FAIL CLOSED, because a skipped scored record
    would silently permit a repeat read.
    """
    p = Path(path)
    if not p.is_file():
        return []
    out: list[dict] = []
    bad = 0
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if on_corrupt == "raise":
                raise ValueError(
                    f"Corrupt registry line {i} in {p}. The OOS gate fails closed on "
                    "corruption — inspect/repair the registry (each line is one JSON "
                    "record) before retrying."
                )
            bad += 1
            print(
                f"[registry] WARNING: skipping corrupt line {i} in {p} "
                "(torn write from a crash?)",
                file=sys.stderr,
            )
    if bad:
        print(
            f"[registry] WARNING: {bad} corrupt line(s) in {p} skipped — reports may "
            "undercount; OOS gates re-read strictly and will refuse.",
            file=sys.stderr,
        )
    return out


def build_record(
    *,
    cohort: str,
    variant_id: str,
    group_id: str | None = None,
    hypothesis: str,
    run_id: str,
    evaluation_tier: int,
    manifest: Mapping[str, Any] | None = None,
    training_summary: Mapping[str, Any] | None = None,
    backtest_summary: Mapping[str, Any] | None = None,
    status: str = "ok",
    failure: str | None = None,
    patch: Mapping[str, Any] | None = None,
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
        "group_id": group_id,
        "patch": dict(patch) if patch else None,
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
        "oos_window": backtest_summary.get("oos_window"),
        "oos_trials_for_window": backtest_summary.get("oos_trials_for_window"),
        "oos_deflated_sharpe": backtest_summary.get("deflated_sharpe"),
        "benchmark_spy": (detailed.get("benchmark_spy") or {}).get("sharpe"),
    }
