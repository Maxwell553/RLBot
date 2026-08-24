"""Lite /api/runs must surface on-disk backtest_summary.json on the list payload."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path


def _load_lite():
    spec = importlib.util.spec_from_file_location(
        "frontend_api_lite_runs_test",
        Path(__file__).resolve().parents[1] / "scripts" / "frontend_api_lite.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub_row(run_id: str, **kwargs):
    row = {
        "run_id": run_id,
        "window": run_id.split("_")[0],
        "training_status": "completed",
        "progress_pct": 100.0,
        "elapsed_timesteps": 50_000_000,
        "nominal_timesteps": 50_000_000,
        "best_eval_step": None,
        "best_eval_score": None,
        "curriculum_stage_at_best": None,
        "early_stop_reason": None,
        "started_at_utc": None,
        "finished_at_utc": None,
        "oos_sharpe": None,
        "oos_deflated_sharpe": None,
        "oos_return": None,
        "oos_max_drawdown": None,
        "ew_excess_return": None,
        "has_backtest": False,
        "labels": [],
        "warnings": [],
        "comparable": True,
        "git_dirty": None,
    }
    row.update(kwargs)
    return row


def _write_backtest(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "training_status": "completed",
                "nominal_timesteps": 50_000_000,
                "elapsed_timesteps": 50_000_000,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "backtest_summary.json").write_text(
        json.dumps(
            {
                "sharpe": 1.596,
                "deflated_sharpe": 0.486,
                "total_return": 0.338,
                "max_drawdown": -0.052,
                "checkpoint_label": "best",
                "excess_return_vs_equal_weight": 0.054,
            }
        ),
        encoding="utf-8",
    )


def test_runs_page_loads_backtest_oos_by_default(tmp_path: Path) -> None:
    runs_root = tmp_path / "Runs"
    _write_backtest(runs_root / "W1_817")
    exec_dir = tmp_path / "execution"
    exec_dir.mkdir()

    mod = _load_lite()
    mod.ROOT = tmp_path
    mod.RUNS = runs_root
    mod.EXEC = exec_dir
    mod.RUNS_CACHE = exec_dir / "api_runs_cache.json"
    mod._rows_cache = [_stub_row("W1_817")]
    mod._rows_at = time.monotonic()

    page = mod._runs_page(offset=0, limit=40, search="817")
    assert page["runs"], "expected the completed 817 row on the list"
    row = page["runs"][0]
    assert row["run_id"] == "W1_817"
    assert abs(float(row["oos_sharpe"]) - 1.596) < 1e-9
    assert abs(float(row["oos_deflated_sharpe"]) - 0.486) < 1e-9
    assert row["has_backtest"] is True
    assert page["counts"]["with_backtest"] == 1


def test_fill_rows_oos_prefers_newest_missing_completed(tmp_path: Path) -> None:
    runs_root = tmp_path / "Runs"
    _write_backtest(runs_root / "W1_817")
    mod = _load_lite()
    mod.ROOT = tmp_path
    mod.RUNS = runs_root
    stale = _stub_row("W1_612")  # no backtest file
    fresh = _stub_row("W1_817")
    filled = mod._fill_rows_oos([stale, fresh], budget_s=5.0)
    by_id = {r["run_id"]: r for r in filled}
    assert by_id["W1_817"]["oos_sharpe"] is not None
    assert by_id["W1_612"]["oos_sharpe"] is None
