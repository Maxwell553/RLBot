"""Phase B: global holdout-burn ledger, per-window budgets, selection-aware
significance (PSR/DSR), the success-gate engine, and the W6 embargo. Torch-free."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rlbot.research import oos_ledger
from rlbot.research.gates import evaluate_success_gates
from rlbot.research.spec import EMBARGOED_WINDOWS, normalize_window
from rlbot.stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe_null,
    probabilistic_sharpe_ratio,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── ledger ────────────────────────────────────────────────────────────────
def test_ledger_counts_distinct_models_not_raw_reads(tmp_path: Path) -> None:
    for _ in range(3):  # re-reading the SAME model adds no selection pressure
        oos_ledger.record_oos_read(
            run_id="m1", holdout_start="2022-01-03", holdout_end="2023-12-29",
            checkpoint="best", context="manual", root=tmp_path,
        )
    oos_ledger.record_oos_read(
        run_id="m2", holdout_start="2022-01-03", holdout_end="2023-12-29",
        checkpoint="best", context="research:c", root=tmp_path,
    )
    records = oos_ledger.read_ledger(tmp_path)
    assert len(records) == 4  # raw reads all recorded
    w = oos_ledger.window_key("2022-01-03", "2023-12-29")
    assert oos_ledger.distinct_models_for_window(records, w) == {"m1", "m2"}
    assert oos_ledger.trials_for_window(w, tmp_path) == 2


def test_ledger_window_budget_blocks_new_models_not_rereads(tmp_path: Path) -> None:
    w = oos_ledger.window_key("2022-01-03", "2023-12-29")
    for i in range(3):
        oos_ledger.record_oos_read(
            run_id=f"m{i}", holdout_start="2022-01-03", holdout_end="2023-12-29",
            root=tmp_path,
        )
    records = oos_ledger.read_ledger(tmp_path)
    # re-reading an already-burned model is free
    oos_ledger.assert_window_budget(records, w, ["m0"], budget=3)
    # a NEW model over budget is refused
    with pytest.raises(PermissionError, match="budget exhausted"):
        oos_ledger.assert_window_budget(records, w, ["m9"], budget=3)
    oos_ledger.assert_window_budget(records, w, ["m9"], budget=4)


def test_ledger_window_key_normalizes_timestamps() -> None:
    assert (
        oos_ledger.window_key("2022-01-03 00:00:00", "2023-12-29")
        == "2022-01-03..2023-12-29"
    )


def test_backtest_records_ledger_read_before_rollout() -> None:
    src = (PROJECT_ROOT / "scripts" / "backtest.py").read_text(encoding="utf-8")
    record_pos = src.find("oos_ledger.record_oos_read(")
    rollout_pos = src.find("rollout_policy_on_slice(", src.find("def run_oos_backtest"))
    assert record_pos != -1 and rollout_pos != -1
    assert record_pos < rollout_pos, "ledger read must be recorded BEFORE the rollout"


# ── significance ─────────────────────────────────────────────────────────
def test_dsr_decreases_with_trials_and_psr_bounds() -> None:
    d = [deflated_sharpe_ratio(1.5, 504, n) for n in (1, 5, 25, 125)]
    assert all(a > b for a, b in zip(d, d[1:])), d
    assert all(0.0 <= x <= 1.0 for x in d)
    assert expected_max_sharpe_null(1, 504) == 0.0
    assert expected_max_sharpe_null(100, 504) > expected_max_sharpe_null(10, 504) > 0.0
    # PSR of a strongly positive daily SR over two years ~ 1; of zero SR ~ 0.5
    assert probabilistic_sharpe_ratio(0.2, 0.0, 504) > 0.99
    assert probabilistic_sharpe_ratio(0.0, 0.0, 504) == pytest.approx(0.5)


def test_dsr_negative_skew_fat_tails_reduce_significance() -> None:
    # Moment corrections widen the SR estimator's variance; when the strategy is
    # ABOVE the selection benchmark (positive z), that lowers significance.
    base = deflated_sharpe_ratio(2.5, 504, 10, skew=0.0, kurt=3.0)
    skewed = deflated_sharpe_ratio(2.5, 504, 10, skew=-1.5, kurt=8.0)
    assert base > 0.5  # premise: above the deflation benchmark
    assert skewed < base


# ── success gates ────────────────────────────────────────────────────────
def _row(seed: int, nav: float, tier: int = 3, **kw) -> dict:
    return {"status": "ok", "seed": seed, "best_eval_nav": nav,
            "evaluation_tier": tier, **kw}


def test_success_gates_pass_fail_inconclusive() -> None:
    gates_cfg = {"min_seeds": 2, "eval_nav_mean_min": 100.0}
    assert evaluate_success_gates(gates_cfg, [_row(1, 120), _row(2, 90)])["verdict"] == "pass"
    assert evaluate_success_gates(gates_cfg, [_row(1, 50), _row(2, 60)])["verdict"] == "fail"
    assert evaluate_success_gates(gates_cfg, [_row(1, 120)])["verdict"] == "inconclusive"


def test_success_gates_oos_keys_need_tier4_evidence() -> None:
    cfg = {"deflated_sharpe_min": 0.95}
    assert evaluate_success_gates(cfg, [_row(1, 120, tier=3)])["verdict"] == "inconclusive"
    ok = evaluate_success_gates(
        cfg, [_row(1, 120, tier=4, oos_deflated_sharpe=0.97)]
    )
    assert ok["verdict"] == "pass"
    bad = evaluate_success_gates(
        cfg, [_row(1, 120, tier=4, oos_deflated_sharpe=0.3)]
    )
    assert bad["verdict"] == "fail"


def test_success_gates_reject_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown success_gates"):
        evaluate_success_gates({"sharpe_min_typo": 1.0}, [])


def test_success_gates_ignore_unscored_rows() -> None:
    rows = [_row(1, 120), {"status": "failed", "seed": 2, "best_eval_nav": 0.0,
                           "evaluation_tier": 3}]
    v = evaluate_success_gates({"eval_nav_mean_min": 100.0}, rows)
    assert v["verdict"] == "pass"
    assert v["checks"]["eval_nav_mean_min"]["observed"] == 120.0


# ── W6 embargo ───────────────────────────────────────────────────────────
def test_embargoed_window_rejected_by_name_and_dates() -> None:
    assert "W6" in EMBARGOED_WINDOWS
    with pytest.raises(PermissionError, match="EMBARGOED"):
        normalize_window({"name": "W6"})
    with pytest.raises(PermissionError, match="EMBARGOED"):
        normalize_window(
            {"train_end": "2025-12-31", "holdout_start": "2026-01-01",
             "holdout_end": "2027-12-31"}
        )
    assert normalize_window({"name": "W5"})["name"] == "W5"


# ── research wiring (source pins; promote/launch need subprocess harness) ──
def test_research_wires_ledger_budget_and_gate_verdicts() -> None:
    src = (PROJECT_ROOT / "scripts" / "research.py").read_text(encoding="utf-8")
    assert "oos_ledger.assert_window_budget(" in src
    assert src.count("oos_ledger.assert_window_budget(") >= 2  # launch AND promote
    assert "evaluate_success_gates" in src or "_evaluate_cohort_gates" in src
    assert "RLBOT_OOS_CONTEXT" in src
    assert "force_gates" in src
