"""Guards added after the Phase-0 adversarial review: the OOS window cross-check
(M8), crashed-variant retry (--overwrite-run plumbing), pre-read registry records
carrying no stale OOS metrics, and source-level pins for the train.py fixes that
need torch to execute (H6 pre-ramp DR pin, M7 sibling-first vecnorm). Torch-free."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlbot.run_artifacts import check_holdout_window_against_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── M8: resolved OOS window vs manifest record ───────────────────────────
def test_window_check_passes_on_exact_match() -> None:
    ch = {"date_start": "2016-01-04 00:00:00", "date_end": "2017-12-29 00:00:00"}
    assert (
        check_holdout_window_against_manifest("2016-01-04", "2017-12-29", ch) is None
    )


def test_window_check_hard_fails_on_silent_shift() -> None:
    ch = {"date_start": "2016-01-04", "date_end": "2017-12-29"}
    with pytest.raises(ValueError, match="OOS window mismatch"):
        check_holdout_window_against_manifest("2016-02-01", "2018-03-15", ch)


def test_window_check_warns_with_explicit_cli_override() -> None:
    ch = {"date_start": "2016-01-04", "date_end": "2017-12-29"}
    warn = check_holdout_window_against_manifest(
        "2018-01-02", "2019-12-31", ch, cli_override=True
    )
    assert warn is not None and "OOS window mismatch" in warn


def test_window_check_skips_manifests_without_recorded_dates() -> None:
    assert check_holdout_window_against_manifest("2016-01-04", "2017-12-29", {}) is None
    assert check_holdout_window_against_manifest("2016-01-04", "2017-12-29", None) is None
    # half-recorded (e.g. empty holdout at train time) → skip, don't crash
    assert (
        check_holdout_window_against_manifest(
            "2016-01-04", "2017-12-29", {"date_start": None, "date_end": None}
        )
        is None
    )


def test_backtest_wires_the_window_check() -> None:
    src = (PROJECT_ROOT / "scripts" / "backtest.py").read_text(encoding="utf-8")
    assert "check_holdout_window_against_manifest(" in src
    assert "cli_override=cli_override" in src


# ── crashed-variant retry: research.py must pass --overwrite-run ─────────
def test_research_retry_passes_overwrite_run() -> None:
    import scripts.research as research

    entry = {"config_path": "c.yaml", "run_id": "r", "seed": 1, "window": {}}
    spec = research.load_spec(PROJECT_ROOT / "specs" / "feature_split_ab.yaml")
    assert "--overwrite-run" not in research._train_cmd(entry, spec)
    assert "--overwrite-run" in research._train_cmd(entry, spec, overwrite_run=True)
    src = (PROJECT_ROOT / "scripts" / "research.py").read_text(encoding="utf-8")
    assert "overwrite_run=stale_run_dir" in src, (
        "launch loop must retry stale unscored run dirs with --overwrite-run"
    )
    sh = (PROJECT_ROOT / "scripts" / "run_seed_ensemble.sh").read_text(encoding="utf-8")
    assert "training_summary.json" in sh and "--overwrite-run" in sh


# ── pre-read registry records carry no stale OOS metrics ─────────────────
def test_oos_read_attempt_record_ignores_stale_backtest_summary(tmp_path, monkeypatch) -> None:
    import rlbot.run_artifacts as ra
    import scripts.research as research

    monkeypatch.setattr(ra, "RUNS_ROOT", tmp_path)
    rp = ra.RunPaths("v1", root=tmp_path)
    rp.mkdirs()
    (rp.run_meta_dir / "backtest_summary.json").write_text(
        json.dumps({"total_return": 0.42, "sharpe": 1.7}), encoding="utf-8"
    )
    cm = {"cohort": "c", "hypothesis": "h", "spec": {}, "variants": []}
    entry = {"run_id": "v1", "variant_id": "v1", "patch": {}, "seed": 1, "window": {}}
    monkeypatch.setattr(research, "RunPaths", lambda rid: ra.RunPaths(rid, root=tmp_path))
    monkeypatch.setattr(research, "read_run_manifest", lambda rid: None)
    rec_attempt = research._collect_one(cm, entry, tier=4, status="oos_read_attempt")
    assert rec_attempt.get("oos_total_return") is None
    assert rec_attempt.get("oos_sharpe") is None
    rec_scored = research._collect_one(cm, entry, tier=4)
    assert rec_scored.get("oos_total_return") == 0.42


# ── source pins for torch-gated train.py fixes ───────────────────────────
def test_train_pins_dr_bounds_before_fee_ramp_and_at_construction() -> None:
    src = (PROJECT_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    # H6: pre-ramp branch returns pinned (no-DR) bounds
    assert "return 1.0, 1.0, lag_fixed, lag_fixed" in src
    # first-episode pin: factory sets bounds before SB3's _setup_learn reset
    body = src.split("def _init():", 1)[1].split("return _init", 1)[0]
    assert "set_randomization_bounds(1.0, 1.0, _lag, _lag)" in body


def test_train_resume_prefers_sibling_vecnorm() -> None:
    src = (PROJECT_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    sib = src.find('sibling = resume_path.parent / "vec_normalize.pkl"')
    grand = src.find('resume_path.parent.parent / "vec_normalize.pkl"')
    assert sib != -1 and grand != -1 and sib < grand, (
        "resume must try the checkpoint-sibling vec_normalize.pkl before the run-level one"
    )
