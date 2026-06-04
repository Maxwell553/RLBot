"""Run artifact path layout."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rlbot.run_artifacts import RunPaths, _run_exists, new_run_id, write_manifest


def test_mkdirs_creates_runs_tree(tmp_path: Path) -> None:
    rp = RunPaths(run_id="T1", root=tmp_path)
    rp.mkdirs()
    assert (tmp_path / "Runs" / "T1" / "models" / "best").is_dir()
    assert (tmp_path / "Runs" / "T1" / "plots").is_dir()
    assert (tmp_path / "Runs" / "T1" / "eval_logs").is_dir()


def test_legacy_models_dir_fallback(tmp_path: Path) -> None:
    legacy = tmp_path / "models" / "W1"
    legacy.mkdir(parents=True)
    (legacy / "best_model.zip").touch()
    rp = RunPaths(run_id="W1", root=tmp_path)
    assert rp.models_dir == legacy


def test_new_layout_preferred_over_legacy(tmp_path: Path) -> None:
    new_models = tmp_path / "Runs" / "W2" / "models"
    new_models.mkdir(parents=True)
    legacy = tmp_path / "models" / "W2"
    legacy.mkdir(parents=True)
    rp = RunPaths(run_id="W2", root=tmp_path)
    assert rp.models_dir == new_models


def test_run_exists_when_run_dir_present(tmp_path: Path) -> None:
    (tmp_path / "Runs" / "W9").mkdir(parents=True)
    assert _run_exists("W9", tmp_path)
    assert not _run_exists("missing", tmp_path)


def test_new_run_id_format_and_duplicates(tmp_path: Path) -> None:
    when = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)
    assert new_run_id(1, root=tmp_path, when=when) == "W1_604"
    (tmp_path / "Runs" / "W1_604").mkdir(parents=True)
    assert new_run_id(1, root=tmp_path, when=when) == "W1_604_a"
    (tmp_path / "Runs" / "W1_604_a").mkdir(parents=True)
    assert new_run_id(1, root=tmp_path, when=when) == "W1_604_b"
