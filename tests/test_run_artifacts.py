"""Run artifact path layout."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rlbot.run_artifacts import (
    RunPaths,
    _run_exists,
    new_run_id,
    persist_dirty_source_snapshot,
    write_manifest,
)


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


def test_persist_dirty_archives_untracked_python(tmp_path: Path, monkeypatch) -> None:
    """Untracked *.py must be copied into provenance/untracked/, not merely listed."""
    import subprocess

    root = tmp_path / "repo"
    (root / "rlbot").mkdir(parents=True)
    (root / "rlbot" / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    untracked = root / "rlbot" / "new_module.py"
    untracked.write_text("y = 2\n", encoding="utf-8")
    (root / "notes.txt").write_text("skip me\n", encoding="utf-8")

    def fake_run(cmd, cwd=None, capture_output=False, text=False, timeout=None):
        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        if cmd[:3] == ["git", "diff", "HEAD"]:
            r = R()
            r.stdout = "diff --git a/rlbot/tracked.py b/rlbot/tracked.py\n"
            return r
        if cmd[:3] == ["git", "ls-files", "--others"]:
            r = R()
            r.stdout = "rlbot/new_module.py\nnotes.txt\n"
            return r
        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    dest = tmp_path / "provenance"
    frag = persist_dirty_source_snapshot(dest, root=root, git_dirty=True)
    assert frag["source_patch"] == "git.diff"
    assert "rlbot/new_module.py" in frag["untracked_archived"]
    assert frag["untracked_archive_dir"] == "provenance/untracked"
    archived = dest / "untracked" / "rlbot" / "new_module.py"
    assert archived.is_file()
    assert archived.read_text(encoding="utf-8") == "y = 2\n"
    # Non-Python untracked files are listed in the patch, not archived.
    assert not (dest / "untracked" / "notes.txt").exists()
    patch = (dest / "git.diff").read_text(encoding="utf-8")
    assert "[archived] rlbot/new_module.py" in patch
    assert "[listed-only] notes.txt" in patch
