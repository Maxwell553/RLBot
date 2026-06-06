"""Run-local config + data-cache binding (P0-2). Torch-free: exercises the resolver
helpers in rlbot.run_artifacts and the config snapshot roundtrip in rlbot.rl_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlbot.rl_config import get_config, load_config, write_config_snapshot
from rlbot.run_artifacts import RunPaths, resolve_run_data_cache


def test_resolver_prefers_run_snapshot(tmp_path: Path) -> None:
    rp = RunPaths("r1", root=tmp_path)
    rp.mkdirs()
    snap = rp.data_snapshot
    snap.write_bytes(b"snapshot")
    default = tmp_path / "global.npz"
    default.write_bytes(b"global")
    assert resolve_run_data_cache("r1", root=tmp_path, default=default) == snap


def test_resolver_override_wins(tmp_path: Path) -> None:
    rp = RunPaths("r2", root=tmp_path)
    rp.mkdirs()
    rp.data_snapshot.write_bytes(b"snapshot")
    override = tmp_path / "override.npz"
    override.write_bytes(b"override")
    got = resolve_run_data_cache(
        "r2", override=str(override), root=tmp_path, default=tmp_path / "g.npz"
    )
    assert got == override


def test_resolver_falls_back_to_default(tmp_path: Path) -> None:
    default = tmp_path / "global.npz"
    got = resolve_run_data_cache("missing", root=tmp_path, default=default)
    assert got == default


def test_resolver_missing_override_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_run_data_cache(
            "r3", override=str(tmp_path / "nope.npz"), root=tmp_path, default=None
        )


def test_config_snapshot_roundtrip(tmp_path: Path) -> None:
    """A run config snapshot reloads to the same effective values (run-local binding)."""
    rp = RunPaths("c1", root=tmp_path)
    rp.mkdirs()
    cfg = get_config()
    write_config_snapshot(cfg, rp.config_snapshot)
    assert rp.config_snapshot.is_file()
    loaded = load_config(rp.config_snapshot)
    assert (
        loaded.environment.max_single_asset_weight
        == cfg.environment.max_single_asset_weight
    )
    assert loaded.data.feature_split_mode == cfg.data.feature_split_mode
    assert loaded.universe.tickers == cfg.universe.tickers
