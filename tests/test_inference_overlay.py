"""Written inference overlays (vol-target, NAV trend, EW blend) — torch-free."""

from __future__ import annotations

import numpy as np
import pytest

from rlbot.inference_overlay import (
    CASH_KILL,
    InferenceOverlay,
    OverlaySpec,
    blend_with_equal_weight,
    default_overlay_grid,
    overlay_kill_reasons,
    parse_overlay_names,
    spec_from_cli,
)
from rlbot.run_artifacts import discover_ensemble_run_ids, walk_forward_group_ids


def test_ew_blend_halfway_to_equal_weight() -> None:
    n = 4
    w = np.array([0.0, 0.40, 0.40, 0.10, 0.10], dtype=np.float64)
    out = blend_with_equal_weight(w, 0.5)
    assert out.sum() == pytest.approx(1.0)
    # EW is 0.25 each; mix = 0.5 * 809 + 0.5 * EW.
    assert out[1] == pytest.approx(0.5 * 0.40 + 0.5 * 0.25)
    assert out[0] == pytest.approx(0.0, abs=1e-9)


def test_vol_target_cuts_gross_when_realized_vol_is_high() -> None:
    spec = OverlaySpec(name="vt", vol_target=0.10, vol_window=30, max_single_asset_weight=0.50)
    ov = InferenceOverlay(spec)
    rng = np.random.default_rng(0)
    nav = 100_000.0
    # ~40% ann. vol daily shocks, fully invested book.
    w = np.array([0.0, 0.5, 0.5], dtype=np.float64)
    last = None
    for i in range(80):
        nav *= float(np.exp(rng.normal(0.0, 0.025)))
        last = ov.apply(w.copy(), nav=nav)
    assert last is not None
    assert float(np.sum(last[1:])) < 0.99
    assert last.sum() == pytest.approx(1.0, abs=1e-6)
    assert last[0] > 0.0


def test_trend_gate_only_cuts_when_fast_sma_is_below_slow() -> None:
    spec = OverlaySpec(
        name="tr",
        trend_fast=5,
        trend_slow=20,
        trend_min_gross=0.50,
        max_single_asset_weight=0.50,
    )
    ov = InferenceOverlay(spec)
    w = np.array([0.0, 0.5, 0.5], dtype=np.float64)
    # Rising NAV: no cut.
    for i in range(25):
        out = ov.apply(w.copy(), nav=100_000.0 * (1.0 + 0.01 * i))
    assert float(np.sum(out[1:])) == pytest.approx(1.0, abs=1e-6)

    ov.reset()
    # Falling NAV: cut gross to the floor.
    for i in range(25):
        out = ov.apply(w.copy(), nav=100_000.0 * (1.0 - 0.01 * i))
    assert float(np.sum(out[1:])) == pytest.approx(0.50, abs=1e-6)
    assert out[0] == pytest.approx(0.50, abs=1e-6)


def test_overlay_kill_reasons() -> None:
    assert overlay_kill_reasons(
        baseline_return=0.50, overlay_return=0.48, mean_cash=0.10
    ) == []
    cash = overlay_kill_reasons(
        baseline_return=0.50, overlay_return=0.48, mean_cash=CASH_KILL
    )
    assert any("mean_cash" in r for r in cash)
    collapsed = overlay_kill_reasons(
        baseline_return=0.50, overlay_return=0.40, mean_cash=0.10
    )
    assert any("collapsed" in r for r in collapsed)


def test_parse_overlay_cli_and_grid() -> None:
    assert parse_overlay_names("vol_target,trend,ew_blend") == [
        "vol_target",
        "trend",
        "ew_blend",
    ]
    spec = spec_from_cli(names=["vol_target", "ew_blend"], ew_alpha=0.5)
    assert spec is not None and spec.enabled()
    assert spec.vol_target == pytest.approx(0.11)
    assert spec.ew_alpha == pytest.approx(0.5)
    grid = default_overlay_grid()
    names = {s.name for s in grid}
    assert "vol_target" in names
    assert "ew_blend_0.25" in names
    assert "vol+trend+ew_0.50" in names
    with pytest.raises(ValueError):
        parse_overlay_names("crash_taper")


def test_walk_forward_group_ids_and_ensemble_discovery(tmp_path, monkeypatch) -> None:
    assert walk_forward_group_ids("809") == [
        "W1_809",
        "W2_809",
        "W3_809",
        "W4_809",
        "W5_809",
    ]
    import rlbot.run_artifacts as ra

    runs = tmp_path / "Runs"
    for name in ("W1_809", "W1_809_s42", "W1_809_s101", "W1_809_seed_7", "other"):
        (runs / name / "models").mkdir(parents=True)
    monkeypatch.setattr(ra, "RUNS_ROOT", runs)
    monkeypatch.setattr(ra, "PROJECT_ROOT", tmp_path)
    ids = discover_ensemble_run_ids("W1_809")
    assert ids == ["W1_809", "W1_809_seed_7", "W1_809_s42", "W1_809_s101"]
    only = discover_ensemble_run_ids("W1_809", seeds=[0, 42, 101])
    assert only == ["W1_809", "W1_809_s42", "W1_809_s101"]
