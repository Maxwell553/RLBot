"""Training plot score-delta and milestone helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rlbot.visualize import (
    plot_training_progress,
    resolve_eval_plot_milestones,
    robust_score_delta_from_best,
    robust_score_delta_post_gate,
)


def test_robust_score_delta_from_best() -> None:
    scores = np.array([-20_000.0, -18_000.0, -22_000.0, -19_000.0])
    delta = robust_score_delta_from_best(scores)
    assert delta[0] == pytest.approx(0.0)
    assert delta[1] == pytest.approx(0.0)  # new running best
    assert delta[2] == pytest.approx(-4_000.0)
    assert delta[3] == pytest.approx(-1_000.0)


def test_resolve_eval_plot_milestones_from_manifest(tmp_path: Path) -> None:
    run = tmp_path / "W1_test"
    (run / "eval_logs").mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    (run / "config.yaml").write_text(
        (root / "config" / "config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    ts = np.array([500_000, 29_250_000, 30_000_000], dtype=np.int64)
    scores = np.array([-25_000.0, -20_000.0, -21_000.0])
    np.savez_compressed(
        run / "eval_logs" / "eval_nav_history.npz",
        timesteps=ts,
        mean_ending_nav=np.array([100_000.0, 101_000.0, 100_500.0]),
        robust_scores=scores,
    )
    (run / "manifest.json").write_text(
        '{"args": {"timesteps": 50000000}, "best_eval_step": 30000000}',
        encoding="utf-8",
    )
    hist = {
        "timesteps": ts,
        "robust_scores": scores,
        "mean_ending_nav": np.array([100_000.0, 101_000.0, 100_500.0]),
    }
    gate, best = resolve_eval_plot_milestones(hist, run_dir=run)
    assert best == 30_000_000
    assert gate == 29_250_000


def test_robust_score_delta_post_gate_resets_at_fee_ramp() -> None:
    ts = np.array([1_000_000, 22_000_000, 29_250_000, 30_000_000, 35_000_000], dtype=np.int64)
    scores = np.array([-15_000.0, -10_000.0, -18_000.0, -14_000.0, -20_000.0])
    pre, delta, post = robust_score_delta_post_gate(scores, ts, 29_250_000)
    assert pre.sum() == 2
    assert post.sum() == 3
    assert np.isnan(delta[0])
    assert np.isnan(delta[1])
    assert delta[2] == pytest.approx(0.0)  # first post-gate eval
    assert delta[3] == pytest.approx(0.0)  # new post-gate best
    assert delta[4] == pytest.approx(-6_000.0)


def test_robust_score_delta_post_gate_no_gate_falls_back() -> None:
    scores = np.array([-20_000.0, -18_000.0])
    ts = np.array([1_000_000, 2_000_000], dtype=np.int64)
    pre, delta, post = robust_score_delta_post_gate(scores, ts, 0)
    assert not pre.any()
    assert post.all()
    np.testing.assert_allclose(delta, robust_score_delta_from_best(scores))


def test_plot_training_progress_score_delta_panel(tmp_path: Path) -> None:
    ts = np.array([1_000_000, 2_000_000, 29_250_000, 30_000_000], dtype=np.int64)
    nav = np.array([102_000.0, 98_000.0, 103_000.0, 101_000.0])
    score = np.array([-22_000.0, -20_000.0, -18_000.0, -21_000.0])
    out = plot_training_progress(
        [100, 200],
        [-1.0, -0.5],
        eval_timesteps=ts,
        eval_ending_navs=nav,
        eval_robust_scores=score,
        best_model_min_step=29_250_000,
        best_eval_step=30_000_000,
        save_path=tmp_path / "training.png",
    )
    assert out.is_file()
    assert out.stat().st_size > 0
