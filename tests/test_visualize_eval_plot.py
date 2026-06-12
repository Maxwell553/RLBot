"""Training plot eval panel (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rlbot.visualize import load_eval_history_npz, plot_training_progress


def test_plot_training_progress_eval_panel(tmp_path: Path) -> None:
    ts = np.array([1_000_000, 2_000_000], dtype=np.int64)
    nav = np.array([102_000.0, 98_000.0])
    std = np.array([2_000.0, 3_000.0])
    dd = np.array([1_500.0, 2_500.0])
    score = nav - 0.75 * std - dd
    dd_frac = np.array([0.015, 0.025])
    np.savez_compressed(
        tmp_path / "eval_nav_history.npz",
        timesteps=ts,
        mean_ending_nav=nav,
        std_ending_nav=std,
        robust_scores=score,
        mean_max_drawdown_nav=dd,
        mean_max_drawdown_frac=dd_frac,
    )
    hist = load_eval_history_npz(tmp_path / "eval_nav_history.npz")
    assert hist is not None
    assert "robust_scores" in hist
    assert "mean_max_drawdown_pct" in hist
    assert hist["mean_max_drawdown_pct"][0] == pytest.approx(1.5)
    out = plot_training_progress(
        [100, 200],
        [-1.0, -0.5],
        eval_timesteps=hist["timesteps"],
        eval_ending_navs=hist["mean_ending_nav"],
        eval_std_navs=hist["std_ending_nav"],
        eval_robust_scores=hist["robust_scores"],
        eval_mean_max_dd_pct=hist["mean_max_drawdown_pct"],
        save_path=tmp_path / "training.png",
    )
    assert out.is_file()
    assert out.stat().st_size > 0
