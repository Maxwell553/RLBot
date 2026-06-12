"""Regenerate training plots from persisted run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlbot.visualize import (
    _mean_max_dd_frac_from_diagnostics_jsonl,
    load_eval_history_npz,
    regenerate_training_plot,
)


def test_load_eval_history_pct_from_diagnostics_jsonl(tmp_path: Path) -> None:
    ts = np.array([1_000_000, 2_000_000], dtype=np.int64)
    nav = np.array([100_000.0, 101_000.0])
    dd_nav = np.array([5_000.0, 4_000.0])
    np.savez_compressed(
        tmp_path / "eval_nav_history.npz",
        timesteps=ts,
        mean_ending_nav=nav,
        mean_max_drawdown_nav=dd_nav,
    )
    jsonl = tmp_path / "eval_portfolio_diagnostics.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"segments": [{"max_drawdown_frac": 0.05}]}),
                json.dumps({"segments": [{"max_drawdown_frac": 0.03}, {"max_drawdown_frac": 0.07}]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fracs = _mean_max_dd_frac_from_diagnostics_jsonl(jsonl)
    assert fracs is not None
    assert fracs[1] == pytest.approx(0.05)
    hist = load_eval_history_npz(tmp_path / "eval_nav_history.npz")
    assert hist is not None
    assert hist["mean_max_drawdown_pct"][0] == pytest.approx(5.0)
    assert hist["mean_max_drawdown_pct"][1] == pytest.approx(5.0)


def test_regenerate_training_plot(tmp_path: Path) -> None:
    run = tmp_path / "W1_test"
    (run / "plots").mkdir(parents=True)
    (run / "eval_logs").mkdir(parents=True)
    ts = np.array([500_000], dtype=np.int64)
    np.savez_compressed(
        run / "eval_logs" / "eval_nav_history.npz",
        timesteps=ts,
        mean_ending_nav=np.array([100_000.0]),
        std_ending_nav=np.array([0.0]),
        robust_scores=np.array([99_000.0]),
        mean_max_drawdown_frac=np.array([0.04]),
        mean_max_drawdown_nav=np.array([4_000.0]),
    )
    np.savez_compressed(
        run / "plots" / "training_episodes.npz",
        episode_ts=np.array([100_000], dtype=np.int64),
        episode_rewards=np.array([-1.0]),
        episode_lengths=np.array([50], dtype=np.int64),
        episode_navs=np.array([100_000.0]),
        episode_nav_ts=np.array([100_000], dtype=np.int64),
    )
    out = regenerate_training_plot(run)
    assert out is not None
    assert out.is_file()
