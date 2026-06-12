"""Robust eval selection score and in-training portfolio diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlbot.eval_selection import (
    EvalBenchmarkContext,
    aggregate_eval_portfolio_diagnostics,
    append_eval_diagnostics_jsonl,
    compute_robust_eval_score,
    compute_stitched_eval_metrics,
    exposure_risk_penalty,
)


def test_compute_robust_eval_score_stitched_blend() -> None:
    """Blended return signal mixes segment-mean excess with stitched excess."""
    pd = pytest.importorskip("pandas")
    episodes = [
        {
            "start_bar": 10,
            "nav_path": [100_000.0, 110_000.0],
            "ending_nav": 110_000.0,
            "max_drawdown_nav": 0.0,
            "max_drawdown_frac": 0.0,
        },
        {
            "start_bar": 20,
            "nav_path": [100_000.0, 90_000.0],
            "ending_nav": 90_000.0,
            "max_drawdown_nav": 1_000.0,
            "max_drawdown_frac": 0.01,
        },
    ]
    ctx = EvalBenchmarkContext(
        ohlcv=np.full((30, 10, 5), 100.0),
        idx=pd.date_range("2020-01-01", periods=30),
        tickers=[f"A{i}" for i in range(10)],
        mode="equal_weight_daily",
    )
    stitched_only = compute_robust_eval_score(
        episodes, benchmark_ctx=ctx, stitched_blend=1.0, std_coef=0.0, dd_coef=0.0
    )
    mean_only = compute_robust_eval_score(
        episodes, benchmark_ctx=ctx, stitched_blend=0.0, std_coef=0.0, dd_coef=0.0
    )
    blended = compute_robust_eval_score(
        episodes, benchmark_ctx=ctx, stitched_blend=0.5, std_coef=0.0, dd_coef=0.0
    )
    assert mean_only["return_signal"] == pytest.approx(mean_only["mean_excess_nav"])
    assert stitched_only["return_signal"] == pytest.approx(stitched_only["stitched_excess_nav"])
    assert blended["return_signal"] == pytest.approx(
        0.5 * mean_only["mean_excess_nav"] + 0.5 * stitched_only["stitched_excess_nav"]
    )


def test_compute_robust_eval_score_prefers_stable_excess() -> None:
    stable = [{"ending_nav": 100_000.0, "max_drawdown_nav": 2_000.0, "nav_path": [100_000.0, 100_000.0]}] * 4
    volatile = [
        {"ending_nav": 120_000.0, "max_drawdown_nav": 2_000.0, "nav_path": [100_000.0, 120_000.0]},
        {"ending_nav": 80_000.0, "max_drawdown_nav": 2_000.0, "nav_path": [100_000.0, 80_000.0]},
        {"ending_nav": 120_000.0, "max_drawdown_nav": 2_000.0, "nav_path": [100_000.0, 120_000.0]},
        {"ending_nav": 80_000.0, "max_drawdown_nav": 2_000.0, "nav_path": [100_000.0, 80_000.0]},
    ]
    s_stable = compute_robust_eval_score(stable, std_coef=0.75, dd_coef=2.0)
    s_volatile = compute_robust_eval_score(volatile, std_coef=0.75, dd_coef=2.0)
    assert s_stable["mean_ending_nav"] == pytest.approx(100_000.0)
    assert s_volatile["mean_ending_nav"] == pytest.approx(100_000.0)
    assert s_stable["score"] > s_volatile["score"]


def test_compute_robust_eval_score_penalizes_drawdown_p75() -> None:
    low_dd = [{"ending_nav": 100_000.0, "max_drawdown_nav": 1_000.0}] * 3
    high_dd = [{"ending_nav": 100_000.0, "max_drawdown_nav": 10_000.0}] * 3
    assert compute_robust_eval_score(low_dd, dd_coef=2.0)["score"] > compute_robust_eval_score(
        high_dd, dd_coef=2.0
    )["score"]
    assert compute_robust_eval_score(high_dd)["p75_max_drawdown_nav"] == pytest.approx(10_000.0)


def test_compute_stitched_eval_metrics_compounds_blocks() -> None:
    episodes = [
        {
            "start_bar": 10,
            "nav_path": [100_000.0, 110_000.0],
            "ending_nav": 110_000.0,
            "max_drawdown_nav": 0.0,
            "max_drawdown_frac": 0.0,
        },
        {
            "start_bar": 20,
            "nav_path": [100_000.0, 105_000.0],
            "ending_nav": 105_000.0,
            "max_drawdown_nav": 0.0,
            "max_drawdown_frac": 0.0,
        },
    ]
    out = compute_stitched_eval_metrics(episodes)
    assert out["stitched_agent_nav"] == pytest.approx(115_500.0)
    assert len(out["stitched_nav_path"]) == 3


def test_aggregate_eval_portfolio_diagnostics_includes_segments() -> None:
    episodes = [
        {
            "ending_nav": 105_000.0,
            "start_nav": 100_000.0,
            "start_bar": 5,
            "max_drawdown_frac": 0.02,
            "max_drawdown_nav": 2_000.0,
            "nav_path": [100_000.0, 102_000.0, 105_000.0],
            "weights": np.array([[0.1, 0.45, 0.45] + [0.0] * 8], dtype=np.float64),
        }
    ]
    tickers = [f"A{i}" for i in range(10)]
    out = aggregate_eval_portfolio_diagnostics(
        episodes, tickers=tickers, max_single_asset_weight=0.20
    )
    assert out["portfolio"]["mean_cash_frac"] == pytest.approx(0.1)
    assert out["portfolio"]["cap_hit_fraction"] == pytest.approx(1.0)
    assert len(out["segments"]) == 1
    assert out["segments"][0]["nav_path"] == [100_000.0, 102_000.0, 105_000.0]


def test_append_eval_diagnostics_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "eval_portfolio_diagnostics.jsonl"
    append_eval_diagnostics_jsonl(path, {"timestep": 1, "score": {"score": 1.0}})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["timestep"] == 1


def test_exposure_risk_penalty_modes() -> None:
    rets = np.array([0.01, -0.005, 0.008, 0.002], dtype=np.float64)
    rv = exposure_risk_penalty(
        gross_exposure=0.9, agent_returns=rets, vix=20.0, mode="realized_vol", scale=10.0
    )
    assert rv > 0.0
    low_vix = exposure_risk_penalty(
        gross_exposure=0.9, agent_returns=rets, vix=10.0, mode="vix_positive", scale=10.0
    )
    high_vix = exposure_risk_penalty(
        gross_exposure=0.9, agent_returns=rets, vix=30.0, mode="vix_positive", scale=10.0
    )
    assert low_vix == 0.0
    assert high_vix > low_vix
    huge = exposure_risk_penalty(
        gross_exposure=1.0, agent_returns=rets, vix=36.0, mode="vix_positive", scale=40.0
    )
    assert huge == pytest.approx(40.0, rel=1e-3)


def test_episode_end_nav_recorder_drawdown_from_episode_start() -> None:
    """Segment drawdown must include episode-start NAV, not only post-step NAVs."""
    import gymnasium as gym

    from rlbot.trading_env import EpisodeEndNavRecorder

    class _ImmediateLossEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self._episode_start_nav = 100_000.0
            self.initial_cash = 100_000.0
            self._t = 0
            self._step_i = 0
            self._navs = [95_000.0, 94_000.0]

        def reset(self, *, seed=None, options=None):
            self._step_i = 0
            self._t = 0
            self._episode_start_nav = 100_000.0
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            nav = float(self._navs[self._step_i])
            self._step_i += 1
            done = self._step_i >= len(self._navs)
            w = np.array([0.0, 1.0], dtype=np.float64)
            return (
                np.zeros(1, dtype=np.float32),
                0.0,
                done,
                False,
                {"nav": nav, "target_weights": w},
            )

    env = EpisodeEndNavRecorder(_ImmediateLossEnv())
    env.reset()
    while True:
        _, _, term, trunc, _ = env.step(np.zeros(1))
        if term or trunc:
            break
    eps = env.pop_eval_episodes()
    assert len(eps) == 1
    ep = eps[0]
    assert ep["start_nav"] == pytest.approx(100_000.0)
    assert ep["start_bar"] == 0
    assert ep["nav_path"][0] == pytest.approx(100_000.0)
    assert ep["max_drawdown_nav"] == pytest.approx(6_000.0)


def test_benchmark_nav_path_rejects_invalid_mode() -> None:
    ctx = EvalBenchmarkContext(
        ohlcv=np.zeros((10, 2, 5)),
        idx=pytest.importorskip("pandas").date_range("2020-01-01", periods=10),
        tickers=["SP500", "BOND10Y"],
        mode="spy_only",
    )
    ep = {"nav_path": [100_000.0, 101_000.0], "start_bar": 0}
    with pytest.raises(ValueError, match="balanced_6040"):
        from rlbot.eval_selection import benchmark_nav_path_for_episode

        benchmark_nav_path_for_episode(ep, ctx)
