"""Robust eval selection score and in-training portfolio diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlbot.eval_selection import (
    EvalBenchmarkContext,
    aggregate_eval_portfolio_diagnostics,
    annualized_sharpe,
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


def test_annualized_sharpe_scales_and_clips() -> None:
    steady = np.full(64, 0.001)
    assert annualized_sharpe(steady) == pytest.approx(10.0)  # zero-vol → clipped
    noisy = np.array([0.01, -0.01] * 32)
    assert abs(annualized_sharpe(noisy)) < 1.0
    assert annualized_sharpe(np.array([0.01])) == 0.0  # too short


def test_excess_sharpe_mode_prefers_high_sharpe_over_big_nav() -> None:
    """excess_sharpe selection must rank a steady small edge above a volatile
    large-NAV path (the legacy NAV-dollar score prefers the latter)."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    n = 64

    def _episode(daily_rets: np.ndarray, start_bar: int) -> dict:
        nav = 100_000.0 * np.exp(np.concatenate(([0.0], np.cumsum(daily_rets))))
        peak = np.maximum.accumulate(nav)
        dd = peak - nav
        return {
            "start_bar": start_bar,
            "nav_path": nav.tolist(),
            "ending_nav": float(nav[-1]),
            "max_drawdown_nav": float(dd.max()),
            "max_drawdown_frac": float((dd / peak).max()),
        }

    # Flat benchmark prices → benchmark NAV ~flat, excess ≈ agent returns.
    ctx = EvalBenchmarkContext(
        ohlcv=np.full((200, 10, 5), 100.0),
        idx=pd.date_range("2020-01-01", periods=200),
        tickers=[f"A{i}" for i in range(10)],
        mode="equal_weight_daily",
    )
    steady = [_episode(np.full(n, 0.0004) + rng.normal(0, 0.0005, n), 10)]
    swingy = [_episode(np.full(n, 0.002) + rng.normal(0, 0.02, n), 10)]

    sharpe_steady = compute_robust_eval_score(
        steady, benchmark_ctx=ctx, score_mode="excess_sharpe", std_coef=0.0, dd_coef=0.0
    )
    sharpe_swingy = compute_robust_eval_score(
        swingy, benchmark_ctx=ctx, score_mode="excess_sharpe", std_coef=0.0, dd_coef=0.0
    )
    nav_steady = compute_robust_eval_score(
        steady, benchmark_ctx=ctx, score_mode="excess_nav", std_coef=0.0, dd_coef=0.0
    )
    nav_swingy = compute_robust_eval_score(
        swingy, benchmark_ctx=ctx, score_mode="excess_nav", std_coef=0.0, dd_coef=0.0
    )
    assert sharpe_steady["score"] > sharpe_swingy["score"]
    assert nav_swingy["score"] > nav_steady["score"]  # legacy mode ranks by dollars
    assert "pooled_excess_sharpe" in sharpe_steady
    assert sharpe_steady["segment_excess_sharpe_mean"] > 0.0


def test_excess_sharpe_mode_uses_frac_drawdown_penalty() -> None:
    pd = pytest.importorskip("pandas")
    ctx = EvalBenchmarkContext(
        ohlcv=np.full((100, 10, 5), 100.0),
        idx=pd.date_range("2020-01-01", periods=100),
        tickers=[f"A{i}" for i in range(10)],
        mode="equal_weight_daily",
    )
    nav = list(np.linspace(100_000.0, 101_000.0, 40))
    base = {
        "start_bar": 5,
        "nav_path": nav,
        "ending_nav": nav[-1],
        "max_drawdown_nav": 0.0,
        "max_drawdown_frac": 0.0,
    }
    deep = {**base, "max_drawdown_nav": 20_000.0, "max_drawdown_frac": 0.20}
    s_shallow = compute_robust_eval_score(
        [base], benchmark_ctx=ctx, score_mode="excess_sharpe", std_coef=0.0, dd_coef=2.0
    )
    s_deep = compute_robust_eval_score(
        [deep], benchmark_ctx=ctx, score_mode="excess_sharpe", std_coef=0.0, dd_coef=2.0
    )
    # Same return path; only the frac drawdown differs → 2.0 × 0.20 = 0.4 score gap.
    assert s_shallow["score"] - s_deep["score"] == pytest.approx(0.4)


def test_score_mode_validation() -> None:
    with pytest.raises(ValueError, match="score_mode"):
        compute_robust_eval_score(
            [{"ending_nav": 1.0, "max_drawdown_nav": 0.0}], score_mode="bogus"
        )


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


def test_apply_episode_burn_in_trims_cash_restart() -> None:
    from rlbot.eval_selection import apply_episode_burn_in

    nav = [100_000.0] + [100_000.0 - 1000.0 * i for i in range(1, 40)]
    weights = np.zeros((39, 3), dtype=np.float64)
    weights[:, 0] = 1.0  # all cash
    weights[10:, 0] = 0.0
    weights[10:, 1] = 1.0
    ep = {
        "nav_path": nav,
        "ending_nav": nav[-1],
        "start_nav": nav[0],
        "start_bar": 50,
        "max_drawdown_nav": 0.0,
        "max_drawdown_frac": 0.0,
        "weights": weights,
    }
    out = apply_episode_burn_in(ep, 10)
    assert out["start_bar"] == 60
    assert len(out["nav_path"]) == len(nav) - 10
    assert out["start_nav"] == pytest.approx(nav[10])
    assert out["ending_nav"] == pytest.approx(nav[-1])
    assert np.asarray(out["weights"]).shape[0] == 29
    assert out["max_drawdown_nav"] >= 0.0


def test_blend_block_and_oos_aligned_scores() -> None:
    from rlbot.eval_selection import blend_block_and_oos_aligned_scores

    assert blend_block_and_oos_aligned_scores(1.0, 3.0, weight=0.75) == pytest.approx(2.5)
    assert blend_block_and_oos_aligned_scores(1.0, None, weight=0.75) == pytest.approx(1.0)
    assert blend_block_and_oos_aligned_scores(1.0, 3.0, weight=0.0) == pytest.approx(1.0)
    assert blend_block_and_oos_aligned_scores(1.0, 3.0, weight=1.0) == pytest.approx(3.0)


def test_aggregate_multi_regime_scores() -> None:
    from rlbot.eval_selection import aggregate_multi_regime_scores

    scores = [0.0, 1.0, 2.0, 3.0]
    assert aggregate_multi_regime_scores(scores, agg="min") == pytest.approx(0.0)
    assert aggregate_multi_regime_scores(scores, agg="mean") == pytest.approx(1.5)
    assert aggregate_multi_regime_scores(scores, agg="median") == pytest.approx(1.5)
    # p25 of [0,1,2,3] ≈ 0.75
    assert aggregate_multi_regime_scores(scores, agg="p25") == pytest.approx(0.75)
    assert aggregate_multi_regime_scores([], agg="p25") == float("-inf")


def test_apply_dd_exposure_taper() -> None:
    from rlbot.eval_selection import apply_dd_exposure_taper

    w = np.array([0.1, 0.3, 0.3, 0.3], dtype=np.float64)
    # Below start: unchanged
    out0 = apply_dd_exposure_taper(w, 0.03, start=0.06, end=0.12, min_gross=0.30)
    assert out0 == pytest.approx(w)
    # At/above end: risky scaled to min_gross
    out1 = apply_dd_exposure_taper(w, 0.20, start=0.06, end=0.12, min_gross=0.30)
    assert out1[1:].sum() == pytest.approx(0.30, abs=1e-9)
    assert out1[0] == pytest.approx(0.70, abs=1e-9)
    assert out1.sum() == pytest.approx(1.0)
    # Mid taper: between 1.0 and min_gross
    out_mid = apply_dd_exposure_taper(w, 0.09, start=0.06, end=0.12, min_gross=0.30)
    assert 0.30 < out_mid[1:].sum() < 0.90


def test_defensive_sharpe_uses_max_dd_not_p75() -> None:
    """defensive_sharpe must penalize the worst segment DD, not the 75th percentile."""
    pd = pytest.importorskip("pandas")
    # Two segments: one mild DD, one deep — p75 can sit near mild, max is deep.
    mild = {
        "start_bar": 0,
        "nav_path": [100_000.0, 101_000.0, 100_500.0, 102_000.0],
        "ending_nav": 102_000.0,
        "max_drawdown_nav": 500.0,
        "max_drawdown_frac": 0.005,
    }
    deep = {
        "start_bar": 10,
        "nav_path": [100_000.0, 110_000.0, 85_000.0, 95_000.0],
        "ending_nav": 95_000.0,
        "max_drawdown_nav": 25_000.0,
        "max_drawdown_frac": 0.227,
    }
    ctx = EvalBenchmarkContext(
        ohlcv=np.full((40, 10, 5), 100.0),
        idx=pd.date_range("2020-01-01", periods=40),
        tickers=[f"A{i}" for i in range(10)],
        mode="equal_weight_daily",
    )
    eps = [mild, deep]
    excess = compute_robust_eval_score(
        eps, benchmark_ctx=ctx, score_mode="excess_sharpe", std_coef=0.0, dd_coef=8.0,
        stitched_blend=0.0,
    )
    defensive = compute_robust_eval_score(
        eps, benchmark_ctx=ctx, score_mode="defensive_sharpe", std_coef=0.0, dd_coef=8.0,
        stitched_blend=0.0,
    )
    assert defensive["max_max_drawdown_frac"] == pytest.approx(0.227)
    # Same return signal → defensive score must be strictly worse (max > p75).
    assert defensive["score"] < excess["score"]


def test_multi_regime_slice_bounds_span_panel() -> None:
    from rlbot.data_utils import multi_regime_slice_bounds

    bounds = multi_regime_slice_bounds(2000, n_slices=5, slice_bars=252)
    assert len(bounds) == 5
    assert bounds[0] == (0, 252)
    assert bounds[-1] == (2000 - 252, 2000)
    for start, end in bounds:
        assert end - start == 252
        assert 0 <= start < end <= 2000
    # Interior slices should be distinct and ordered.
    starts = [s for s, _ in bounds]
    assert starts == sorted(starts)
    assert len(set(starts)) == 5


def test_build_multi_regime_walkforward_packs() -> None:
    import pandas as pd
    from rlbot.data_utils import build_multi_regime_walkforward_packs

    n, n_assets = 800, 3
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    ohlcv = np.ones((n, n_assets, 5), dtype=np.float64)
    macro = np.ones((n, 4), dtype=np.float64)
    live = np.ones((n, n_assets), dtype=np.float64)
    feat = np.zeros((n, n_assets), dtype=np.float64)
    packs = build_multi_regime_walkforward_packs(
        idx,
        ohlcv,
        macro,
        live,
        rsi=feat,
        macd=feat,
        fracdiff=feat,
        fracdiff_macro=np.zeros((n, 4), dtype=np.float64),
        trend=feat,
        asset_vol=feat,
        macro_vol=np.zeros((n, 4), dtype=np.float64),
        n_slices=4,
        slice_bars=126,
    )
    assert len(packs) == 4
    for p in packs:
        assert len(p.idx) == 126
        assert p.ohlcv.shape[0] == 126
        assert p.block_boundaries == []


def test_compute_robust_eval_score_burn_in_changes_signal() -> None:
    """Burn-in should drop the early cash-heavy steps from the scored path."""
    pd = pytest.importorskip("pandas")
    # Flat then rising: burn-in past the flat region changes excess dynamics.
    nav = [100_000.0] * 15 + [100_000.0 * (1.001 ** i) for i in range(1, 40)]
    ep = {
        "start_bar": 0,
        "nav_path": nav,
        "ending_nav": nav[-1],
        "max_drawdown_nav": 0.0,
        "max_drawdown_frac": 0.0,
    }
    ctx = EvalBenchmarkContext(
        ohlcv=np.full((80, 10, 5), 100.0),
        idx=pd.date_range("2020-01-01", periods=80),
        tickers=[f"A{i}" for i in range(10)],
        mode="equal_weight_daily",
    )
    full = compute_robust_eval_score(
        [ep], benchmark_ctx=ctx, stitched_blend=1.0, std_coef=0.0, dd_coef=0.0,
        score_mode="excess_sharpe", burn_in_bars=0,
    )
    burned = compute_robust_eval_score(
        [ep], benchmark_ctx=ctx, stitched_blend=1.0, std_coef=0.0, dd_coef=0.0,
        score_mode="excess_sharpe", burn_in_bars=14,
    )
    assert np.isfinite(full["score"])
    assert np.isfinite(burned["score"])
    # Not required to differ for flat EW bench, but burn-in must not crash / go -inf.
    assert burned["score"] > float("-inf")
