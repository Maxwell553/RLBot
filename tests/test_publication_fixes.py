"""Regression tests for publication-readiness fixes (VecNormalize pairing, weights, cache, manifest)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from rlbot.baselines import (
    benchmark_only_nav,
    cash_nav,
    equal_weight_daily_cost_aware_nav,
    equal_weight_monthly_nav,
)
from rlbot.data_utils import save_cache, select_tradeable_columns
from rlbot.rl_config import get_config, slice_config_to_n_assets
from rlbot.run_artifacts import merge_manifest, write_manifest
from rlbot.trading_env import MultiAssetPortfolioEnv


def _tiny_panel(n_bars: int = 80, n_a: int = 5, seed: int = 0):
    from rlbot.data_utils import N_MACRO

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n_bars, freq="B")
    rets = rng.normal(0.0003, 0.01, size=(n_bars, n_a))
    price = 100.0 * np.exp(np.cumsum(rets, axis=0))
    ohlcv = np.zeros((n_bars, n_a, 5), dtype=np.float64)
    for c in range(4):
        ohlcv[:, :, c] = price
    ohlcv[:, :, 4] = 1e6
    macro = 10.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, size=(n_bars, N_MACRO)), axis=0))
    rsi = np.full((n_bars, n_a), 50.0)
    macd = np.zeros((n_bars, n_a))
    fd = np.zeros((n_bars, n_a))
    fdm = np.zeros((n_bars, N_MACRO))
    trend = np.zeros((n_bars, n_a))
    live = np.ones((n_bars, n_a))
    avol = np.full((n_bars, n_a), 0.01)
    mvol = np.full((n_bars, N_MACRO), 0.01)
    return idx, ohlcv, rsi, macd, macro, fd, fdm, trend, avol, mvol, live


def test_merge_manifest_preserves_holdout_block(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(
        path,
        {
            "run_id": "r1",
            "chronological_holdout": {"holdout_days": 365, "train_end": "2015-12-31"},
            "n_trainable_bars": 1000,
        },
    )
    merge_manifest(path, {"run_id": "r1", "finished_at_utc": "2026-01-01", "best_eval_nav": 1.0})
    data = json.loads(path.read_text())
    assert data["chronological_holdout"]["train_end"] == "2015-12-31"
    assert data["n_trainable_bars"] == 1000
    assert data["best_eval_nav"] == 1.0


def test_run_cache_snapshot_width_matches_sliced_universe(tmp_path: Path) -> None:
    full_cfg = get_config()
    n_wide = full_cfg.universe.n_assets
    idx, ohlcv, rsi, macd, macro, fd, fdm, trend, avol, mvol, live = _tiny_panel(n_a=n_wide)
    wide_tickers = list(full_cfg.universe.tickers)
    cache_path = tmp_path / "wide.npz"
    save_cache(
        str(cache_path),
        idx,
        ohlcv,
        rsi,
        macd,
        macro,
        fd,
        fdm,
        trend,
        avol,
        mvol,
        asset_live=live,
        tickers=wide_tickers,
    )
    cfg = slice_config_to_n_assets(full_cfg, 5)
    want = list(cfg.universe.tickers)
    ohlcv_s, rsi_s, macd_s, fd_s, trend_s, tickers_s, live_s, avol_s, mvol_s = select_tradeable_columns(
        ohlcv,
        rsi,
        macd,
        fd,
        trend,
        wide_tickers,
        want,
        asset_live=live,
        asset_vol=avol,
        macro_vol=mvol,
    )
    run_cache = tmp_path / "run.npz"
    save_cache(
        str(run_cache),
        idx,
        ohlcv_s,
        rsi_s,
        macd_s,
        macro,
        fd_s,
        fdm,
        trend_s,
        avol_s,
        mvol_s,
        asset_live=live_s,
        tickers=tickers_s,
    )
    from rlbot.data_utils import load_cache

    _, ohlcv_r, *_rest, tickers_r = load_cache(str(run_cache))
    assert ohlcv_r.shape[1] == 5
    assert list(tickers_r) == want


def test_target_weights_match_smoothed_execution() -> None:
    n_a = get_config().universe.n_assets
    n_bars = 40
    idx, ohlcv, rsi, macd, macro, fd, fdm, trend, avol, mvol, live = _tiny_panel(n_bars=n_bars, n_a=n_a)
    env = MultiAssetPortfolioEnv(
        ohlcv,
        rsi,
        macd,
        macro=macro,
        fracdiff=fd,
        fracdiff_macro=fdm,
        trend=trend,
        asset_realized_vol=avol,
        macro_realized_vol=mvol,
        asset_live=live,
        random_start=False,
        max_episode_steps=n_bars - 2,
        obs_lag=0,
        action_smoothing_alpha=0.25,
        domain_randomize=False,
    )
    env.reset(seed=0)
    raw = np.zeros(n_a + 1, dtype=np.float64)
    raw[0] = 1.0
    raw[1:] = np.linspace(-1.0, 3.0, n_a)
    _, _, _, _, info1 = env.step(raw)
    _, _, _, _, info2 = env.step(raw * 0.5)
    assert "target_weights" in info1
    assert info1["target_weights"].shape == (n_a + 1,)
    assert np.isclose(info1["target_weights"].sum(), 1.0)
    assert not np.allclose(info1["target_weights"], info2["target_weights"])


def test_train_saves_paired_vecnormalize_on_best_eval() -> None:
    from rlbot.run_artifacts import PROJECT_ROOT

    src = (PROJECT_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "self._train_vec_env.save" in src
    assert 'self._best_model_dir / "vec_normalize.pkl"' in src
    assert "shutil.copy2(root_vn, best_vn)" not in src


def test_backtest_collects_executed_target_weights() -> None:
    from rlbot.run_artifacts import PROJECT_ROOT

    src = (PROJECT_ROOT / "scripts" / "backtest.py").read_text(encoding="utf-8")
    assert 'info.get("target_weights")' in src
    assert "portfolio_weights_from_action" not in src.split("collect_weights")[1].split("def ")[0]


def test_cash_nav_is_flat() -> None:
    navs = np.array([100_000.0, 100_000.0, 100_000.0])
    out = cash_nav(navs, np.zeros((5, 2, 5)), 0)
    np.testing.assert_allclose(out, navs[0])


def test_new_benchmark_nav_paths_have_model_length() -> None:
    n_bars = 60
    n_a = get_config().universe.n_assets
    idx, ohlcv, *_rest, live = _tiny_panel(n_bars=n_bars, n_a=n_a)
    tickers = list(get_config().universe.tickers)
    navs = np.linspace(100_000, 105_000, n_bars)
    start = 0
    for fn, kwargs in (
        (benchmark_only_nav, {"tickers": tickers, "benchmark_col": 0}),
        (equal_weight_daily_cost_aware_nav, {"asset_live": live}),
        (equal_weight_monthly_nav, {"test_idx": idx, "asset_live": live}),
    ):
        out = fn(navs, ohlcv, start, **kwargs)
        assert len(out) == len(navs)


def test_finetune_and_resume_mutually_exclusive() -> None:
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "train.py"), "--resume", "a.zip", "--finetune", "b.zip"],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert proc.returncode != 0
    assert "only one" in proc.stderr.lower() or "only one" in proc.stdout.lower()
