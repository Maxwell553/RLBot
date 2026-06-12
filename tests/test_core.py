"""Core math and allocation smoke tests (no network, no full training loop)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlbot.data_utils import (
    N_MACRO,
    MACRO_TICKERS,
    _hy_oas_proxy_pct,
    compute_feature_panel,
    compute_trend_signals,
    fracdiff_weights,
    WalkforwardEnvPack,
    train_test_split_alternating,
)
from rlbot.rl_config import (
    UNIVERSE_MAX_ASSETS,
    UNIVERSE_MIN_ASSETS,
    get_config,
    slice_config_to_n_assets,
    validate_config_for_universe,
    validate_universe_asset_count,
)


def test_benchmark_cap_weights_normalize() -> None:
    cfg = get_config()
    w = cfg.reward.benchmark_cap_weights_array()
    assert w.shape == (cfg.universe.n_assets,)
    assert np.isclose(w.sum(), 1.0)
    # Equal-weight passive benchmark (1/N) — feasible under max_single_asset_weight 0.20
    assert np.allclose(w, 1.0 / cfg.universe.n_assets)


def test_fracdiff_weights_start_at_one() -> None:
    w = fracdiff_weights(0.4)
    assert w[0] == pytest.approx(1.0)
    assert len(w) > 10


def test_hy_oas_proxy_in_sane_range() -> None:
    hyg = np.array([80.0, 75.0, 90.0])
    ief = np.array([100.0, 100.0, 100.0])
    spread = _hy_oas_proxy_pct(hyg, ief)
    assert spread.shape == (3,)
    assert np.all(spread >= 2.0)
    assert np.all(spread <= 15.0)


def test_macro_ticker_count() -> None:
    assert N_MACRO == 4
    assert "VIX" in MACRO_TICKERS
    assert "HY_OAS" in MACRO_TICKERS


def test_universe_asset_count_bounds() -> None:
    with pytest.raises(ValueError, match="between"):
        validate_universe_asset_count(UNIVERSE_MIN_ASSETS - 1)
    with pytest.raises(ValueError, match="between"):
        validate_universe_asset_count(UNIVERSE_MAX_ASSETS + 1)
    validate_universe_asset_count(UNIVERSE_MIN_ASSETS)
    validate_universe_asset_count(UNIVERSE_MAX_ASSETS)
    validate_universe_asset_count(get_config().universe.n_assets)


def test_validate_config_for_universe_mismatch() -> None:
    cfg = get_config()
    with pytest.raises(ValueError, match="universe.assets"):
        validate_config_for_universe(cfg, cfg.universe.n_assets + 1)


def test_benchmark_combined_abs_cap_validation() -> None:
    from rlbot.rl_config import RewardConfig, _validate_reward_config

    base = dict(
        reward_scale=1.0,
        max_step_log_return=0.1,
        max_step_log_return_downside=-0.1,
        risk_window=10,
        sortino_min_steps=1,
        sortino_downside_floor=1e-4,
        risk_bonus_scale=1.0,
        benchmark_cap_weights=(1.0,),
        benchmark_excess_scale=1.0,
        benchmark_excess_clip=0.01,
        benchmark_combined_abs_cap=24.0,
        churn_penalty=1.0,
        turnover_penalty=0.0,
        drawdown_downside_gamma=1.0,
        drawdown_increase_penalty=0.75,
        drawdown_level_penalty=3.0,
        drawdown_level_floor=0.08,
        concentration_penalty=0.35,
        concentration_target_eff_assets=5.5,
        cash_daily_yield=0.0,
        inactivity_penalty_over_50=1.0,
        inactivity_penalty_over_90=1.0,
        eval_inactivity_penalty_scale=1.0,
        participation_bonus=0.0,
        participation_reward_scale=1.0,
        exposure_risk_mode="realized_vol",
        exposure_risk_penalty_scale=0.0,
    )
    _validate_reward_config(RewardConfig(**base))
    _validate_reward_config(RewardConfig(**{**base, "benchmark_combined_abs_cap": 0.0}))
    with pytest.raises(ValueError, match="benchmark_combined_abs_cap"):
        _validate_reward_config(RewardConfig(**{**base, "benchmark_combined_abs_cap": -1.0}))


def test_legacy_benchmark_relative_max_share_translates_to_constant_cap() -> None:
    """Old run snapshots parse: share s → cap = s/(1-s) × reward_scale × 1%."""
    import copy

    from rlbot.rl_config import _parse_config

    raw = copy.deepcopy(get_config().raw)
    raw["reward"].pop("benchmark_combined_abs_cap", None)
    raw["reward"]["benchmark_relative_max_share"] = 0.6
    cfg = _parse_config(raw, get_config().path)
    expected = (0.6 / 0.4) * raw["reward"]["reward_scale"] * 0.01
    assert cfg.reward.benchmark_combined_abs_cap == pytest.approx(expected)
    raw["reward"]["benchmark_relative_max_share"] = 0.0
    assert _parse_config(raw, get_config().path).reward.benchmark_combined_abs_cap == 0.0


def test_slice_config_to_n_assets() -> None:
    full = get_config()
    n = 7
    sliced = slice_config_to_n_assets(full, n)
    assert sliced.universe.n_assets == n
    assert sliced.universe.tickers == full.universe.tickers[:n]
    assert full.universe.benchmark in sliced.universe.assets
    w = sliced.reward.benchmark_cap_weights_array()
    assert w.shape == (n,)
    assert np.isclose(w.sum(), 1.0)
    assert len(sliced.transaction_costs.slippage) == n
    validate_config_for_universe(sliced, n)


def test_slice_config_to_n_assets_rejects_over_config() -> None:
    full = get_config()
    with pytest.raises(ValueError, match="defines only"):
        slice_config_to_n_assets(full, full.universe.n_assets + 1)


def test_trend_signals_shape() -> None:
    t, n = 120, get_config().universe.n_assets
    ohlcv = np.random.rand(t, n, 5) * 50 + 100
    ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
    trend = compute_trend_signals(ohlcv)
    assert trend.shape == (t, n)
    assert np.all(np.isfinite(trend[-1]))


def _mock_ohlcv(bars: int, n_assets: int) -> np.ndarray:
    ohlcv = np.zeros((bars, n_assets, 5), dtype=np.float64)
    ohlcv[:, :, 3] = np.cumsum(np.random.rand(bars, n_assets) * 0.01, axis=0) + 100.0
    ohlcv[:, :, 0] = ohlcv[:, :, 3] * 0.999
    return ohlcv


def test_portfolio_step_simple_return_identity() -> None:
    from rlbot.baselines import portfolio_step_nav

    n_assets = get_config().universe.n_assets
    ohlcv = _mock_ohlcv(40, n_assets)
    w = np.full(n_assets, 1.0 / n_assets)
    prev = 100_000.0
    t = 10
    nav_next = portfolio_step_nav(prev, ohlcv, t, w)
    close_pre = ohlcv[t, :, 3]
    open_n = ohlcv[t + 1, :, 0]
    close_n = ohlcv[t + 1, :, 3]
    r_on = np.expm1(np.log((open_n + 1e-12) / (close_pre + 1e-12)))
    r_id = np.expm1(np.log((close_n + 1e-12) / (open_n + 1e-12)))
    expected = prev * (1.0 + float(np.dot(w, r_on))) * (1.0 + float(np.dot(w, r_id)))
    assert nav_next == pytest.approx(expected)


def test_portfolio_step_friction_lowers_nav_vs_frictionless() -> None:
    from rlbot.baselines import _transaction_cost_arrays, portfolio_step_nav

    n_assets = get_config().universe.n_assets
    ohlcv = _mock_ohlcv(50, n_assets)
    w = np.full(n_assets, 1.0 / n_assets)
    slip, fee, hold = _transaction_cost_arrays(n_assets)
    prev = 100_000.0
    t = 10
    free = portfolio_step_nav(prev, ohlcv, t, w)
    costly = portfolio_step_nav(
        prev,
        ohlcv,
        t,
        w,
        prev_weights=w,
        slippage=slip,
        tx_fee=fee,
        daily_holding=hold,
    )
    assert costly < free


def test_equal_weight_buyhold_nav_length() -> None:
    from rlbot.baselines import benchmark_buyhold_nav, equal_weight_buyhold_nav

    n_assets = get_config().universe.n_assets
    start = 2
    navs = np.linspace(100_000.0, 105_000.0, 20)
    t = start + len(navs)
    ohlcv = np.random.rand(t, n_assets, 5) * 100 + 50
    ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
    ew = equal_weight_buyhold_nav(navs, ohlcv, start)
    spy = benchmark_buyhold_nav(navs, ohlcv, start)
    assert len(ew) == len(navs)
    assert len(spy) == len(navs)
    assert ew[0] == pytest.approx(navs[0])


def test_risk_parity_ignores_pre_ipo_flat_assets() -> None:
    from rlbot.baselines import naive_risk_parity_nav

    n_assets = 4
    bars = 80
    ohlcv = np.zeros((bars, n_assets, 5), dtype=np.float64)
    ohlcv[:, :, 3] = 100.0
    ohlcv[:, :, 0] = 100.0
    ohlcv[40:, 1, 3] = np.linspace(100.0, 110.0, bars - 40)
    ohlcv[40:, 1, 0] = ohlcv[40:, 1, 3]
    ohlcv[40:, 2, 3] = np.linspace(100.0, 105.0, bars - 40) + np.random.default_rng(1).normal(
        0, 0.5, bars - 40
    )
    ohlcv[40:, 2, 0] = ohlcv[40:, 2, 3]
    live = np.zeros((bars, n_assets), dtype=np.float64)
    live[40:, 1] = 1.0
    live[40:, 2] = 1.0
    live[:40, 2] = 0.0
    navs = np.full(20, 100_000.0)
    nav_rp = naive_risk_parity_nav(
        navs,
        ohlcv,
        start_bar=35,
        asset_live=live,
        lookback=10,
        apply_costs=False,
    )
    assert len(nav_rp) == len(navs)
    assert np.isfinite(nav_rp).all()


def test_balanced_6040_and_risk_parity_nav() -> None:
    from rlbot.baselines import balanced_6040_nav, naive_risk_parity_nav

    n_assets = get_config().universe.n_assets
    tickers = get_config().universe.tickers
    start = 25
    navs = np.full(15, 100_000.0)
    t = start + len(navs)
    ohlcv = np.random.rand(t, n_assets, 5) * 100 + 50
    ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
    idx = pd.date_range("2020-01-01", periods=t, freq="B")
    nav_6040 = balanced_6040_nav(navs, ohlcv, start, idx, tickers=tickers)
    nav_rp = naive_risk_parity_nav(navs, ohlcv, start, lookback=20)
    assert len(nav_6040) == len(navs)
    assert len(nav_rp) == len(navs)
    assert nav_6040[0] == pytest.approx(100_000.0)
    assert nav_rp[0] == pytest.approx(100_000.0)


def test_realized_vol_panel_matches_env_on_the_fly() -> None:
    from rlbot.trading_env import MultiAssetPortfolioEnv

    n = 120
    n_assets = get_config().universe.n_assets
    n_macro = N_MACRO
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    ohlcv = np.cumsum(np.random.default_rng(1).normal(0, 0.01, (n, n_assets, 5)), axis=0) + 100.0
    ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
    macro = np.full((n, n_macro), 10.0, dtype=np.float64)
    _, _, _, _, _, avol, mvol = compute_feature_panel(ohlcv, macro, lookback=20)
    env = MultiAssetPortfolioEnv(
        ohlcv,
        np.full((n, n_assets), 50.0),
        np.zeros((n, n_assets)),
        macro=macro,
        fracdiff=np.zeros((n, n_assets)),
        fracdiff_macro=np.zeros((n, n_macro)),
        trend=np.zeros((n, n_assets)),
        asset_realized_vol=avol,
        macro_realized_vol=mvol,
        random_start=False,
        domain_randomize=False,
    )
    t = 60
    assert np.allclose(env._asset_vol_panel[t], env._realized_vol(t), rtol=0, atol=1e-9)
    assert np.allclose(env._macro_vol_panel[t], env._macro_realized_vol(t), rtol=0, atol=1e-9)


def test_hy_proxy_calibration_uses_only_past_overlap() -> None:
    from rlbot.data_utils import _calibrate_hy_proxy_expanding

    n = 80
    proxy = np.linspace(3.0, 5.0, n)
    fred = np.full(n, np.nan)
    fred[40:] = proxy[40:] * 2.0 + 1.0
    out = _calibrate_hy_proxy_expanding(proxy, fred, min_overlap=10)
    assert np.isfinite(out[39])
    assert abs(out[39] - proxy[39]) < 1e-6
    assert abs(out[-1] - fred[-1]) < 0.05


def test_walkforward_pack_env_kwargs_feature_order() -> None:
    """Pack maps macro before fracdiff (matches MultiAssetPortfolioEnv keyword layout)."""
    n = 400
    n_assets = 3
    n_macro = N_MACRO
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    ohlcv = np.ones((n, n_assets, 5), dtype=np.float64) * 100.0
    macro = np.full((n, n_macro), 7.0, dtype=np.float64)
    fd = np.full((n, n_assets), 3.0, dtype=np.float64)
    pack = WalkforwardEnvPack(
        idx,
        ohlcv,
        np.zeros((n, n_assets)),
        np.zeros((n, n_assets)),
        macro,
        fd,
        np.zeros((n, n_macro)),
        np.zeros((n, n_assets)),
        np.zeros((n, n_assets)),
        np.zeros((n, n_macro)),
        [],
        np.ones((n, n_assets)),
    )
    kw = pack.env_kwargs()
    assert kw["macro"] is macro
    assert kw["fracdiff"] is fd
    assert np.all(kw["macro"] == 7.0)
    assert np.all(kw["fracdiff"] == 3.0)


def test_train_test_split_accepts_precomputed_features() -> None:
    n = 400
    n_assets = 3
    n_macro = N_MACRO
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    ohlcv = np.ones((n, n_assets, 5), dtype=np.float64) * 100.0
    macro = np.zeros((n, n_macro), dtype=np.float64)
    live = np.ones((n, n_assets), dtype=np.float64)
    sentinel = np.full((n, n_assets), 99.0, dtype=np.float64)
    train_pack, _ = train_test_split_alternating(
        idx,
        ohlcv,
        macro,
        asset_live=live,
        block_size=126,
        eval_stride=4,
        feature_split_mode="continuous",
        rsi=sentinel,
        macd=sentinel,
        fracdiff=sentinel,
        fracdiff_macro=np.zeros((n, n_macro)),
        trend=sentinel,
        asset_vol=sentinel,
        macro_vol=np.zeros((n, n_macro)),
    )
    assert np.all(train_pack[2] == 99.0)


def test_risk_parity_vol_uses_decision_bar_not_settlement() -> None:
    from rlbot.baselines import _risk_parity_weights, realized_vol_at_bar

    bars = 50
    n_assets = 2
    close = np.ones((bars, n_assets), dtype=np.float64) * 100.0
    close[30:, 0] = np.linspace(100.0, 120.0, bars - 30)
    close[30:, 1] = 100.0
    t = 35
    lookback = 5
    vol_t = realized_vol_at_bar(close, t, lookback)
    vol_tp1 = realized_vol_at_bar(close, t + 1, lookback)
    assert vol_t[0] > vol_tp1[0] or not np.isclose(vol_t[0], vol_tp1[0])


def test_train_test_split_slices_global_fracdiff_panel() -> None:
    """Block slices must match continuous feature panel (no per-block cold start)."""
    n = 400
    n_assets = 4
    n_macro = N_MACRO
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    ohlcv = np.zeros((n, n_assets, 5), dtype=np.float64)
    ohlcv[:, :, 3] = np.cumsum(np.random.default_rng(0).normal(0, 0.01, (n, n_assets)), axis=0) + 100.0
    ohlcv[:, :, 0] = ohlcv[:, :, 3] * 0.999
    macro = np.full((n, n_macro), 10.0, dtype=np.float64)
    live = np.ones((n, n_assets), dtype=np.float64)

    _, _, fd_g, _, _, avol_g, _ = compute_feature_panel(ohlcv, macro)
    train_pack, _ = train_test_split_alternating(
        idx, ohlcv, macro, asset_live=live, block_size=126, eval_stride=4,
        feature_split_mode="continuous",
    )
    tr_idx, tr_fd = train_pack[0], train_pack[5]
    assert len(tr_idx) > 126
    assert np.any(np.abs(tr_fd[30]) > 1e-6)
    assert np.allclose(tr_fd[30], fd_g[30], rtol=0, atol=1e-9)
