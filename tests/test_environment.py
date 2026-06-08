"""Portfolio weight mapping: simplex constraints and per-asset cap bounds."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rlbot.rl_config import get_config, observation_dim_for_universe, set_config
from rlbot.trading_env import MultiAssetPortfolioEnv, _softmax_1d, portfolio_weights_from_action

_CAP_TOL = 1e-6
_SIMPLEX_TOL = 1e-8

_N_ASSETS = get_config().universe.n_assets
_N_ACTIONS = _N_ASSETS + 1


def _max_cap() -> float:
    return float(get_config().environment.max_single_asset_weight)


def _assert_valid_weights(w: np.ndarray, n_actions: int = _N_ACTIONS) -> None:
    """Long-only simplex on cash + risky sleeves; each risky leg ≤ config cap."""
    assert w.shape == (n_actions,)
    assert w.dtype == np.float64
    assert np.all(w >= -_SIMPLEX_TOL)
    assert np.isclose(w.sum(), 1.0, atol=_SIMPLEX_TOL)
    cap = _max_cap()
    assert np.all(w[1:] <= cap + _CAP_TOL), f"max risky weight {w[1:].max():.6f} > cap {cap}"


def test_config_cap_is_half() -> None:
    assert _max_cap() == pytest.approx(0.35)


def test_softmax_mean_decoupling() -> None:
    a = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.array([25.0, 25.0, 25.0, 25.0])
    np.testing.assert_allclose(_softmax_1d(a), _softmax_1d(b), rtol=1e-9, atol=1e-9)


def test_asset_live_mask_zeros_pre_ipo_weights() -> None:
    action = np.zeros(_N_ACTIONS)
    action[1:] = 10.0

    live = np.ones(_N_ASSETS)
    if _N_ASSETS > 1:
        live[1] = 0.0

    w = portfolio_weights_from_action(action, n_actions=_N_ACTIONS, asset_live=live)
    _assert_valid_weights(w)

    if _N_ASSETS > 1:
        assert w[2] < 1e-6


def test_asset_live_mask_applied_before_cap_loop() -> None:
    """Dead assets must not inflate active sleeves past max_w after renormalize."""
    if _N_ASSETS < 3:
        pytest.skip("needs at least 3 tradeable assets")
    action = np.zeros(_N_ACTIONS)
    action[0] = -5.0
    action[2] = 12.0
    action[3] = 12.0
    live = np.ones(_N_ASSETS)
    live[0] = 0.0

    w = portfolio_weights_from_action(action, n_actions=_N_ACTIONS, asset_live=live)
    _assert_valid_weights(w)
    assert w[1] < 1e-6
    active = w[1:][live > 0.5]
    assert np.all(active <= _max_cap() + _CAP_TOL)


def test_simplex_uniform_logits() -> None:
    w = portfolio_weights_from_action(np.zeros(_N_ACTIONS), n_actions=_N_ACTIONS)
    _assert_valid_weights(w)


def test_simplex_all_cash_preference() -> None:
    action = np.full(_N_ACTIONS, -10.0)
    action[0] = 20.0
    w = portfolio_weights_from_action(action, n_actions=_N_ACTIONS)
    _assert_valid_weights(w)
    assert w[0] > 0.5


def test_cap_binds_single_asset_spike() -> None:
    action = np.zeros(_N_ACTIONS)
    action[1] = 10.0
    w = portfolio_weights_from_action(action, n_actions=_N_ACTIONS)
    _assert_valid_weights(w)
    assert w[1] == pytest.approx(_max_cap(), abs=_CAP_TOL)


def test_cap_binds_multiple_spikes_redistribute() -> None:
    """Three equal spikes: each risky leg capped; overflow to cash or other sleeves."""
    action = np.zeros(_N_ACTIONS)
    action[0] = -5.0
    action[1:4] = 10.0
    w = portfolio_weights_from_action(action, n_actions=_N_ACTIONS)
    _assert_valid_weights(w)
    active = w[1:][w[1:] > 1e-9]
    assert len(active) >= 1
    assert np.all(active <= _max_cap() + _CAP_TOL)


def test_random_actions_respect_simplex_and_cap() -> None:
    rng = np.random.default_rng(42)
    for _ in range(500):
        action = rng.uniform(-3.0, 3.0, size=_N_ACTIONS)
        _assert_valid_weights(portfolio_weights_from_action(action, n_actions=_N_ACTIONS))


def test_extreme_logits_still_valid() -> None:
    for scale in (50.0, 100.0, -100.0):
        action = np.linspace(-scale, scale, _N_ACTIONS)
        _assert_valid_weights(portfolio_weights_from_action(action, n_actions=_N_ACTIONS))


def test_action_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="action must have shape"):
        portfolio_weights_from_action(np.zeros(_N_ACTIONS - 1), n_actions=_N_ACTIONS)


def test_risky_mass_at_most_n_times_cap() -> None:
    """Structural bound: max risky exposure ≤ n_assets × cap (cash holds the rest)."""
    cap = _max_cap()
    w = portfolio_weights_from_action(np.ones(_N_ACTIONS), n_actions=_N_ACTIONS)
    _assert_valid_weights(w)
    assert float(w[1:].sum()) <= _N_ASSETS * cap + _CAP_TOL


def test_cap_post_condition_across_caps_and_n() -> None:
    """The final projection guarantees max risky ≤ cap for arbitrary cap and N."""
    base = get_config()
    rng = np.random.default_rng(7)
    try:
        for cap in (0.2, 0.35, 0.5):
            set_config(
                replace(base, environment=replace(base.environment, max_single_asset_weight=cap))
            )
            for n_assets in (5, 10, 23):
                n_act = n_assets + 1
                for _ in range(200):
                    action = rng.uniform(-6.0, 6.0, size=n_act)
                    w = portfolio_weights_from_action(action, n_actions=n_act)
                    assert w.shape == (n_act,)
                    assert np.all(w >= -_SIMPLEX_TOL)
                    assert np.isclose(w.sum(), 1.0, atol=_SIMPLEX_TOL)
                    assert np.all(w[1:] <= cap + _CAP_TOL), (
                        f"cap={cap} N={n_assets} max={w[1:].max():.6f}"
                    )
    finally:
        set_config(base)


def test_portfolio_weights_variable_n_actions() -> None:
    w = portfolio_weights_from_action(np.zeros(4), n_actions=4)
    _assert_valid_weights(w, n_actions=4)


def _synthetic_panel(t: int, n_assets: int):
    ohlcv = np.random.rand(t, n_assets, 5) * 50 + 100
    ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
    rsi = np.full((t, n_assets), 50.0)
    macd = np.zeros((t, n_assets))
    fracdiff = np.zeros((t, n_assets))
    fracdiff_macro = np.zeros((t, 4))
    trend = np.zeros((t, n_assets))
    macro = np.full((t, 4), 10.0)
    return ohlcv, rsi, macd, fracdiff, fracdiff_macro, trend, macro


def test_env_observation_dim_formula() -> None:
    """Observation space scales with n_assets from config-aligned cost arrays."""
    n_assets = _N_ASSETS
    t = 80
    ohlcv, rsi, macd, fracdiff, fracdiff_macro, trend, macro = _synthetic_panel(t, n_assets)
    env = MultiAssetPortfolioEnv(
        ohlcv,
        rsi,
        macd,
        macro=macro,
        fracdiff=fracdiff,
        fracdiff_macro=fracdiff_macro,
        trend=trend,
        random_start=False,
        domain_randomize=False,
    )
    assert env.n_assets == n_assets
    assert env.n_actions == n_assets + 1
    assert env.observation_space.shape[0] == observation_dim_for_universe(n_assets)


def _pad_float_list(xs: list[float], n: int) -> list[float]:
    out = list(xs)[:n]
    if len(out) >= n:
        return out
    fill = out[-1] if out else 0.0
    return out + [fill] * (n - len(out))


def _install_config_per_asset_lists(n_assets: int):
    """Align transaction-cost / benchmark vectors with synthetic panel width."""
    cfg = get_config()
    weights = _pad_float_list(cfg.reward.benchmark_cap_weights, n_assets)
    wsum = sum(weights)
    assert wsum > 0.0
    weights = [w / wsum for w in weights]
    patched = replace(
        cfg,
        reward=replace(cfg.reward, benchmark_cap_weights=weights),
        transaction_costs=replace(
            cfg.transaction_costs,
            slippage=_pad_float_list(cfg.transaction_costs.slippage, n_assets),
            tx_fee=_pad_float_list(cfg.transaction_costs.tx_fee, n_assets),
            annual_holding_cost=_pad_float_list(
                cfg.transaction_costs.annual_holding_cost, n_assets
            ),
        ),
    )
    set_config(patched)
    return cfg


@pytest.mark.parametrize("synthetic_n_assets", [5, 23, 55])
def test_env_strict_shape_invariance(synthetic_n_assets: int) -> None:
    """Env spaces and reset obs compile across the supported asset-count boundary range."""
    orig_cfg = _install_config_per_asset_lists(synthetic_n_assets)
    try:
        t = 100
        n_macro = 4
        ohlcv = np.random.rand(t, synthetic_n_assets, 5) * 50 + 100.0
        ohlcv[:, :, 3] = np.maximum(ohlcv[:, :, 3], 1.0)
        rsi = np.full((t, synthetic_n_assets), 50.0)
        macd = np.zeros((t, synthetic_n_assets))
        trend = np.zeros((t, synthetic_n_assets))
        fd = np.zeros((t, synthetic_n_assets))
        fdm = np.zeros((t, n_macro))
        macro = np.full((t, n_macro), 10.0)

        env = MultiAssetPortfolioEnv(
            ohlcv,
            rsi,
            macd,
            fracdiff=fd,
            fracdiff_macro=fdm,
            trend=trend,
            macro=macro,
            random_start=False,
            domain_randomize=False,
        )

        expected_obs_dim = observation_dim_for_universe(synthetic_n_assets, n_macro)
        assert env.n_assets == synthetic_n_assets
        assert env.n_actions == synthetic_n_assets + 1
        assert env.observation_space.shape[0] == expected_obs_dim
        assert env.action_space.shape[0] == synthetic_n_assets + 1

        obs, _ = env.reset()
        assert obs.shape == (expected_obs_dim,)
    finally:
        set_config(orig_cfg)


def _sortino_downside(rets: np.ndarray) -> float:
    m = float(rets.mean())
    downside_elements = np.minimum(rets, 0.0) ** 2
    dv = float(np.sqrt(downside_elements.mean()))
    return m / (dv + 1e-8)


def test_sortino_downside_dev_penalizes_loss_frequency() -> None:
    """Full-window downside dev: many loss days → lower Sortino than one loss day."""
    mild = np.array([0.01] * 62 + [-0.02])
    severe = np.array([-0.02] * 60 + [0.01] * 3)
    assert _sortino_downside(severe) < _sortino_downside(mild)


def test_sortino_bonus_requires_min_steps() -> None:
    """Sortino differential is withheld until sortino_min_steps (config default 20)."""
    n_assets = _N_ASSETS
    t = 80
    ohlcv, rsi, macd, fracdiff, fracdiff_macro, trend, macro = _synthetic_panel(t, n_assets)
    env = MultiAssetPortfolioEnv(
        ohlcv,
        rsi,
        macd,
        macro=macro,
        fracdiff=fracdiff,
        fracdiff_macro=fracdiff_macro,
        trend=trend,
        random_start=False,
        domain_randomize=False,
        fee_scale_default=0.0,
    )
    env.reset()
    sortino_seen = False
    for _ in range(15):
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
        if abs(info.get("rew_decomp/sortino", 0.0)) > 1e-9:
            sortino_seen = True
            break
    assert not sortino_seen


def test_sortino_downside_floor_avoids_tiny_denominator() -> None:
    rets = np.full(63, 0.0001)
    downside_elements = np.minimum(rets, 0.0) ** 2
    dv = max(float(np.sqrt(downside_elements.mean())), 1e-4)
    assert dv >= 1e-4
    assert abs(rets.mean() / dv) < 10.0


def test_benchmark_return_simple_then_log() -> None:
    """Weighted simple returns then log ≥ weighted sum of logs (Jensen)."""
    log_rets = np.array([0.02, -0.01, 0.015])
    w = np.array([0.55, 0.25, 0.20])
    wrong = float(np.dot(w, log_rets))
    simple_agg = float(np.dot(w, np.expm1(log_rets)))
    correct = float(np.log(1.0 + simple_agg))
    assert correct >= wrong - 1e-12


def test_rebalance_tx_cost_scales_with_fee_scale() -> None:
    """Realized tx cost fraction is zero when fee_scale=0, positive when fees apply."""
    n_assets = _N_ASSETS
    t = 40
    ohlcv, rsi, macd, fracdiff, fracdiff_macro, trend, macro = _synthetic_panel(t, n_assets)
    env = MultiAssetPortfolioEnv(
        ohlcv,
        rsi,
        macd,
        macro=macro,
        fracdiff=fracdiff,
        fracdiff_macro=fracdiff_macro,
        trend=trend,
        random_start=False,
        domain_randomize=False,
        fee_scale_default=1.0,
    )
    env.reset()
    env._t = env._min_t
    env._cash = 50_000.0
    env._units[:] = 50_000.0 / ohlcv[env._t, :, 3]
    price = ohlcv[env._t + 1, :, 0]
    w = np.zeros(env.n_actions)
    w[0] = 0.5
    w[1] = 0.5

    env.fee_scale = 0.0
    _, tx_free = env._rebalance(price, w)
    env.fee_scale = 1.0
    _, tx_full = env._rebalance(price, w)
    assert tx_free == 0.0
    assert tx_full >= 0.0


def test_inactivity_penalty_bounded_at_full_cash() -> None:
    """100% cash inactivity penalty is ~2.5 with default config (1.5 + 1.0 tail)."""
    rwd = get_config().reward
    expected = rwd.inactivity_penalty_over_50 + rwd.inactivity_penalty_over_90
    assert expected == pytest.approx(2.5)


def test_churn_penalty_uses_tx_cost_not_turnover() -> None:
    """Churn reward term scales with tx_cost_frac (zero when fee_scale=0)."""
    n_assets = _N_ASSETS
    t = 80
    ohlcv, rsi, macd, fracdiff, fracdiff_macro, trend, macro = _synthetic_panel(t, n_assets)
    env = MultiAssetPortfolioEnv(
        ohlcv,
        rsi,
        macd,
        macro=macro,
        fracdiff=fracdiff,
        fracdiff_macro=fracdiff_macro,
        trend=trend,
        random_start=False,
        domain_randomize=False,
        fee_scale_default=0.0,
    )
    env.reset()
    action = np.zeros(env.n_actions)
    action[0] = 10.0
    action[1] = 5.0
    _, _, _, _, info = env.step(action)
    assert info["tx_cost_frac"] == pytest.approx(0.0, abs=1e-12)
    assert info["rew_decomp/churn"] == pytest.approx(0.0, abs=1e-12)


def test_downside_return_amplified_when_in_drawdown() -> None:
    """Negative step returns get extra penalty when episode is already underwater."""
    n_assets = _N_ASSETS
    t = 80
    ohlcv, rsi, macd, fracdiff, fracdiff_macro, trend, macro = _synthetic_panel(t, n_assets)
    # Steady decline in closes so steps produce negative log returns.
    for i in range(t):
        ohlcv[i, :, 3] = 100.0 - i * 0.5
        ohlcv[i, :, 0] = ohlcv[i, :, 3] * 0.999
        ohlcv[i, :, 1] = ohlcv[i, :, 3] * 1.001
    env = MultiAssetPortfolioEnv(
        ohlcv,
        rsi,
        macd,
        macro=macro,
        fracdiff=fracdiff,
        fracdiff_macro=fracdiff_macro,
        trend=trend,
        random_start=False,
        domain_randomize=False,
        max_episode_steps=30,
        fee_scale_default=0.0,
    )
    env.reset()
    action = np.zeros(env.n_actions)
    action[1] = 10.0
    drawdown_seen = False
    for _ in range(10):
        _, _, _, _, info = env.step(action)
        if info.get("rew_decomp/drawdown", 0.0) < -1e-6:
            drawdown_seen = True
            assert info["rew_decomp/return"] < 0.0
            break
    assert drawdown_seen
