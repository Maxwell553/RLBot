"""Portfolio weight mapping: simplex constraints and per-asset cap bounds."""

from __future__ import annotations

import numpy as np
import pytest

from rlbot.rl_config import get_config, observation_dim_for_universe
from rlbot.trading_env import MultiAssetPortfolioEnv, portfolio_weights_from_action

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
        fracdiff,
        fracdiff_macro,
        trend,
        macro=macro,
        random_start=False,
        domain_randomize=False,
    )
    assert env.n_assets == n_assets
    assert env.n_actions == n_assets + 1
    assert env.observation_space.shape[0] == observation_dim_for_universe(n_assets)
