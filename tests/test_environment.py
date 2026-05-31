"""Portfolio weight mapping: simplex constraints and per-asset cap bounds."""

from __future__ import annotations

import numpy as np
import pytest

from rl_config import get_config
from trading_env import N_ACTIONS, N_ASSETS, portfolio_weights_from_action

_CAP_TOL = 1e-6
_SIMPLEX_TOL = 1e-8


def _max_cap() -> float:
    return float(get_config().environment.max_single_asset_weight)


def _assert_valid_weights(w: np.ndarray) -> None:
    """Long-only simplex on cash + risky sleeves; each risky leg ≤ config cap."""
    assert w.shape == (N_ACTIONS,)
    assert w.dtype == np.float64
    assert np.all(w >= -_SIMPLEX_TOL)
    assert np.isclose(w.sum(), 1.0, atol=_SIMPLEX_TOL)
    cap = _max_cap()
    assert np.all(w[1:] <= cap + _CAP_TOL), f"max risky weight {w[1:].max():.6f} > cap {cap}"


def test_config_cap_is_half() -> None:
    assert _max_cap() == pytest.approx(0.50)


def test_simplex_uniform_logits() -> None:
    w = portfolio_weights_from_action(np.zeros(N_ACTIONS))
    _assert_valid_weights(w)


def test_simplex_all_cash_preference() -> None:
    action = np.full(N_ACTIONS, -10.0)
    action[0] = 20.0
    w = portfolio_weights_from_action(action)
    _assert_valid_weights(w)
    assert w[0] > 0.5


def test_cap_binds_single_asset_spike() -> None:
    action = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w = portfolio_weights_from_action(action)
    _assert_valid_weights(w)
    assert w[1] == pytest.approx(_max_cap(), abs=_CAP_TOL)


def test_cap_binds_multiple_spikes_redistribute() -> None:
    """Three equal spikes: each risky leg capped; overflow to cash or other sleeves."""
    action = np.array([-5.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w = portfolio_weights_from_action(action)
    _assert_valid_weights(w)
    active = w[1:][w[1:] > 1e-9]
    assert len(active) >= 1
    assert np.all(active <= _max_cap() + _CAP_TOL)


def test_random_actions_respect_simplex_and_cap() -> None:
    rng = np.random.default_rng(42)
    for _ in range(500):
        action = rng.uniform(-3.0, 3.0, size=N_ACTIONS)
        _assert_valid_weights(portfolio_weights_from_action(action))


def test_extreme_logits_still_valid() -> None:
    for scale in (50.0, 100.0, -100.0):
        action = np.linspace(-scale, scale, N_ACTIONS)
        _assert_valid_weights(portfolio_weights_from_action(action))


def test_action_must_have_eleven_elements() -> None:
    with pytest.raises(ValueError, match="action must have shape"):
        portfolio_weights_from_action(np.zeros(N_ACTIONS - 1))


def test_risky_mass_at_most_ten_times_cap() -> None:
    """Structural bound: max risky exposure ≤ 10 × cap (cash holds the rest)."""
    cap = _max_cap()
    w = portfolio_weights_from_action(np.ones(N_ACTIONS))
    _assert_valid_weights(w)
    assert float(w[1:].sum()) <= N_ASSETS * cap + _CAP_TOL
