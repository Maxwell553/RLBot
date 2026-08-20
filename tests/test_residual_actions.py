"""Residual policy mapping: locked core + clipped tilts — torch-free."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rlbot.residual_actions import portfolio_weights_residual, residual_core_risky
from rlbot.rl_config import get_config, load_config, set_config
from rlbot.trading_env import portfolio_weights_from_action


def test_residual_core_equal_weight_on_live_mask() -> None:
    live = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float64)
    core = residual_core_risky(4, asset_live=live, core="equal_weight")
    assert core.sum() == pytest.approx(1.0)
    assert core[2] == pytest.approx(0.0)
    assert core[0] == pytest.approx(1.0 / 3.0)


def test_residual_zero_action_is_the_core() -> None:
    action = np.zeros(5, dtype=np.float64)
    w = portfolio_weights_residual(
        action, n_actions=5, residual_clip=0.08, residual_core="equal_weight",
        max_single_asset_weight=0.50,
    )
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(w[1:], 0.25)


def test_residual_clip_bounds_per_name_tilt() -> None:
    action = np.array([9.0, 3.0, -3.0, 0.0, 0.0], dtype=np.float64)
    w = portfolio_weights_residual(
        action, n_actions=5, residual_clip=0.08, residual_core="equal_weight",
        max_single_asset_weight=0.50,
    )
    # core 0.25 ± 0.08, negatives floored, leftover → cash or renormalize.
    assert w[1] == pytest.approx(0.33, abs=1e-6)
    assert w[2] == pytest.approx(0.17, abs=1e-6)
    assert np.all(w >= -1e-12)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w[1:] <= 0.50 + 1e-9)


def test_residual_takes_precedence_over_two_head(monkeypatch) -> None:
    cfg = load_config()
    env = replace(cfg.environment, residual_actions=True, two_head_actions=True, residual_clip=0.08)
    set_config(replace(cfg, environment=env))
    try:
        action = np.zeros(get_config().universe.n_assets + 1, dtype=np.float64)
        w = portfolio_weights_from_action(action)
        n = get_config().universe.n_assets
        assert w.shape == (n + 1,)
        assert w[0] == pytest.approx(0.0, abs=1e-6)
        assert np.allclose(w[1:], 1.0 / n, atol=1e-6)
    finally:
        set_config(load_config())
