"""Unit tests for reward-term helpers (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from rlbot.reward_terms import (
    concentration_penalty_from_weights,
    drawdown_penalty_from_nav,
)
from rlbot.rl_config import RewardConfig, get_config


def _reward_cfg(**overrides) -> RewardConfig:
    base = get_config().reward
    fields = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    fields.update(overrides)
    return RewardConfig(**fields)


def test_drawdown_penalty_increase_and_level() -> None:
    rwd = _reward_cfg(
        reward_scale=2000.0,
        drawdown_increase_penalty=0.75,
        drawdown_level_penalty=3.0,
        drawdown_level_floor=0.08,
    )
    pen, dd_next, dd_inc = drawdown_penalty_from_nav(
        peak_before=100_000.0,
        v_pre=100_000.0,
        v_next=99_000.0,
        dd_frac_pre=0.0,
        rwd=rwd,
    )
    assert dd_next == pytest.approx(0.01, rel=1e-6)
    assert dd_inc == pytest.approx(0.01, rel=1e-6)
    assert pen == pytest.approx(15.0, rel=1e-6)

    pen2, _, _ = drawdown_penalty_from_nav(
        peak_before=100_000.0,
        v_pre=85_000.0,
        v_next=85_000.0,
        dd_frac_pre=0.15,
        rwd=rwd,
    )
    assert pen2 == pytest.approx(0.21, rel=1e-6)


def test_concentration_penalty_shortfall() -> None:
    rwd = _reward_cfg(concentration_penalty=0.35, concentration_target_eff_assets=5.5)
    w = np.zeros(11, dtype=np.float64)
    w[1] = 1.0
    pen, eff_n = concentration_penalty_from_weights(w, rwd)
    assert eff_n == pytest.approx(1.0)
    assert pen == pytest.approx(0.35 * (5.5 - 1.0), rel=1e-6)

    w2 = np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2] + [0.0] * 5, dtype=np.float64)
    pen2, eff2 = concentration_penalty_from_weights(w2, rwd)
    assert eff2 == pytest.approx(5.0, rel=1e-6)
    assert pen2 == pytest.approx(0.35 * 0.5, rel=1e-6)
