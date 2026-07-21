"""Unit tests for reward-term helpers (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from rlbot.reward_terms import (
    concentration_penalty_from_weights,
    downside_vol_from_returns,
    drawdown_amp_factor,
    drawdown_penalty_from_nav,
    inactivity_vix_relief_multiplier,
    vol_penalty_from_returns,
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


def test_drawdown_amp_factor_uncapped_and_capped() -> None:
    uncapped = _reward_cfg(drawdown_downside_gamma=12.0, drawdown_amp_max=0.0)
    assert drawdown_amp_factor(0.0, uncapped) == pytest.approx(1.0)
    assert drawdown_amp_factor(0.10, uncapped) == pytest.approx(2.2)
    assert drawdown_amp_factor(0.50, uncapped) == pytest.approx(7.0)  # legacy: unbounded

    capped = _reward_cfg(drawdown_downside_gamma=12.0, drawdown_amp_max=4.0)
    assert drawdown_amp_factor(0.10, capped) == pytest.approx(2.2)  # below cap unchanged
    assert drawdown_amp_factor(0.25, capped) == pytest.approx(4.0)  # saturation point
    assert drawdown_amp_factor(0.50, capped) == pytest.approx(4.0)  # bounded


def test_inactivity_vix_relief_multiplier() -> None:
    # Disabled scale or degenerate VIX → no relief (legacy behavior).
    assert inactivity_vix_relief_multiplier(36.0, 0.0) == pytest.approx(1.0)
    assert inactivity_vix_relief_multiplier(0.0, 1.0) == pytest.approx(1.0)
    # Calm market → full penalty; stress → progressively waived.
    assert inactivity_vix_relief_multiplier(15.0, 1.0) == pytest.approx(1.0)
    assert inactivity_vix_relief_multiplier(18.0, 1.0) == pytest.approx(1.0)
    assert inactivity_vix_relief_multiplier(27.0, 1.0) == pytest.approx(0.5)
    assert inactivity_vix_relief_multiplier(36.0, 1.0) == pytest.approx(0.0)
    assert inactivity_vix_relief_multiplier(80.0, 1.0) == pytest.approx(0.0)  # floor at 0
    # Steeper scale reaches full relief sooner.
    assert inactivity_vix_relief_multiplier(27.0, 2.0) == pytest.approx(0.0)


def test_participation_vix_relief_parse_and_validation() -> None:
    import copy

    from rlbot.rl_config import _parse_config, _validate_reward_config

    # Active config ships relief 1.0 on both shaping terms (cash-vs-invested
    # neutral at VIX >= 36).
    assert get_config().reward.participation_vix_relief == pytest.approx(1.0)

    # Legacy run snapshots without the key parse to 0 (bonus always paid).
    raw = copy.deepcopy(get_config().raw)
    raw["reward"].pop("participation_vix_relief", None)
    parsed = _parse_config(raw, get_config().path)
    assert parsed.reward.participation_vix_relief == 0.0

    with pytest.raises(ValueError, match="participation_vix_relief"):
        _validate_reward_config(_reward_cfg(participation_vix_relief=-0.5))


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


def test_downside_vol_uses_floor_on_no_losses() -> None:
    rets = np.array([0.01, 0.02, 0.015], dtype=np.float64)
    assert downside_vol_from_returns(rets, floor=0.001) == pytest.approx(0.001)


def test_vol_penalty_only_on_excess_downside_vol() -> None:
    rwd = _reward_cfg(vol_penalty_scale=300.0, sortino_downside_floor=0.001)
    agent = np.array([-0.02, -0.01, 0.01, 0.0], dtype=np.float64)
    bench = np.array([-0.01, -0.005, 0.01, 0.0], dtype=np.float64)
    agent_dv = downside_vol_from_returns(agent, rwd.sortino_downside_floor)
    bench_dv = downside_vol_from_returns(bench, rwd.sortino_downside_floor)
    pen, got_agent, got_bench = vol_penalty_from_returns(agent, bench, rwd)
    assert got_agent == pytest.approx(agent_dv)
    assert got_bench == pytest.approx(bench_dv)
    assert pen == pytest.approx(300.0 * rwd.reward_scale * max(agent_dv - bench_dv, 0.0))

    calmer = np.array([-0.005, 0.01, 0.0, 0.0], dtype=np.float64)
    pen0, _, _ = vol_penalty_from_returns(calmer, bench, rwd)
    assert pen0 == pytest.approx(0.0)

    pen_off, _, _ = vol_penalty_from_returns(agent, bench, _reward_cfg(vol_penalty_scale=0.0))
    assert pen_off == pytest.approx(0.0)
