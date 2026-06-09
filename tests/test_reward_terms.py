"""Per-term reward values vs config coefficients via info["rew_decomp/*"]
(inactivity, participation, churn, sortino floor, full decomposition). Torch-free."""

from __future__ import annotations

import numpy as np
import pytest

from rlbot.data_utils import MACRO_VIX_INDEX
from rlbot.rl_config import get_config
from rlbot.trading_env import (
    VIX_CHURN_BASELINE,
    MultiAssetPortfolioEnv,
    portfolio_weights_from_action,
)

_N = get_config().universe.n_assets
_N_ACT = _N + 1


def _flat_panel(t: int = 80, vix: float = VIX_CHURN_BASELINE):
    """Constant prices (no returns / drawdown) with a controllable VIX level."""
    ohlcv = np.zeros((t, _N, 5), dtype=np.float64)
    ohlcv[:, :, :4] = 100.0
    ohlcv[:, :, 4] = 1e6
    macro = np.full((t, 4), 10.0)
    macro[:, MACRO_VIX_INDEX] = vix
    return ohlcv, macro


def _make_env(ohlcv: np.ndarray, macro: np.ndarray) -> MultiAssetPortfolioEnv:
    t = ohlcv.shape[0]
    return MultiAssetPortfolioEnv(
        ohlcv,
        np.full((t, _N), 50.0),
        np.zeros((t, _N)),
        macro=macro,
        fracdiff=np.zeros((t, _N)),
        fracdiff_macro=np.zeros((t, 4)),
        trend=np.zeros((t, _N)),
        random_start=False,
        domain_randomize=False,
        action_smoothing_alpha=0.0,
    )


def _cash_action() -> np.ndarray:
    a = np.full(_N_ACT, -30.0)
    a[0] = 30.0
    return a


# ── (a) inactivity penalty at ~100% cash ─────────────────────────────────
def test_inactivity_penalty_at_full_cash_matches_config() -> None:
    rwd = get_config().reward
    env = _make_env(*_flat_panel())
    env.reset()
    _, reward, _, _, info = env.step(_cash_action())
    cash_frac = env._cash / info["nav"]
    assert cash_frac == pytest.approx(1.0, abs=1e-9)  # saturated cash action
    expected = cash_frac * rwd.inactivity_penalty_over_50
    expected += ((cash_frac - 0.90) / 0.10) * rwd.inactivity_penalty_over_90
    assert info["rew_decomp/inactivity"] == pytest.approx(-expected, rel=1e-9)
    # with the shipped config this is the full -(10 + 15) at 100% cash
    assert info["rew_decomp/inactivity"] == pytest.approx(
        -(rwd.inactivity_penalty_over_50 + rwd.inactivity_penalty_over_90), abs=1e-6
    )
    # flat prices, no positions: inactivity dominates the reward
    assert reward == pytest.approx(
        info["rew_decomp/inactivity"] + info["rew_decomp/participation"], abs=1e-9
    )


def test_inactivity_scale_parameter_scales_linear_term() -> None:
    rwd = get_config().reward
    ohlcv, macro = _flat_panel()
    t = ohlcv.shape[0]
    env = MultiAssetPortfolioEnv(
        ohlcv,
        np.full((t, _N), 50.0),
        np.zeros((t, _N)),
        macro=macro,
        fracdiff=np.zeros((t, _N)),
        fracdiff_macro=np.zeros((t, 4)),
        trend=np.zeros((t, _N)),
        random_start=False,
        domain_randomize=False,
        action_smoothing_alpha=0.0,
        inactivity_penalty_scale=0.25,
    )
    env.reset()
    _, _, _, _, info = env.step(_cash_action())
    expected = 0.25 * (rwd.inactivity_penalty_over_50 + rwd.inactivity_penalty_over_90)
    assert info["rew_decomp/inactivity"] == pytest.approx(-expected, rel=1e-9)


# ── (b) participation bonus ──────────────────────────────────────────────
def test_participation_bonus_equals_gross_exposure_times_config() -> None:
    rwd = get_config().reward
    env = _make_env(*_flat_panel())
    env.reset()
    action = np.zeros(_N_ACT)
    live = env.asset_live[max(env._t - env.obs_lag, 0)]
    w = portfolio_weights_from_action(action, n_actions=_N_ACT, asset_live=live)
    gross = float(np.sum(w[1:]))
    assert gross > 0.5  # uniform logits put ~N/(N+1) into risky sleeves
    _, _, _, _, info = env.step(action)
    expected = gross * rwd.participation_bonus * rwd.participation_reward_scale
    assert info["rew_decomp/participation"] == pytest.approx(expected, rel=1e-9)


# ── (c) churn penalty ────────────────────────────────────────────────────
def test_churn_equals_turnover_times_penalty_at_baseline_vix() -> None:
    rwd = get_config().reward
    env = _make_env(*_flat_panel(vix=VIX_CHURN_BASELINE))  # vix_mult == 1.0
    env.reset()
    _, _, _, _, info = env.step(np.zeros(_N_ACT))
    assert info["rew_decomp/vix_churn_mult"] == pytest.approx(1.0)
    turnover = info["turnover"]
    assert turnover > 0.1  # the first rebalance buys ~N/(N+1) of NAV
    assert info["rew_decomp/churn"] == pytest.approx(-turnover * rwd.churn_penalty, rel=1e-9)


def test_churn_scales_with_vix_multiplier_and_curriculum_scale() -> None:
    rwd = get_config().reward
    # high-VIX panel: mult = clip(36/18, 0.75, 1.5) = 1.5
    env = _make_env(*_flat_panel(vix=2.0 * VIX_CHURN_BASELINE))
    env.reset()
    _, _, _, _, info = env.step(np.zeros(_N_ACT))
    assert info["rew_decomp/vix_churn_mult"] == pytest.approx(1.5)
    assert info["rew_decomp/churn"] == pytest.approx(
        -info["turnover"] * 1.5 * rwd.churn_penalty, rel=1e-9
    )
    # curriculum churn scale multiplies in
    env2 = _make_env(*_flat_panel(vix=VIX_CHURN_BASELINE))
    env2.set_curriculum_state(None, 0.5)
    env2.reset()
    _, _, _, _, info2 = env2.step(np.zeros(_N_ACT))
    assert info2["rew_decomp/churn"] == pytest.approx(
        -info2["turnover"] * 1.0 * 0.5 * rwd.churn_penalty, rel=1e-9
    )
    # same turnover (identical action/prices) → churn strictly proportional to the scales
    assert info["turnover"] == pytest.approx(info2["turnover"], rel=1e-9)
    assert info["rew_decomp/churn"] == pytest.approx(3.0 * info2["rew_decomp/churn"], rel=1e-9)


# ── (d) sortino downside floor ───────────────────────────────────────────
def test_compute_sortino_uses_config_floor_for_no_loss_window() -> None:
    rwd = get_config().reward
    env = _make_env(*_flat_panel())
    rets = np.full(rwd.risk_window, 0.0005)  # all non-negative: downside dev is 0
    s = env._compute_sortino(rets)
    assert s == pytest.approx(rets.mean() / rwd.sortino_downside_floor, rel=1e-12)
    # config floor (0.001) keeps a realistic no-loss mean far from the ±3 clip ...
    assert rwd.sortino_downside_floor == pytest.approx(0.001)
    assert abs(s) < 3.0
    # ... while the legacy 1e-4 floor would have saturated the clipped differential
    assert rets.mean() / 1e-4 > 3.0


def test_compute_sortino_real_downside_ignores_floor() -> None:
    env = _make_env(*_flat_panel())
    rets = np.array([0.01, -0.02, 0.005, -0.03, 0.0, 0.015] * 10)
    downside = np.sqrt((np.minimum(rets, 0.0) ** 2).mean())
    assert downside > get_config().reward.sortino_downside_floor
    assert env._compute_sortino(rets) == pytest.approx(rets.mean() / downside, rel=1e-12)


# ── (e) decomposition completeness ───────────────────────────────────────
_DECOMP_KEYS = (
    "rew_decomp/return",
    "rew_decomp/sortino",
    "rew_decomp/inactivity",
    "rew_decomp/participation",
    "rew_decomp/churn",
    "rew_decomp/drawdown",
)


def test_rew_decomp_terms_sum_to_reward() -> None:
    rng = np.random.default_rng(7)
    t = 120
    rets = rng.normal(0.0, 0.015, size=(t, _N))
    price = 100.0 * np.exp(np.cumsum(rets, axis=0))
    ohlcv = np.zeros((t, _N, 5), dtype=np.float64)
    for c in range(4):
        ohlcv[:, :, c] = price
    ohlcv[:, :, 0] *= 1.001  # opens differ from closes
    ohlcv[:, :, 4] = 1e6
    macro = np.full((t, 4), 10.0)
    macro[:, MACRO_VIX_INDEX] = 22.0
    env = _make_env(ohlcv, macro)
    env.reset()
    sortino_seen = False
    for step in range(40):
        action = rng.uniform(-3.0, 3.0, size=_N_ACT)
        _, reward, terminated, truncated, info = env.step(action)
        for key in _DECOMP_KEYS:
            assert key in info, f"missing {key} at step {step}"
        assert "rew_decomp/vix_churn_mult" in info
        total = sum(info[k] for k in _DECOMP_KEYS)
        assert reward == pytest.approx(total, abs=1e-9), f"step {step}"
        sortino_seen = sortino_seen or abs(info["rew_decomp/sortino"]) > 1e-12
        if terminated or truncated:
            break
    # the sortino term actually activated (>= sortino_min_steps elapsed)
    assert sortino_seen
