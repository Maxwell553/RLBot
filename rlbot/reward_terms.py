"""Pure reward-term helpers shared by the environment (torch-free, unit-testable)."""

from __future__ import annotations

import numpy as np

from rlbot.rl_config import RewardConfig


def downside_vol_from_returns(rets: np.ndarray, floor: float) -> float:
    """Root mean square of negative returns (Sortino downside deviation), floored."""
    downside_elements = np.minimum(rets, 0.0) ** 2
    return max(float(np.sqrt(downside_elements.mean())), float(floor))


def vol_penalty_from_returns(
    agent_rets: np.ndarray,
    benchmark_rets: np.ndarray,
    rwd: RewardConfig,
) -> tuple[float, float, float]:
    """Return (penalty, agent_downside_vol, benchmark_downside_vol).

    Penalty is ``vol_penalty_scale * max(agent_downside_vol - benchmark_downside_vol, 0)``.
    """
    if rwd.vol_penalty_scale <= 0.0:
        return 0.0, 0.0, 0.0
    agent_dv = downside_vol_from_returns(agent_rets, rwd.sortino_downside_floor)
    bench_dv = downside_vol_from_returns(benchmark_rets, rwd.sortino_downside_floor)
    excess = max(agent_dv - bench_dv, 0.0)
    return float(rwd.vol_penalty_scale * excess), agent_dv, bench_dv


def concentration_penalty_from_weights(
    weights: np.ndarray,
    rwd: RewardConfig,
) -> tuple[float, float]:
    """Return (penalty, effective_n) for the risky sleeve of a weight vector."""
    gross = float(np.sum(weights[1:]))
    if gross <= 1e-12:
        return 0.0, 0.0
    p = np.asarray(weights[1:], dtype=np.float64) / gross
    hhi = float(np.sum(p * p))
    eff_n = 1.0 / max(hhi, 1e-12)
    shortfall = max(float(rwd.concentration_target_eff_assets) - eff_n, 0.0)
    return float(rwd.concentration_penalty * shortfall), eff_n


def exposure_risk_penalty_from_state(
    *,
    gross_exposure: float,
    agent_returns: np.ndarray,
    vix: float,
    rwd: RewardConfig,
) -> float:
    from rlbot.eval_selection import exposure_risk_penalty

    return exposure_risk_penalty(
        gross_exposure=gross_exposure,
        agent_returns=agent_returns,
        vix=vix,
        mode=rwd.exposure_risk_mode,
        scale=rwd.exposure_risk_penalty_scale,
    )


def drawdown_penalty_from_nav(
    *,
    peak_before: float,
    v_pre: float,
    v_next: float,
    dd_frac_pre: float,
    rwd: RewardConfig,
) -> tuple[float, float, float]:
    """Return (penalty, dd_next, dd_increase) using post-step drawdown state."""
    peak = max(float(peak_before), 1e-12)
    dd_next = max(0.0, (peak - float(v_next)) / peak)
    dd_increase = max(dd_next - float(dd_frac_pre), 0.0)
    dd_excess = max(dd_next - float(rwd.drawdown_level_floor), 0.0)
    penalty = (
        dd_increase * rwd.reward_scale * rwd.drawdown_increase_penalty
        + dd_excess * rwd.drawdown_level_penalty
    )
    return float(penalty), dd_next, dd_increase
