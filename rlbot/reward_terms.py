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
    *,
    gross_exposure: float = 1.0,
) -> tuple[float, float, float]:
    """Return (penalty, agent_downside_vol, benchmark_downside_vol).

    ``vol_penalty_mode="excess"`` (legacy):
      ``scale * reward_scale * max(agent_dv - bench_dv, 0)``
      — dead when the agent is calmer than the equal-weight book (common under cash).

    ``vol_penalty_mode="absolute"`` (cohort 729+):
      ``scale * reward_scale * agent_dv * gross``
      — taxes invested downside vol even when below the passive benchmark, so
      risk-on books cannot hide behind a volatile EW sleeve.
    """
    if rwd.vol_penalty_scale <= 0.0:
        return 0.0, 0.0, 0.0
    agent_dv = downside_vol_from_returns(agent_rets, rwd.sortino_downside_floor)
    bench_dv = downside_vol_from_returns(benchmark_rets, rwd.sortino_downside_floor)
    mode = str(getattr(rwd, "vol_penalty_mode", "excess")).lower()
    if mode == "absolute":
        g = float(np.clip(gross_exposure, 0.0, 1.0))
        pen = float(rwd.vol_penalty_scale * rwd.reward_scale * agent_dv * g)
    else:
        excess = max(agent_dv - bench_dv, 0.0)
        pen = float(rwd.vol_penalty_scale * rwd.reward_scale * excess)
    return pen, agent_dv, bench_dv


def drawdown_amp_factor(dd_frac_pre: float, rwd: RewardConfig) -> float:
    """Amplification on negative step returns: ``1 + gamma * dd_pre``, capped.

    ``drawdown_amp_max > 0`` bounds the factor so worst-day reward outliers stay
    within the VecNormalize clip band instead of all saturating identically;
    ``0`` = uncapped (legacy behavior for old run snapshots).
    """
    amp = 1.0 + float(rwd.drawdown_downside_gamma) * max(float(dd_frac_pre), 0.0)
    cap = float(rwd.drawdown_amp_max)
    if cap > 0.0:
        amp = min(amp, max(cap, 1.0))
    return float(amp)


def inactivity_vix_relief_multiplier(
    vix: float,
    relief_scale: float,
    baseline: float = 18.0,
) -> float:
    """Multiplier in [0, 1] applied to the inactivity penalty.

    Relief grows as VIX rises above the calm baseline:
    ``1 - clip((vix / baseline - 1) * relief_scale, 0, 1)``. With scale 1.0 the
    penalty is halved at VIX 27 and fully waived at VIX >= 36, so holding cash in
    stress regimes is not punished like idling in a calm market. ``vix <= 1``
    (missing/degenerate macro data) and ``relief_scale <= 0`` return 1.0.
    """
    if relief_scale <= 0.0 or float(vix) <= 1.0:
        return 1.0
    relief = float(np.clip((float(vix) / float(baseline) - 1.0) * float(relief_scale), 0.0, 1.0))
    return 1.0 - relief


def inactivity_drawdown_relief_multiplier(
    dd_frac: float,
    floor: float,
    relief_scale: float,
    relief_span: float,
) -> float:
    """Multiplier in [0, 1] on cash/participation shaping from the agent's own DD.

    Relief grows as drawdown rises above ``floor``:
    ``1 - clip((dd - floor) / span, 0, 1) * relief_scale``. With scale 1.0 and
    span 0.10, inactivity/participation are fully waived once dd >= floor + 0.10
    (15% off peak at the default 5% floor) — cash becomes the escape hatch from
    sustained underwater risk, not a calm-market idling tax. ``relief_scale <= 0``
    or ``relief_span <= 0`` return 1.0 (legacy: no DD-conditional relief).
    """
    if relief_scale <= 0.0 or relief_span <= 0.0:
        return 1.0
    excess = max(float(dd_frac) - float(floor), 0.0)
    relief = float(np.clip(excess / float(relief_span), 0.0, 1.0)) * float(relief_scale)
    return 1.0 - min(relief, 1.0)


def shaping_stress_relief_multiplier(
    *,
    vix: float,
    dd_frac: float,
    vix_relief: float,
    drawdown_relief: float,
    drawdown_floor: float,
    drawdown_relief_span: float,
    vix_baseline: float = 18.0,
) -> float:
    """Most-relief-wins combine of VIX and own-drawdown shaping multipliers.

    Either stress channel can waive cash/participation shaping; taking the min
    avoids requiring both VIX and NAV drawdown before cash is allowed.
    """
    vix_mult = inactivity_vix_relief_multiplier(vix, vix_relief, baseline=vix_baseline)
    dd_mult = inactivity_drawdown_relief_multiplier(
        dd_frac, drawdown_floor, drawdown_relief, drawdown_relief_span
    )
    return float(min(vix_mult, dd_mult))


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


def drawdown_level_exposure_factor(gross_exposure: float, coupling: float) -> float:
    """Scale factor in [0, 1] applied to the persistent drawdown *level* term.

    ``coupling = 0`` → always 1 (legacy: level penalty accrues in cash too).
    ``coupling = 1`` → factor equals clipped gross exposure, so fully de-risking
    to cash zeros the level term — the structural escape hatch that was missing
    when level penalty punished underwater NAV regardless of book risk.
    Intermediate couplings blend: ``(1 - c) + c * gross``.
    """
    c = float(np.clip(coupling, 0.0, 1.0))
    if c <= 0.0:
        return 1.0
    g = float(np.clip(gross_exposure, 0.0, 1.0))
    return float((1.0 - c) + c * g)


def drawdown_penalty_from_nav(
    *,
    peak_before: float,
    v_pre: float,
    v_next: float,
    dd_frac_pre: float,
    rwd: RewardConfig,
    gross_exposure: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """Return ``(penalty, dd_next, dd_increase, increase_term, level_term)``.

    ``increase_term`` is the fresh-expansion component (× ``reward_scale``);
    ``level_term`` is the persistent underwater component, optionally scaled by
    gross exposure when ``drawdown_level_exposure_coupling > 0`` so that
    de-risking to cash stops the ongoing level tax. When
    ``drawdown_level_times_reward_scale`` is True, level also multiplies
    ``reward_scale`` (731+); legacy snapshots keep raw units.
    """
    peak = max(float(peak_before), 1e-12)
    dd_next = max(0.0, (peak - float(v_next)) / peak)
    dd_increase = max(dd_next - float(dd_frac_pre), 0.0)
    dd_excess = max(dd_next - float(rwd.drawdown_level_floor), 0.0)
    increase_term = float(dd_increase * rwd.reward_scale * rwd.drawdown_increase_penalty)
    level_raw = float(dd_excess * rwd.drawdown_level_penalty)
    if bool(getattr(rwd, "drawdown_level_times_reward_scale", False)):
        level_raw *= float(rwd.reward_scale)
    level_term = level_raw * drawdown_level_exposure_factor(
        gross_exposure, float(rwd.drawdown_level_exposure_coupling)
    )
    penalty = increase_term + level_term
    return float(penalty), dd_next, dd_increase, increase_term, level_term
