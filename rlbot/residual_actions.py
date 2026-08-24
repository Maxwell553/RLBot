"""Residual action mapping: lock a fully-invested core sleeve, learn tilts only.

When ``environment.residual_actions`` is on, action[1:] are per-asset deltas,
clipped to ±``residual_clip`` (default 8 pp), added to an equal-weight (or
reward-benchmark) core, then projected back to the long-only capped simplex.

Cohort 816 (``residual_fully_invested`` off): leftover mass after flooring
negatives sits in cash — a stealth de-gross. Action[0] is ignored.
Cohort 817 turns that flag on: tilts are demeaned (sum-zero) and the live
sleeve is renormalized, so cash is only dead names in the live mask.
Action[0] is still ignored — 817 deleted 809's exposure lever.

Cohort 818 (``residual_keep_exposure`` on): keep the 817 sleeve, then scale it
by the two-head sigmoid on action[0]. Cash is a first-class gross decision,
not a leftover and not a forced 100% invest.
"""

from __future__ import annotations

import numpy as np

from rlbot.rl_config import get_config
from rlbot.two_head_actions import project_long_only_capped

RESIDUAL_CORES = frozenset({"equal_weight", "reward_benchmark"})
_ACTION_BOUND = 3.0


def residual_core_risky(
    n_assets: int,
    *,
    asset_live: np.ndarray | None = None,
    core: str = "equal_weight",
) -> np.ndarray:
    """Fully-invested risky core (sums to 1 over live assets)."""
    live = (
        np.clip(np.asarray(asset_live, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if asset_live is not None
        else np.ones(n_assets, dtype=np.float64)
    )
    if live.shape[0] != n_assets:
        raise ValueError(f"asset_live must have length {n_assets}, got {live.shape[0]}")
    mode = str(core or "equal_weight").strip().lower()
    if mode not in RESIDUAL_CORES:
        raise ValueError(f"residual_core must be one of {sorted(RESIDUAL_CORES)}, got {core!r}")
    if mode == "reward_benchmark":
        raw = get_config().reward.benchmark_cap_weights_array()
        if raw.shape[0] != n_assets:
            raise ValueError(
                f"benchmark_cap_weights length {raw.shape[0]} != n_assets {n_assets}"
            )
        sleeve = raw * live
    else:
        sleeve = live.copy()
    s = float(np.sum(sleeve))
    if s <= 1e-12:
        return np.zeros(n_assets, dtype=np.float64)
    return sleeve / s


def _sum_zero_live_tilts(delta: np.ndarray, live: np.ndarray, clip: float) -> np.ndarray:
    """Demean tilts on live names and rescale so max |tilt| stays ≤ clip."""
    out = np.array(delta, dtype=np.float64, copy=True)
    idx = live > 1e-12
    n_live = int(np.count_nonzero(idx))
    if n_live == 0:
        return out
    out[~idx] = 0.0
    slab = out[idx]
    slab = slab - float(np.mean(slab))
    peak = float(np.max(np.abs(slab)))
    if peak > clip + 1e-12 and peak > 1e-12:
        slab *= clip / peak
    out[idx] = slab
    return out


def _sigmoid_exposure(logit: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(logit, -20.0, 20.0)))))


def portfolio_weights_residual(
    action: np.ndarray,
    *,
    n_actions: int | None = None,
    asset_live: np.ndarray | None = None,
    max_single_asset_weight: float | None = None,
    residual_clip: float | None = None,
    residual_core: str | None = None,
    residual_fully_invested: bool | None = None,
    residual_keep_exposure: bool | None = None,
) -> np.ndarray:
    """Map ``[exposure_or_ignored, *asset_deltas]`` → core + clipped tilts → simplex.

    ``action[1:]`` is treated as a Box(-3, 3) vector and linearly mapped onto
    ``[-clip, +clip]`` percentage-point tilts vs the locked core. ``action[0]``
    is ignored unless ``residual_keep_exposure`` is on, in which case it is the
    two-head gross logit and the residual sleeve is scaled by ``sigmoid(action[0])``.
    """
    x = np.asarray(action, dtype=np.float64).reshape(-1)
    n_act = int(n_actions if n_actions is not None else x.shape[0])
    if x.shape[0] != n_act:
        raise ValueError(f"action must have shape ({n_act},), got {x.shape}")
    if n_act < 2:
        raise ValueError("residual actions require n_actions >= 2")

    cfg = get_config()
    clip = float(
        residual_clip
        if residual_clip is not None
        else getattr(cfg.environment, "residual_clip", 0.08)
    )
    clip = float(np.clip(clip, 0.0, 1.0))
    core_name = str(
        residual_core
        if residual_core is not None
        else getattr(cfg.environment, "residual_core", "equal_weight")
    )
    fully = (
        bool(residual_fully_invested)
        if residual_fully_invested is not None
        else bool(getattr(cfg.environment, "residual_fully_invested", False))
    )
    keep_exp = (
        bool(residual_keep_exposure)
        if residual_keep_exposure is not None
        else bool(getattr(cfg.environment, "residual_keep_exposure", False))
    )
    n_assets = n_act - 1
    live = (
        np.clip(np.asarray(asset_live, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if asset_live is not None
        else np.ones(n_assets, dtype=np.float64)
    )
    core = residual_core_risky(n_assets, asset_live=live, core=core_name)

    # action ∈ [-3, 3] → delta ∈ [-clip, clip]
    delta = clip * np.clip(x[1:] / _ACTION_BOUND, -1.0, 1.0)
    delta = delta * live
    if fully:
        delta = _sum_zero_live_tilts(delta, live, clip)
    risky = np.maximum(core + delta, 0.0) * live
    gross = float(np.sum(risky))
    w = np.zeros(n_act, dtype=np.float64)
    if fully:
        if gross > 1e-12:
            w[1:] = risky / gross
            w[0] = 0.0
        else:
            w[0] = 1.0
    elif gross > 1.0:
        w[1:] = risky / gross
        w[0] = 0.0
    else:
        w[1:] = risky
        w[0] = max(0.0, 1.0 - gross)
    if keep_exp:
        sleeve_mass = float(np.sum(w[1:]))
        if sleeve_mass > 1e-12:
            exposure = _sigmoid_exposure(float(x[0]))
            w[1:] = (w[1:] / sleeve_mass) * exposure
            w[0] = 1.0 - exposure
    max_w = (
        float(max_single_asset_weight)
        if max_single_asset_weight is not None
        else float(cfg.environment.max_single_asset_weight)
    )
    return project_long_only_capped(w, max_w)
