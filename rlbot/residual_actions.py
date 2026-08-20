"""Residual action mapping: lock a fully-invested core sleeve, learn tilts only.

When ``environment.residual_actions`` is on, the policy no longer chooses cash
via softmax / the two-head exposure logit. Action[0] is ignored. Action[1:] are
per-asset deltas, clipped to ±``residual_clip`` (default 8 pp), added to an
equal-weight (or reward-benchmark) core, then projected back to the long-only
capped simplex. Leftover mass after clipping negatives sits in cash.

This is the cohort-816 training change: PPO only learns tilts around a locked
core, so it stops relearning "don't go to cash".
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


def portfolio_weights_residual(
    action: np.ndarray,
    *,
    n_actions: int | None = None,
    asset_live: np.ndarray | None = None,
    max_single_asset_weight: float | None = None,
    residual_clip: float | None = None,
    residual_core: str | None = None,
) -> np.ndarray:
    """Map ``[ignored, *asset_deltas]`` → core + clipped tilts → capped simplex.

    ``action[1:]`` is treated as a Box(-3, 3) vector and linearly mapped onto
    ``[-clip, +clip]`` percentage-point tilts vs the locked core.
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
    n_assets = n_act - 1
    core = residual_core_risky(n_assets, asset_live=asset_live, core=core_name)

    # action ∈ [-3, 3] → delta ∈ [-clip, clip]
    delta = clip * np.clip(x[1:] / _ACTION_BOUND, -1.0, 1.0)
    risky = np.maximum(core + delta, 0.0)
    if asset_live is not None:
        live = np.clip(np.asarray(asset_live, dtype=np.float64).reshape(-1), 0.0, 1.0)
        risky = risky * live
    gross = float(np.sum(risky))
    w = np.zeros(n_act, dtype=np.float64)
    if gross > 1.0:
        w[1:] = risky / gross
        w[0] = 0.0
    else:
        w[1:] = risky
        w[0] = max(0.0, 1.0 - gross)
    max_w = (
        float(max_single_asset_weight)
        if max_single_asset_weight is not None
        else float(cfg.environment.max_single_asset_weight)
    )
    return project_long_only_capped(w, max_w)
