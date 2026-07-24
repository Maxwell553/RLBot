"""Two-head action mapping: exposure head + risky allocation head.

Default policy remains a single softmax over cash+assets. When
``environment.two_head_actions`` is enabled, action[0] chooses gross risky
exposure via a sigmoid and action[1:] allocate that exposure across assets.

This changes the policy class — implement and unit-test now, but do **not** mix
with reward/curriculum experiments until validated in isolation.
"""

from __future__ import annotations

import numpy as np

from rlbot.rl_config import get_config


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64).reshape(-1)
    z = z - np.max(z)
    e = np.exp(z)
    s = float(np.sum(e))
    if s <= 0.0 or not np.isfinite(s):
        return np.ones_like(z) / max(len(z), 1)
    return e / s


def _enforce_long_only_simplex(w: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(w, dtype=np.float64), 0.0)
    s = float(np.sum(out))
    if s <= 1e-12:
        out[:] = 0.0
        out[0] = 1.0
        return out
    return out / s


def _clip_redistribute_cap(w: np.ndarray, max_w: float) -> np.ndarray:
    """Per-risky-asset cap with overflow parked in cash (same post-condition as legacy)."""
    out = np.asarray(w, dtype=np.float64).copy()
    risky = out[1:].copy()
    for _ in range(5):
        overflow_mask = risky > max_w
        if not np.any(overflow_mask):
            break
        overflow = float(np.sum(risky[overflow_mask] - max_w))
        risky[overflow_mask] = max_w
        underflow_mask = (risky < max_w) & (risky > 0.0)
        total_underflow = float(np.sum(risky[underflow_mask]))
        if total_underflow > 1e-12:
            risky[underflow_mask] += (risky[underflow_mask] / total_underflow) * overflow
        else:
            out[0] += overflow
            break
    out[1:] = risky
    out = _enforce_long_only_simplex(out)
    over = out[1:] > max_w + 1e-9
    if np.any(over):
        excess = float(np.sum(out[1:][over] - max_w))
        out[1:][over] = max_w
        out[0] += excess
    return out


def portfolio_weights_two_head(
    action: np.ndarray,
    *,
    n_actions: int | None = None,
    asset_live: np.ndarray | None = None,
    max_single_asset_weight: float | None = None,
) -> np.ndarray:
    """Map ``[exposure_logit, *asset_logits]`` → long-only capped simplex.

    ``exposure = sigmoid(action[0])`` is the gross risky weight; cash gets ``1 - exposure``.
    Remaining logits softmax over live risky assets and scale by exposure.
    """
    x = np.asarray(action, dtype=np.float64).reshape(-1)
    n_act = int(n_actions if n_actions is not None else x.shape[0])
    if x.shape[0] != n_act:
        raise ValueError(f"action must have shape ({n_act},), got {x.shape}")
    if n_act < 2:
        raise ValueError("two-head actions require n_actions >= 2")

    exposure = 1.0 / (1.0 + np.exp(-float(np.clip(x[0], -20.0, 20.0))))
    asset_logits = x[1:]
    if asset_live is not None:
        live = np.asarray(asset_live, dtype=np.float64).reshape(-1)
        if live.shape[0] != n_act - 1:
            raise ValueError(f"asset_live must have length {n_act - 1}, got {live.shape[0]}")
        # Mask dead assets by driving their logits to -inf before softmax.
        masked = asset_logits.copy()
        dead = live < 0.5
        masked[dead] = -1e9
        if np.all(dead):
            alloc = np.zeros(n_act - 1, dtype=np.float64)
            exposure = 0.0
        else:
            alloc = _softmax_1d(masked)
            alloc[dead] = 0.0
            s = float(np.sum(alloc))
            alloc = alloc / s if s > 1e-12 else alloc
    else:
        alloc = _softmax_1d(asset_logits)

    w = np.zeros(n_act, dtype=np.float64)
    w[0] = 1.0 - exposure
    w[1:] = alloc * exposure
    w = _enforce_long_only_simplex(w)
    max_w = (
        float(max_single_asset_weight)
        if max_single_asset_weight is not None
        else float(get_config().environment.max_single_asset_weight)
    )
    return _clip_redistribute_cap(w, max_w)
