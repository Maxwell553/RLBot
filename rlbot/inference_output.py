"""Torch-free assembly + validation of an inference target-weights payload.

Separated from scripts/infer_weights.py (which needs torch to run the policy) so the
provenance/output contract is unit-testable without the training stack."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

_SIMPLEX_TOL = 1e-6


def validate_target_weights(weights: Sequence[float], cap: float) -> None:
    """Assert a long-only simplex (cash + risky) with each risky leg ≤ cap."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] < 2:
        raise ValueError("weights must be [cash, asset_1, ...]")
    if np.any(w < -_SIMPLEX_TOL):
        raise ValueError(f"negative weight (not long-only): min={w.min():.6f}")
    if not np.isclose(w.sum(), 1.0, atol=1e-4):
        raise ValueError(f"weights must sum to 1, got {w.sum():.6f}")
    if np.any(w[1:] > cap + _SIMPLEX_TOL):
        raise ValueError(f"risky weight {w[1:].max():.6f} exceeds cap {cap}")


def build_weights_payload(
    *,
    run_id: str,
    checkpoint: str,
    as_of: str,
    weights: Sequence[float],
    tickers: Sequence[str],
    cap: float,
    asset_live: Sequence[float] | None = None,
    action_logits: Sequence[float] | None = None,
    provenance: Mapping | None = None,
) -> dict:
    """Build the audited target-weights JSON payload (validates the simplex + cap)."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    tickers = list(tickers)
    if w.shape[0] != len(tickers) + 1:
        raise ValueError(f"weights length {w.shape[0]} != n_tickers+1 ({len(tickers) + 1})")
    validate_target_weights(w, cap)
    risky = w[1:]
    live = (
        [int(round(float(x))) for x in asset_live]
        if asset_live is not None
        else [1] * len(tickers)
    )
    payload = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "as_of": as_of,
        "cap": float(cap),
        "cash_weight": float(w[0]),
        "gross_exposure": float(risky.sum()),
        "target_weights": {"CASH": float(w[0]), **{t: float(x) for t, x in zip(tickers, risky)}},
        "asset_live": {t: lv for t, lv in zip(tickers, live)},
    }
    if action_logits is not None:
        payload["action_logits"] = [float(x) for x in action_logits]
    if provenance:
        payload["provenance"] = dict(provenance)
    return payload
