"""Portfolio weight diagnostics from executed target-weight panels (torch-free)."""

from __future__ import annotations

import numpy as np


def _effective_n_and_hhi(risky_weights: np.ndarray, gross: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row effective number of assets and HHI on the risky sleeve."""
    n_rows, n_risky = risky_weights.shape
    eff = np.zeros(n_rows, dtype=np.float64)
    hhi = np.ones(n_rows, dtype=np.float64)
    for i in range(n_rows):
        g = float(gross[i])
        if g <= 1e-12:
            eff[i] = 0.0
            hhi[i] = 1.0
            continue
        p = risky_weights[i] / g
        s = float(np.sum(p * p))
        hhi[i] = s
        eff[i] = 1.0 / max(s, 1e-12)
    return eff, hhi


def summarize_weight_panel(
    weights: np.ndarray,
    *,
    tickers: list[str],
    max_single_asset_weight: float,
) -> dict:
    """Summarize a (T, n_actions) panel of executed target weights (cash index 0).

    Returns averages for cash, gross exposure, diversification, concentration,
    cap-binding frequency, turnover, and per-asset mean weights.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] == 0:
        return {}

    cash = w[:, 0]
    risky = w[:, 1:]
    n_risky = risky.shape[1]
    gross = np.sum(risky, axis=1)
    eff_n, hhi = _effective_n_and_hhi(risky, gross)

    top3 = np.zeros(w.shape[0], dtype=np.float64)
    for i in range(w.shape[0]):
        g = float(gross[i])
        if g <= 1e-12:
            continue
        p = np.sort(risky[i] / g)[::-1]
        top3[i] = float(np.sum(p[: min(3, n_risky)]))

    cap_floor = float(max_single_asset_weight) - 1e-4
    cap_hit = np.any(risky >= cap_floor - 1e-12, axis=1)

    turnover = np.zeros(max(w.shape[0] - 1, 0), dtype=np.float64)
    if w.shape[0] > 1:
        turnover = 0.5 * np.sum(np.abs(np.diff(w, axis=0)), axis=1)

    asset_keys = ["cash"] + list(tickers[:n_risky])
    per_asset_mean = {k: float(np.mean(w[:, j])) for j, k in enumerate(asset_keys)}

    return {
        "n_steps": int(w.shape[0]),
        "mean_cash_frac": float(np.mean(cash)),
        "mean_gross_exposure": float(np.mean(gross)),
        "mean_effective_n_assets": float(np.mean(eff_n)),
        "mean_hhi": float(np.mean(hhi)),
        "mean_top3_concentration": float(np.mean(top3)),
        "cap_hit_fraction": float(np.mean(cap_hit)),
        "mean_turnover": float(np.mean(turnover)) if turnover.size else 0.0,
        "per_asset_mean_weights": per_asset_mean,
    }
