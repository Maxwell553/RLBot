"""Portfolio weight diagnostics (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from rlbot.portfolio_metrics import summarize_weight_panel


def test_cap_hit_fraction_at_projected_cap() -> None:
    """Near-cap detection: weights projected to cap should count as cap hits."""
    w = np.array([[0.5, 0.25, 0.25] + [0.0] * 8], dtype=np.float64)
    tickers = [f"A{i}" for i in range(10)]
    s = summarize_weight_panel(w, tickers=tickers, max_single_asset_weight=0.25)
    assert s["cap_hit_fraction"] == pytest.approx(1.0)


def test_summarize_weight_panel_basic() -> None:
    w = np.array(
        [
            [0.1, 0.45, 0.45] + [0.0] * 8,
            [0.2, 0.4, 0.4] + [0.0] * 8,
        ],
        dtype=np.float64,
    )
    tickers = [f"A{i}" for i in range(10)]
    s = summarize_weight_panel(w, tickers=tickers, max_single_asset_weight=0.25)
    assert s["n_steps"] == 2
    assert s["mean_cash_frac"] == pytest.approx(0.15)
    assert s["mean_gross_exposure"] == pytest.approx(0.85)
    assert s["mean_turnover"] == pytest.approx(0.1)
    assert s["cap_hit_fraction"] == pytest.approx(1.0)
    assert s["per_asset_mean_weights"]["A0"] == pytest.approx(0.425)
