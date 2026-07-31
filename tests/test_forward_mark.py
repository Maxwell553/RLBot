"""Forward-mark payload helpers (torch-free)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from rlbot.forward_mark import build_forward_mark_payload, call_with_timeout


def test_call_with_timeout_returns_and_does_not_hang_on_overrun() -> None:
    assert call_with_timeout(lambda: 7, 1.0) == 7

    def _slow() -> None:
        time.sleep(5)

    t0 = time.perf_counter()
    with pytest.raises(TimeoutError):
        call_with_timeout(_slow, 0.2)
    # Must return promptly — not wait for the sleepy worker (wait=False shutdown).
    assert time.perf_counter() - t0 < 1.5


def test_build_forward_mark_payload_rebases_and_stats() -> None:
    dates = ["2026-07-28", "2026-07-29", "2026-07-30"]
    model = np.array([100_000.0, 101_000.0, 102_000.0])
    spy = np.array([50.0, 51.0, 52.0])  # different start — must rebase
    ew = np.array([200.0, 198.0, 202.0])
    weights = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.2, 0.2, 0.6],
            [0.15, 0.25, 0.6],
        ]
    )
    payload = build_forward_mark_payload(
        run_id="LIVE_TEST",
        checkpoint_label="best",
        dates=dates,
        nav_model=model,
        nav_spy=spy,
        nav_ew=ew,
        weights=weights,
        asset_labels=["Cash", "SP500", "GOLD"],
        initial_cash=100_000.0,
        holdout_start="2026-07-28",
        holdout_end="2027-12-31",
    )
    assert payload["n_bars"] == 3
    assert payload["nav"]["model"][0] == 100_000.0
    assert payload["nav"]["spy"][0] == 100_000.0
    assert payload["nav"]["equal_weight"][0] == 100_000.0
    assert abs(payload["stats"]["model"]["total_return"] - 0.02) < 1e-12
    assert payload["latest_weights"]["Cash"] == 0.15
    assert len(payload["weights"]) == 3
