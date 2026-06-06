"""Inference target-weights payload (P4-1). Torch-free core; CLI import gated on torch."""

from __future__ import annotations

import numpy as np
import pytest

from rlbot.inference_output import build_weights_payload, validate_target_weights
from rlbot.rl_config import get_config
from rlbot.trading_env import portfolio_weights_from_action


def test_payload_from_valid_policy_weights() -> None:
    cap = get_config().environment.max_single_asset_weight
    tickers = list(get_config().universe.tickers)
    n_act = len(tickers) + 1
    rng = np.random.default_rng(0)
    w = portfolio_weights_from_action(rng.uniform(-3, 3, size=n_act), n_actions=n_act)
    payload = build_weights_payload(
        run_id="r1", checkpoint="best", as_of="2022-12-31",
        weights=w, tickers=tickers, cap=cap,
        asset_live=np.ones(len(tickers)),
        provenance={"config_hash": "h", "data_cache_hash": "d"},
    )
    assert payload["run_id"] == "r1"
    assert set(payload["target_weights"]) == {"CASH", *tickers}
    assert abs(sum(payload["target_weights"].values()) - 1.0) < 1e-4
    assert max(v for k, v in payload["target_weights"].items() if k != "CASH") <= cap + 1e-6
    assert payload["provenance"]["config_hash"] == "h"


def test_validate_rejects_bad_weights() -> None:
    with pytest.raises(ValueError):
        validate_target_weights([0.5, -0.1, 0.6], cap=0.5)  # negative
    with pytest.raises(ValueError):
        validate_target_weights([0.0, 0.9, 0.1], cap=0.5)  # exceeds cap
    with pytest.raises(ValueError):
        validate_target_weights([0.3, 0.3, 0.3], cap=0.5)  # doesn't sum to 1


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        build_weights_payload(
            run_id="r", checkpoint="best", as_of="x",
            weights=[0.5, 0.5], tickers=["A", "B", "C"], cap=0.5,
        )


def test_cli_importable_when_torch_present() -> None:
    pytest.importorskip("torch")
    import importlib

    mod = importlib.import_module("scripts.infer_weights")
    assert hasattr(mod, "main")
