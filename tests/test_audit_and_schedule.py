"""Tests for cohort labels, curriculum preflight, checkpoint selection, two-head actions."""

from __future__ import annotations

import numpy as np
import pytest

from rlbot.checkpoint_selection import (
    blend_canonical_stress,
    trailing_aggregate,
    update_confirmation,
)
from rlbot.cohort_labels import label_run
from rlbot.curriculum_preflight import build_curriculum_preflight
from rlbot.rl_config import get_config, load_config, set_config
from rlbot.two_head_actions import portfolio_weights_two_head


def test_label_invalid_623_625_not_622_bundled() -> None:
    a = label_run("W1_623")
    assert "invalid_vol_penalty_scale" in a.labels
    assert not a.comparable

    b = label_run("W2_622")
    assert "vol_penalty_bug_present" in b.labels
    assert "invalid_vol_penalty_scale" not in b.labels
    assert not b.comparable


def test_label_resumed_619_windows() -> None:
    for rid in ("W3_619", "W4_619", "W5_619"):
        lab = label_run(rid)
        assert "resumed_long_budget" in lab.labels
        assert not lab.comparable
    clean = label_run("W1_619")
    assert "resumed_long_budget" not in clean.labels


def test_curriculum_preflight_default_early_stop_reachable() -> None:
    cfg = load_config()
    pf = build_curriculum_preflight(cfg, budget=50_000_000)
    # Rephased schedule: fee-free 10%, full fees/gate 45%, DR widen span 25%.
    assert pf.fee_free_until == 5_000_000
    assert pf.fee_ramp_end == 22_500_000
    assert pf.best_model_min_step == 22_500_000
    # span 0.25 × 50M = 12.5M → full DR at 35M, 15M settled residual.
    assert pf.dr_widen_end == 35_000_000
    assert pf.stationary_full_dr_steps == 15_000_000
    # Cohort 729+: early stop off (full 50M budget); patience=0 → not "reachable".
    assert pf.early_stop_patience == 0
    assert pf.early_stop_reachable is False
    assert not any("early_stop" in w for w in pf.warnings)
    # Phase-aware LR: full LR through DR widening, decayed after.
    assert pf.lr_schedule == "phase_aware"
    assert pf.lr_hold_until_step == 35_000_000
    by_name = {m.name: m for m in pf.milestones}
    assert by_name["fee_ramp_end / checkpoint_gate"].learning_rate == pytest.approx(3.0e-4)
    assert by_name["full_DR_start"].learning_rate == pytest.approx(3.0e-4)
    assert by_name["budget_end"].learning_rate == pytest.approx(1.0e-6, abs=1e-9)


def test_curriculum_preflight_long_dr_makes_early_stop_unreachable() -> None:
    cfg = load_config()
    from dataclasses import replace

    # Force patience on so the unreachable warning path is exercised.
    tr = replace(cfg.training, early_stop_patience=12)
    cur = replace(cfg.curriculum, dr_widen_span_fraction=0.65)
    cfg2 = replace(cfg, training=tr, curriculum=cur)
    pf = build_curriculum_preflight(cfg2, budget=50_000_000)
    assert pf.dr_widen_end == pf.budget
    assert pf.stationary_full_dr_steps == 0
    assert pf.early_stop_reachable is False
    assert any("early_stop" in w for w in pf.warnings)


def test_curriculum_preflight_shorter_dr_makes_early_stop_reachable() -> None:
    cfg = load_config()
    from dataclasses import replace

    cur = replace(cfg.curriculum, dr_widen_span_fraction=0.10)
    cfg2 = replace(cfg, curriculum=cur)
    pf = build_curriculum_preflight(cfg2, budget=50_000_000)
    assert pf.dr_widen_end < pf.budget
    assert pf.stationary_full_dr_steps > 0
    if pf.early_stop_patience > 0:
        assert pf.stationary_full_dr_steps == pf.budget - pf.dr_widen_end


def test_trailing_aggregate_and_confirmation() -> None:
    assert trailing_aggregate([1.0, 2.0, 3.0], window=3, agg="median") == 2.0
    assert trailing_aggregate([1.0, 2.0, 3.0], window=2, agg="mean") == pytest.approx(2.5)
    state, replace = update_confirmation(None, score=1.0, best_score=0.0, confirms_needed=0)
    assert replace is True and state is None
    state, replace = update_confirmation(None, score=1.0, best_score=0.0, confirms_needed=2)
    assert replace is False and state is not None and state.confirms_seen == 1
    state, replace = update_confirmation(state, score=1.0, best_score=0.0, confirms_needed=2)
    assert replace is True


def test_post_gate_scores_excludes_pre_gate() -> None:
    from rlbot.checkpoint_selection import post_gate_scores

    steps = [1_000_000, 5_000_000, 30_000_000, 31_000_000]
    scores = [9.0, 8.0, 1.0, 2.0]
    post = post_gate_scores(steps, scores, gate_step=29_250_000)
    assert post == [1.0, 2.0]
    # Trailing over post-gate only — pre-gate 9.0 must not enter the window.
    assert trailing_aggregate(post, window=3, agg="median") == pytest.approx(1.5)
    assert post_gate_scores(steps, scores, gate_step=0) == scores


def test_blend_canonical_stress() -> None:
    assert blend_canonical_stress(1.0, None) == 1.0
    assert blend_canonical_stress(1.0, 0.0, stress_weight=0.3) == pytest.approx(0.7)


def test_two_head_actions_exposure_and_cap(monkeypatch) -> None:
    set_config(load_config())
    # High exposure logit, equal asset logits → nearly fully invested, capped.
    action = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    w = portfolio_weights_two_head(action, n_actions=6, max_single_asset_weight=0.20)
    assert w.shape == (6,)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert w[0] == pytest.approx(0.0, abs=1e-2)  # high exposure → little cash
    assert np.all(w[1:] <= 0.20 + 1e-9)

    # Low exposure logit → mostly cash.
    action2 = np.array([-5.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    w2 = portfolio_weights_two_head(action2, n_actions=6, max_single_asset_weight=0.20)
    assert w2[0] > 0.9
