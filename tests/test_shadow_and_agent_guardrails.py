"""Phase E: agent spec interface (validate --agent), git-dirty OOS guard, and the
tier-5 shadow-trading skeleton (drift alarm, reconcile math). Torch-free."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── validate command ─────────────────────────────────────────────────────
def _validate(spec_path, agent: bool = False):
    import scripts.research as research

    return research.cmd_validate(argparse.Namespace(spec=str(spec_path), agent=agent))


def test_validate_passes_shipped_specs() -> None:
    for spec in sorted((PROJECT_ROOT / "specs").glob("*.yaml")):
        _validate(spec)


def test_validate_agent_mode_requires_provenance_fields(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(
        "id: agent_spec\nevaluation_tier: 3\nseeds: [0]\n"
        "patch:\n  reward.reward_scale: 1500.0\n",
        encoding="utf-8",
    )
    _validate(p)  # plain validation: fine
    with pytest.raises(SystemExit, match="problem"):
        _validate(p, agent=True)  # no hypothesis/parent/success_gates
    p.write_text(
        "id: agent_spec\nhypothesis: scale matters\nparent: base\n"
        "evaluation_tier: 3\nseeds: [0]\n"
        "patch:\n  reward.reward_scale: 1500.0\n"
        "success_gates:\n  eval_nav_mean_min: 100000\n",
        encoding="utf-8",
    )
    _validate(p, agent=True)


def test_validate_agent_mode_refuses_oos_tiers(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(
        "id: agent_oos\nhypothesis: h\nparent: base\nevaluation_tier: 4\nseeds: [0]\n"
        "windows:\n  - name: W4\n"
        "success_gates:\n  eval_nav_mean_min: 100000\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="problem"):
        _validate(p, agent=True)


def test_validate_has_no_side_effects(tmp_path: Path, monkeypatch) -> None:
    import scripts.research as research

    monkeypatch.setattr(research, "_cohort_dir", lambda c: tmp_path / c)
    _validate(PROJECT_ROOT / "specs" / "feature_split_ab.yaml")
    assert not (tmp_path / "feature_split_ab").exists()


# ── git-dirty OOS guard ──────────────────────────────────────────────────
def test_dirty_tree_blocks_oos_actions(monkeypatch) -> None:
    import scripts.research as research

    monkeypatch.setattr(
        research, "git_provenance", lambda: {"git_commit": "x", "git_dirty": True}
    )
    with pytest.raises(SystemExit, match="dirty"):
        research._assert_clean_tree_for_oos(allow_dirty=False)
    research._assert_clean_tree_for_oos(allow_dirty=True)  # explicit override
    monkeypatch.setattr(
        research, "git_provenance", lambda: {"git_commit": "x", "git_dirty": False}
    )
    research._assert_clean_tree_for_oos(allow_dirty=False)


# ── shadow loop ──────────────────────────────────────────────────────────
def test_obs_drift_alarm_thresholds() -> None:
    from scripts.shadow_trade import obs_drift_alarm

    calm = np.random.default_rng(0).normal(0, 1, 128)
    assert not obs_drift_alarm(calm)
    shocked = calm.copy()
    shocked[:10] = 9.0  # ~8% of features beyond 5 sigma
    assert obs_drift_alarm(shocked)


def test_realized_portfolio_return_open_to_open() -> None:
    from scripts.shadow_trade import realized_portfolio_return

    t, n = 0, 2
    ohlcv = np.zeros((4, n, 5))
    ohlcv[:, :, 0] = [[100.0, 50.0], [110.0, 50.0], [121.0, 45.0], [121.0, 45.0]]
    w = {"A": 0.5, "B": 0.25}  # rest cash (earns 0)
    got = realized_portfolio_return(w, ["A", "B"], ohlcv, t)
    # A: 110→121 = +10%; B: 50→45 = −10% → 0.5*0.10 + 0.25*(−0.10) = +0.025
    assert got == pytest.approx(0.025)


def test_linear_costs_turnover_plus_holding() -> None:
    from scripts.shadow_trade import linear_costs

    tickers = ["A", "B"]
    slip = np.array([0.001, 0.002])
    fee = np.array([0.0005, 0.0005])
    hold = np.array([0.0001, 0.0002])
    # first day: full entry turnover
    c0 = linear_costs({"A": 0.6, "B": 0.4}, None, tickers, slip, fee, hold)
    assert c0 == pytest.approx(0.6 * 0.0015 + 0.4 * 0.0025 + 0.6 * 0.0001 + 0.4 * 0.0002)
    # unchanged book: holding cost only
    c1 = linear_costs({"A": 0.6, "B": 0.4}, {"A": 0.6, "B": 0.4}, tickers, slip, fee, hold)
    assert c1 == pytest.approx(0.6 * 0.0001 + 0.4 * 0.0002)


def test_shadow_reconcile_fills_realized_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    import pandas as pd

    import scripts.shadow_trade as shadow
    from rlbot.data_utils import save_cache
    from rlbot.rl_config import get_config

    cfg = get_config()
    n = cfg.universe.n_assets
    tickers = list(cfg.universe.tickers)
    bars = 8
    idx = pd.bdate_range("2024-01-02", periods=bars)
    ohlcv = np.full((bars, n, 5), 100.0)
    ohlcv[:, :, 4] = 1e6
    ohlcv[2, 0, 0] = 100.0   # fill bar open for as_of = idx[1]
    ohlcv[3, 0, 0] = 105.0   # +5% on asset 0
    zeros = np.zeros((bars, n))
    macro = np.full((bars, 4), 10.0)
    cache = tmp_path / "cache.npz"
    save_cache(
        str(cache), idx, ohlcv, zeros, zeros, macro, zeros, np.zeros((bars, 4)),
        zeros, zeros, np.zeros((bars, 4)), asset_live=np.ones((bars, n)),
        tickers=tickers,
    )

    monkeypatch.setattr(shadow, "EXECUTION_DIR", tmp_path / "execution")
    monkeypatch.setattr(shadow, "read_run_manifest", lambda rid: {"universe": {"tickers": tickers}})
    monkeypatch.setattr(shadow, "_bind_run_config", lambda rid, cur: None)

    decision_bar = str(idx[1].date())
    shadow._append_jsonl(
        shadow.ledger_path("r1"),
        {"run_id": "r1", "as_of": str(idx[3].date()), "decision_bar": decision_bar,
         "checkpoint": "best",
         "target_weights": {"CASH": 0.5, tickers[0]: 0.5}, "obs_drift": None},
    )
    args = argparse.Namespace(run_id="r1", data_cache=str(cache), use_current_config=False)
    shadow.cmd_reconcile(args)
    rows = shadow._read_jsonl(shadow.reconciled_path("r1"))
    assert len(rows) == 1
    realized = rows[0]["realized"]
    # gross: half the book in asset 0, which moved +5% open[t+1]→open[t+2]
    assert realized["model_return_gross"] == pytest.approx(0.5 * 0.05)
    assert realized["fill_bar"] == str(idx[2].date())
    # costs: first record → full entry turnover on 0.5 of asset 0, + daily holding
    cfg2 = get_config()
    slip = cfg2.transaction_costs.slippage_array()[0]
    fee = cfg2.transaction_costs.tx_fee_array()[0]
    hold = cfg2.transaction_costs.daily_holding_cost_array()
    exp_cost = 0.5 * (slip + fee) + 0.5 * hold[0]
    assert realized["model_linear_costs"] == pytest.approx(exp_cost)
    assert realized["model_return_net"] == pytest.approx(0.5 * 0.05 - exp_cost)
    # benchmark (buy & hold): asset 0 (cap weight .55) moved +5%, pays holding only
    bench_w = cfg2.reward.benchmark_cap_weights_array()
    exp_bench = 0.55 * 0.05 - float(np.dot(bench_w, hold))
    assert realized["benchmark_return_net"] == pytest.approx(exp_bench, rel=1e-6)
    # idempotent: second reconcile adds nothing
    shadow.cmd_reconcile(args)
    assert len(shadow._read_jsonl(shadow.reconciled_path("r1"))) == 1


def test_shadow_ledger_lives_under_gitignored_execution_dir() -> None:
    import scripts.shadow_trade as shadow

    assert shadow.EXECUTION_DIR.name == "execution"
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "execution/" in gitignore
