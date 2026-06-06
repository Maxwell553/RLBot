"""Auto-research harness: spec allow-list firewall, variant expansion, registry,
report, and OOS gates (P2). Torch-free."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlbot.rl_config import get_config, load_config
from rlbot.research import gates, registry, report
from rlbot.research.spec import (
    ExperimentSpec,
    build_variant_config_dict,
    is_allowed_patch_key,
    load_spec,
    resolve_variants,
)
from rlbot.run_artifacts import PROJECT_ROOT


# ── allow-list firewall ──────────────────────────────────────────────────
def test_allowed_and_denied_patch_keys() -> None:
    assert is_allowed_patch_key("reward.churn_penalty")
    assert is_allowed_patch_key("data.feature_split_mode")
    assert is_allowed_patch_key("training.early_stop_patience")
    assert is_allowed_patch_key("environment.max_single_asset_weight")
    # denied: change what OOS / the universe / the split is
    assert not is_allowed_patch_key("training.holdout_days")
    assert not is_allowed_patch_key("training.block_size")
    assert not is_allowed_patch_key("universe.benchmark")
    assert not is_allowed_patch_key("transaction_costs.slippage")
    assert not is_allowed_patch_key("data.since")


def test_spec_rejects_disallowed_patch() -> None:
    with pytest.raises(PermissionError):
        ExperimentSpec(id="bad", grid={"training.holdout_days": [100, 200]})


# ── variant expansion ────────────────────────────────────────────────────
def test_resolve_variants_cartesian() -> None:
    spec = ExperimentSpec(
        id="x",
        grid={"reward.churn_penalty": [4.0, 8.5]},
        seeds=[1, 2, 3],
        windows=[{"name": "w1"}],
    )
    variants = resolve_variants(spec)
    assert len(variants) == 2 * 3 * 1
    assert len({v.variant_id for v in variants}) == 6
    assert all(v.concrete_patch["reward.churn_penalty"] in (4.0, 8.5) for v in variants)


def test_build_variant_config_applies_patch_and_validates(tmp_path: Path) -> None:
    base = get_config().to_dict()
    patched = build_variant_config_dict(base, {"data.feature_split_mode": "independent"})
    assert patched["data"]["feature_split_mode"] == "independent"
    # round-trips through the real parser
    import yaml

    p = tmp_path / "variant.yaml"
    p.write_text(yaml.safe_dump(patched), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.data.feature_split_mode == "independent"
    # base is untouched (deep copy)
    assert base["data"]["feature_split_mode"] == get_config().data.feature_split_mode


def test_shipped_spec_loads_and_expands() -> None:
    spec = load_spec(PROJECT_ROOT / "specs" / "feature_split_ab.yaml")
    variants = resolve_variants(spec)
    assert len(variants) == 2 * 3 * 1  # 2 modes × 3 seeds × 1 window
    assert spec.evaluation_tier == 3


def test_all_shipped_specs_valid() -> None:
    """Every specs/*.yaml loads, passes the patch firewall, and expands to ≥1 variant."""
    files = sorted((PROJECT_ROOT / "specs").glob("*.yaml"))
    assert files, "no shipped specs found"
    for f in files:
        spec = load_spec(f)  # __post_init__ enforces the allow-list
        assert len(resolve_variants(spec)) >= 1, f"{f.name} expanded to 0 variants"


# ── registry + report ────────────────────────────────────────────────────
def test_registry_roundtrip_and_record(tmp_path: Path) -> None:
    reg = tmp_path / "registry.jsonl"
    manifest = {
        "universe": {"n_assets": 10, "tickers": ["SP500"]},
        "chronological_holdout": {"train_end": "2020-12-31", "holdout_start": "2021-01-01"},
        "args": {"seed": 42},
        "git_commit": "abc",
        "config_hash": "h1",
        "feature_split_mode": "independent",
    }
    training = {"timesteps": 3_000_000, "best_eval_nav": 123456.0, "best_eval_step": 2_000_000}
    rec = registry.build_record(
        cohort="c", variant_id="c__v1", hypothesis="h", run_id="c__v1",
        evaluation_tier=3, manifest=manifest, training_summary=training,
    )
    registry.append_record(reg, rec)
    registry.append_record(reg, {**rec, "run_id": "c__v2", "variant_id": "c__v2"})
    out = registry.read_records(reg)
    assert len(out) == 2
    assert out[0]["best_eval_nav"] == 123456.0
    assert out[0]["feature_split_mode"] == "independent"


def test_report_renders_table(tmp_path: Path) -> None:
    records = [
        {"variant_id": "v1", "evaluation_tier": 4, "oos_total_return": 0.12,
         "oos_sharpe": 1.1, "oos_max_drawdown": -0.08, "feature_split_mode": "continuous"},
        {"variant_id": "v1", "evaluation_tier": 4, "oos_total_return": 0.10,
         "oos_sharpe": 0.9, "oos_max_drawdown": -0.10, "feature_split_mode": "continuous"},
    ]
    out = report.write_report(records, tmp_path / "report.md", title="t")
    text = out.read_text(encoding="utf-8")
    assert "| variant |" in text
    assert "v1" in text


# ── OOS firewall ─────────────────────────────────────────────────────────
def test_tier4_requires_promotion() -> None:
    with pytest.raises(PermissionError):
        gates.assert_tier_allowed(4, promoted=False)
    gates.assert_tier_allowed(4, promoted=True)  # ok
    gates.assert_tier_allowed(2, promoted=False)  # dev tier ok without promotion


def test_no_repeat_oos_read() -> None:
    existing = [{"variant_id": "v1", "evaluation_tier": 4, "run_id": "r1"}]
    with pytest.raises(PermissionError):
        gates.assert_no_repeat_oos(existing, "v1")
    gates.assert_no_repeat_oos(existing, "v2")  # different variant ok


def test_tier_oos_flags() -> None:
    assert not gates.tier_touches_oos(3)
    assert gates.tier_touches_oos(4)
