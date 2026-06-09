"""Auto-research harness: spec allow-list firewall, variant expansion, registry,
report, and OOS gates (P2). Torch-free."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlbot.rl_config import get_config, load_config
from rlbot.research import gates, registry, report
from rlbot.research.spec import (
    CANONICAL_WINDOWS,
    ExperimentSpec,
    build_variant_config_dict,
    is_allowed_patch_key,
    load_spec,
    normalize_window,
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
    assert "| variant (across seeds) |" in text
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


# ── canonical windows (normalize_window) ─────────────────────────────────
def test_normalize_window_name_only_fills_canonical_dates() -> None:
    w = normalize_window({"name": "W4"})
    assert w == {
        "name": "W4",
        "train_end": "2021-12-31",
        "holdout_start": "2022-01-01",
        "holdout_end": "2023-12-31",
    }
    assert w == {"name": "W4", **CANONICAL_WINDOWS["W4"]}


def test_normalize_window_name_is_case_insensitive() -> None:
    assert normalize_window({"name": "w4"}) == normalize_window({"name": "W4"})


def test_normalize_window_exact_canonical_dates_resolve_to_name() -> None:
    w = normalize_window(dict(CANONICAL_WINDOWS["W2"]))
    assert w["name"] == "W2"
    assert w["train_end"] == CANONICAL_WINDOWS["W2"]["train_end"]


def test_normalize_window_non_canonical_dates_rejected() -> None:
    with pytest.raises(PermissionError, match="canonical"):
        normalize_window(
            {"train_end": "2021-06-30", "holdout_start": "2021-07-01",
             "holdout_end": "2022-12-31"}
        )


def test_normalize_window_unknown_key_rejected() -> None:
    # a typo'd date key must not be silently dropped (would change the holdout)
    with pytest.raises(ValueError, match="unknown key"):
        normalize_window({"name": "W4", "holdout_strt": "2022-01-01"})


def test_normalize_window_unknown_name_rejected() -> None:
    with pytest.raises(ValueError, match="not canonical"):
        normalize_window({"name": "W99"})


def test_normalize_window_name_date_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="dates match"):
        normalize_window({"name": "W1", **CANONICAL_WINDOWS["W3"]})


def test_spec_windows_normalized_in_post_init() -> None:
    spec = ExperimentSpec(id="x", windows=[{"name": "w4"}])
    assert spec.windows[0]["holdout_start"] == "2022-01-01"
    with pytest.raises(PermissionError):
        ExperimentSpec(
            id="x",
            windows=[{"train_end": "2020-06-30", "holdout_start": "2020-07-01"}],
        )


# ── base_config pinning ──────────────────────────────────────────────────
def test_spec_rejects_non_canonical_base_config() -> None:
    with pytest.raises(PermissionError, match="base_config"):
        ExperimentSpec(id="x", base_config="config/other.yaml")
    with pytest.raises(PermissionError):
        ExperimentSpec(id="x", base_config="/tmp/evil.yaml")
    assert ExperimentSpec(id="x").base_config == "config/config.yaml"  # default ok


# ── OOS budget gate ──────────────────────────────────────────────────────
def test_assert_oos_budget() -> None:
    gates.assert_oos_budget(1, 1)  # within budget ok
    gates.assert_oos_budget(0, 1)
    with pytest.raises(PermissionError, match="budget"):
        gates.assert_oos_budget(2, 1)
    with pytest.raises(PermissionError):
        gates.assert_oos_budget(13, 12)


# ── no-repeat OOS with attempt / failed records ──────────────────────────
def test_attempt_records_block_by_default() -> None:
    records = [
        {"variant_id": "v1", "evaluation_tier": 4, "run_id": "r1", "status": "oos_read_attempt"}
    ]
    with pytest.raises(PermissionError, match="unscored"):
        gates.assert_no_repeat_oos(records, "v1")
    # the retry escape hatch works only while no scored result exists
    gates.assert_no_repeat_oos(records, "v1", allow_failed_rescore=True)
    gates.assert_no_repeat_oos(records, "v2")  # other variants unaffected


def test_failed_record_alone_never_blocks() -> None:
    """A tier-4 variant whose TRAIN crashed (no attempt record → holdout never read)
    must not brick the relaunch; the attempt record is the only read marker."""
    records = [
        {"variant_id": "v1", "evaluation_tier": 4, "run_id": "r1", "status": "failed"}
    ]
    gates.assert_no_repeat_oos(records, "v1")
    # failed AFTER an attempt: the attempt still blocks
    records.append(
        {"variant_id": "v1", "evaluation_tier": 4, "run_id": "r1", "status": "oos_read_attempt"}
    )
    with pytest.raises(PermissionError, match="unscored"):
        gates.assert_no_repeat_oos(records, "v1")
    gates.assert_no_repeat_oos(records, "v1", allow_failed_rescore=True)


def test_scored_oos_record_blocks_even_with_rescore_flag() -> None:
    records = [
        {"variant_id": "v1", "evaluation_tier": 4, "run_id": "r1", "status": "oos_read_attempt"},
        {"variant_id": "v1", "evaluation_tier": 4, "run_id": "r1", "status": "ok"},
    ]
    with pytest.raises(PermissionError, match="already has"):
        gates.assert_no_repeat_oos(records, "v1")
    with pytest.raises(PermissionError, match="already has"):
        gates.assert_no_repeat_oos(records, "v1", allow_failed_rescore=True)


def test_low_tier_records_never_block_oos() -> None:
    records = [{"variant_id": "v1", "evaluation_tier": 3, "run_id": "r1", "status": "ok"}]
    gates.assert_no_repeat_oos(records, "v1")  # tier < 4 is not a holdout read


# ── report: OOS firewall + multiplicity header ───────────────────────────
def _variant_row(text: str, variant_id: str) -> str:
    rows = [l for l in text.splitlines() if l.startswith(f"| {variant_id} |")]
    assert len(rows) == 1, f"expected one row for {variant_id}"
    return rows[0]


def test_report_hides_oos_metrics_from_low_tier_records(tmp_path: Path) -> None:
    """A holdout number smuggled into a tier-3 record must not surface in the table."""
    records = [
        {"variant_id": "v3", "evaluation_tier": 3, "status": "ok",
         "oos_sharpe": 2.5, "oos_total_return": 0.5, "oos_max_drawdown": -0.05},
        {"variant_id": "v4", "evaluation_tier": 4, "status": "ok",
         "oos_sharpe": 1.25, "oos_total_return": 0.10, "oos_max_drawdown": -0.08},
    ]
    text = report.write_report(records, tmp_path / "r.md").read_text(encoding="utf-8")
    row3 = _variant_row(text, "v3")
    assert "2.50" not in row3 and "50.00%" not in row3  # tier-3 OOS values suppressed
    row4 = _variant_row(text, "v4")
    assert "1.25" in row4 and "10.00%" in row4  # tier-4 scored values surface


def test_report_ignores_unscored_oos_records(tmp_path: Path) -> None:
    records = [
        {"variant_id": "v1", "evaluation_tier": 4, "status": "oos_read_attempt",
         "oos_sharpe": 9.9},
        {"variant_id": "v1", "evaluation_tier": 4, "status": "failed", "oos_sharpe": 9.9},
    ]
    text = report.write_report(records, tmp_path / "r.md").read_text(encoding="utf-8")
    row = _variant_row(text, "v1")
    assert "9.90" not in row
    assert "| 0 |" in row  # zero scored seeds, variant still listed


def test_report_header_states_variant_multiplicity(tmp_path: Path) -> None:
    records = [
        {"variant_id": f"v{i}", "evaluation_tier": 4, "status": "ok", "oos_sharpe": 1.0}
        for i in range(3)
    ]
    text = report.write_report(records, tmp_path / "r.md").read_text(encoding="utf-8")
    assert "selected from 3 variant(s)" in text
    assert "multiplicity" in text
    assert "1 holdout read(s)" not in text  # 3 tier>=4 records were recorded
    assert "3 holdout read(s)" in text


def test_report_aggregates_across_seeds_via_group_id(tmp_path: Path) -> None:
    """H5 regression: seed-bearing variant ids must aggregate into ONE group row with
    a true cross-seed median, not one single-record row per seed."""
    from rlbot.research import report

    records = [
        {"variant_id": "ab__x=1__seed42__W1", "group_id": "ab__x=1__W1", "seed": 42,
         "run_id": "ab__x=1__seed42__W1", "evaluation_tier": 3, "best_eval_nav": 100.0},
        {"variant_id": "ab__x=1__seed101__W1", "group_id": "ab__x=1__W1", "seed": 101,
         "run_id": "ab__x=1__seed101__W1", "evaluation_tier": 3, "best_eval_nav": 120.0},
        {"variant_id": "ab__x=1__seed777__W1", "group_id": "ab__x=1__W1", "seed": 777,
         "run_id": "ab__x=1__seed777__W1", "evaluation_tier": 3, "best_eval_nav": 140.0},
        # the promoted (best) seed gets a SECOND record at tier 4 — it must not be
        # double-counted in the cross-seed medians
        {"variant_id": "ab__x=1__seed777__W1", "group_id": "ab__x=1__W1", "seed": 777,
         "run_id": "ab__x=1__seed777__W1", "evaluation_tier": 4, "best_eval_nav": 140.0,
         "oos_total_return": 0.1, "oos_sharpe": 0.5, "oos_max_drawdown": -0.2},
    ]
    out = report.write_report(records, tmp_path / "r.md", title="t")
    text = out.read_text(encoding="utf-8")
    # one row, 3 seeds, true cross-seed median NAV (promoted run deduped), tier max 4
    assert "| ab__x=1__W1 | 4 | 3 | 120.00 |" in text
    assert "seed42" not in text.split("|---|")[1]  # no per-seed rows


def test_report_group_key_strips_seed_for_legacy_records(tmp_path: Path) -> None:
    from rlbot.research.report import _group_key

    assert _group_key({"variant_id": "ab__x=1__seed42__W1"}) == "ab__x=1__W1"
    assert _group_key({"variant_id": "ab__seed7"}) == "ab"
    assert _group_key({"variant_id": "ab", "group_id": "g"}) == "g"


def test_resolve_variants_sets_group_id() -> None:
    from rlbot.research.spec import ExperimentSpec, resolve_variants

    spec = ExperimentSpec(
        id="ab",
        hypothesis="h",
        evaluation_tier=3,
        seeds=[1, 2],
        grid={"reward.reward_scale": [100, 200]},
    )
    variants = resolve_variants(spec)
    groups = {v.group_id for v in variants}
    assert len(variants) == 4 and len(groups) == 2
    for v in variants:
        assert f"seed{v.seed}" in v.variant_id
        assert "seed" not in v.group_id


def test_registry_read_skips_torn_lines(tmp_path: Path, capsys) -> None:
    from rlbot.research import registry

    reg = tmp_path / "registry.jsonl"
    registry.append_record(reg, {"run_id": "a", "status": "ok"})
    with reg.open("a", encoding="utf-8") as f:
        f.write('{"run_id": "b", "status"')  # torn tail from a crash
    records = registry.read_records(reg)
    assert [r["run_id"] for r in records] == ["a"]
    assert "corrupt line" in capsys.readouterr().err


def test_gate_reads_fail_closed_on_corrupt_registry(tmp_path: Path) -> None:
    """A corrupt line must BLOCK gate reads (a skipped scored record would silently
    permit a repeat holdout read); lenient reads remain available for reports."""
    from rlbot.research import registry

    reg = tmp_path / "registry.jsonl"
    registry.append_record(reg, {"run_id": "a", "variant_id": "a", "status": "ok",
                                 "evaluation_tier": 4})
    with reg.open("a", encoding="utf-8") as f:
        f.write('{"run_id": "b", "variant_id": "b", "status": "ok", "evaluation_tier": 4')
    with pytest.raises(ValueError, match="fails closed"):
        registry.read_records(reg, on_corrupt="raise")
    assert len(registry.read_records(reg)) == 1  # lenient path still works


def test_registry_lock_serializes_check_then_append(tmp_path: Path) -> None:
    import threading
    import time

    from rlbot.research import registry

    reg = tmp_path / "registry.jsonl"
    order: list[str] = []

    def holder() -> None:
        with registry.registry_lock(reg):
            order.append("h_in")
            registry.append_record(reg, {"run_id": "a"})
            time.sleep(0.25)
            order.append("h_out")

    def contender() -> None:
        time.sleep(0.05)  # let the holder acquire first
        with registry.registry_lock(reg):
            order.append("c_in")
            registry.append_record(reg, {"run_id": "b"})

    th, tc = threading.Thread(target=holder), threading.Thread(target=contender)
    th.start(); tc.start(); th.join(timeout=10); tc.join(timeout=10)
    # flock contention across separate open()s: the contender must enter only after
    # the holder released — a regression to a no-op lock would interleave c_in first
    assert order == ["h_in", "h_out", "c_in"], order
    assert [r["run_id"] for r in registry.read_records(reg)] == ["a", "b"]


def test_spec_rejects_seed_collision_hazards() -> None:
    from rlbot.research.spec import ExperimentSpec

    with pytest.raises(ValueError, match="__seed"):
        ExperimentSpec(id="abl__seed42_test", evaluation_tier=1)
    with pytest.raises(ValueError, match="non-negative"):
        ExperimentSpec(id="ok", seeds=[-1], evaluation_tier=1)


def test_resolve_variants_rejects_id_collisions_and_disambiguates_long_values() -> None:
    from rlbot.research.spec import ExperimentSpec, resolve_variants

    # long values that share a 16-char prefix must yield DISTINCT variant ids
    spec = ExperimentSpec(
        id="g",
        evaluation_tier=1,
        seeds=[1],
        grid={"policy.net_arch_pi": [[1024, 1024, 1024, 128], [1024, 1024, 1024, 256]]},
    )
    ids = [v.variant_id for v in resolve_variants(spec)]
    assert len(set(ids)) == 2, ids
    # grid keys SHARING a last segment must fall back to full dotted tags
    spec2 = ExperimentSpec(
        id="g2",
        evaluation_tier=1,
        seeds=[1],
        grid={"reward.participation_bonus": [1.0], "curriculum.participation_bonus": [2.0]},
    )
    v = resolve_variants(spec2)[0]
    assert "reward-participation_bonus=" in v.variant_id
    assert "curriculum-participation_bonus=" in v.variant_id


def test_materialize_refuses_edited_spec_over_trained_cohort(tmp_path: Path, monkeypatch) -> None:
    """Re-planning an edited spec over a cohort with registry records must hard-fail
    (it would relabel the plan of record and refresh promote's spec_sha256 guard)."""
    import scripts.research as research
    from rlbot.research import registry

    monkeypatch.setattr(research, "RUNS_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(research, "_cohort_dir", lambda c: tmp_path / c)
    monkeypatch.setattr(research, "_registry_path", lambda c: tmp_path / c / "registry.jsonl")
    spec_path = tmp_path / "s.yaml"
    spec_path.write_text(
        "id: edited_spec_guard\nhypothesis: h\nevaluation_tier: 1\nseeds: [0]\n"
        "patch:\n  reward.reward_scale: 1000.0\n",
        encoding="utf-8",
    )
    spec = research.load_spec(spec_path)
    research._materialize(spec)  # first plan: fine
    research._materialize(spec)  # idempotent re-plan, same sha: fine
    registry.append_record(
        tmp_path / "edited_spec_guard" / "registry.jsonl",
        {"run_id": "r", "status": "ok", "evaluation_tier": 1},
    )
    spec_path.write_text(
        "id: edited_spec_guard\nhypothesis: h CHANGED\nevaluation_tier: 1\nseeds: [0]\n"
        "patch:\n  reward.reward_scale: 500.0\n",
        encoding="utf-8",
    )
    edited = research.load_spec(spec_path)
    with pytest.raises(SystemExit, match="NEW spec id"):
        research._materialize(edited)
