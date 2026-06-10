"""End-to-end OOS firewall in scripts/research.py (launch/promote) with subprocess
mocked out — no training, no torch. Proves the gate ordering and registry records."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlbot.research import registry
from rlbot.run_artifacts import RunPaths as _RealRunPaths

ROOT = Path(__file__).resolve().parents[1]
_MOD = None


def _research_mod():
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location(
            "research_cli_under_test", ROOT / "scripts" / "research.py"
        )
        _MOD = importlib.util.module_from_spec(spec)
        sys.modules["research_cli_under_test"] = _MOD
        spec.loader.exec_module(_MOD)
    return _MOD


def _write_spec(tmp: Path, *, spec_id: str, tier: int, seeds: list[int]) -> Path:
    text = (
        f"id: {spec_id}\n"
        "hypothesis: gate test\n"
        "base_config: config/config.yaml\n"
        "patch:\n"
        "  reward.churn_penalty: 4.0\n"
        f"seeds: [{', '.join(str(s) for s in seeds)}]\n"
        "windows:\n"
        "  - name: W4\n"
        "timesteps: 1000\n"
        f"evaluation_tier: {tier}\n"
    )
    p = tmp / f"{spec_id}.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def _manifest_path(tmp: Path, run_id: str) -> Path:
    return tmp / "Runs" / run_id / "manifest.json"


def _fabricate_manifest(tmp: Path, run_id: str) -> None:
    p = _manifest_path(tmp, run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "universe": {"n_assets": 10, "tickers": ["SP500"]},
                "chronological_holdout": {
                    "train_end": "2021-12-31",
                    "holdout_start": "2022-01-01",
                    "holdout_end": "2023-12-31",
                },
                "args": {"seed": 0},
                "git_commit": "deadbeef",
                "config_hash": "h",
                "feature_split_mode": "continuous",
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """research.py rooted at tmp_path with subprocess.run replaced by a recorder."""
    mod = _research_mod()
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config" / "config.yaml", tmp_path / "config" / "config.yaml")
    monkeypatch.setattr(mod, "REPO", tmp_path)
    # OOS actions refuse dirty trees; the harness pins a clean fake so tests are
    # independent of the developer's working-tree state.
    monkeypatch.setattr(mod, "git_provenance", lambda: {"git_commit": "test", "git_dirty": False})
    monkeypatch.setattr(mod, "RUNS", tmp_path / "Runs")
    monkeypatch.setattr(mod, "RunPaths", lambda rid: _RealRunPaths(run_id=rid, root=tmp_path))

    def _read_manifest(rid: str):
        p = _manifest_path(tmp_path, rid)
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    monkeypatch.setattr(mod, "read_run_manifest", _read_manifest)

    state = {
        "calls": [],  # (kind, run_id) per subprocess invocation, in order
        "registry_status_at_backtest": [],  # last registry status seen when backtest starts
        "backtest_raises": False,
    }

    def fake_run(cmd, check=True, cwd=None, **kwargs):
        cmd = [str(c) for c in cmd]
        script = cmd[1]
        if script.endswith("train.py"):
            run_id = cmd[cmd.index("--run-id") + 1]
            state["calls"].append(("train", run_id))
            _fabricate_manifest(tmp_path, run_id)
        elif script.endswith("backtest.py"):
            run_id = cmd[cmd.index("--run-id") + 1]
            state["calls"].append(("backtest", run_id))
            # observe the registry at the moment the OOS read begins
            cohort_regs = sorted((tmp_path / "Runs").glob("*/registry.jsonl"))
            records = registry.read_records(cohort_regs[0]) if cohort_regs else []
            state["registry_status_at_backtest"].append(
                records[-1]["status"] if records else None
            )
            if state["backtest_raises"]:
                raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        else:  # pragma: no cover - unexpected subprocess use
            raise AssertionError(f"unexpected subprocess call: {cmd}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    return mod, tmp_path, state


def _launch_args(spec_path: Path, *, promote: bool = False, oos_budget: int = 1):
    return SimpleNamespace(
        spec=str(spec_path), backend="local", promote=promote,
        dry_run=False, oos_budget=oos_budget,
    )


def _promote_args(spec_path: Path, variant: str, *, promote: bool = True,
                  allow_failed_rescore: bool = False):
    return SimpleNamespace(
        spec=str(spec_path), variant=variant, promote=promote,
        dry_run=False, allow_failed_rescore=allow_failed_rescore,
    )


# ── (a) tier-4 launch needs --promote ────────────────────────────────────
def test_tier4_launch_without_promote_is_refused(harness) -> None:
    mod, tmp, state = harness
    spec_path = _write_spec(tmp, spec_id="exp_a", tier=4, seeds=[0])
    with pytest.raises(PermissionError, match="requires"):
        mod.cmd_launch(_launch_args(spec_path, promote=False))
    assert state["calls"] == []  # gate fires before any subprocess


# ── (b) OOS budget enforced before any run ───────────────────────────────
def test_tier4_launch_over_oos_budget_is_refused_before_subprocess(harness) -> None:
    mod, tmp, state = harness
    spec_path = _write_spec(tmp, spec_id="exp_b", tier=4, seeds=[0, 1])  # 2 variants
    with pytest.raises(PermissionError, match="budget"):
        mod.cmd_launch(_launch_args(spec_path, promote=True, oos_budget=1))
    assert state["calls"] == []
    # within budget the same launch proceeds (train → attempt → backtest → scored)
    mod.cmd_launch(_launch_args(spec_path, promote=True, oos_budget=2))
    kinds = [k for k, _ in state["calls"]]
    assert kinds == ["train", "backtest", "train", "backtest"]
    records = registry.read_records(tmp / "Runs" / "exp_b" / "registry.jsonl")
    assert [r["status"] for r in records] == ["oos_read_attempt", "ok"] * 2
    # the attempt record was on disk before each backtest started
    assert state["registry_status_at_backtest"] == ["oos_read_attempt"] * 2


# ── (c) promote: attempt-before-read, no-repeat, failed-rescore ──────────
def test_promote_records_attempt_then_score_and_blocks_repeat(harness) -> None:
    mod, tmp, state = harness
    spec_path = _write_spec(tmp, spec_id="exp_c", tier=3, seeds=[0])
    variant = "exp_c__seed0__W4"
    _fabricate_manifest(tmp, variant)  # already trained at tier 3
    mod._materialize(mod.load_spec(spec_path))  # promote loads cohort.json, never rewrites it

    mod.cmd_promote(_promote_args(spec_path, variant))
    assert [k for k, _ in state["calls"]] == ["backtest"]
    # the oos_read_attempt record was appended BEFORE the backtest subprocess ran
    assert state["registry_status_at_backtest"] == ["oos_read_attempt"]
    records = registry.read_records(tmp / "Runs" / "exp_c" / "registry.jsonl")
    assert [r["status"] for r in records] == ["oos_read_attempt", "ok"]
    # group_id threading: materialize → cohort entry → _collect_one → record
    assert all(r.get("group_id") == "exp_c__W4" for r in records), records
    assert all(int(r["evaluation_tier"]) >= 4 for r in records)

    # a second promote of the same variant is refused (multiple-testing guard)
    with pytest.raises(PermissionError, match="already has"):
        mod.cmd_promote(_promote_args(spec_path, variant))
    # ... even when claiming a failed rescore (a scored result exists)
    with pytest.raises(PermissionError):
        mod.cmd_promote(_promote_args(spec_path, variant, allow_failed_rescore=True))
    assert [k for k, _ in state["calls"]] == ["backtest"]  # no extra holdout read


def test_promote_crash_fails_closed_and_rescore_needs_flag(harness) -> None:
    mod, tmp, state = harness
    spec_path = _write_spec(tmp, spec_id="exp_d", tier=3, seeds=[0])
    variant = "exp_d__seed0__W4"
    _fabricate_manifest(tmp, variant)
    mod._materialize(mod.load_spec(spec_path))
    reg = tmp / "Runs" / "exp_d" / "registry.jsonl"

    state["backtest_raises"] = True
    with pytest.raises(SystemExit, match="backtest failed"):
        mod.cmd_promote(_promote_args(spec_path, variant))
    records = registry.read_records(reg)
    assert [r["status"] for r in records] == ["oos_read_attempt", "failed"]

    # retry without --allow-failed-rescore is refused (holdout may have been read)
    state["backtest_raises"] = False
    with pytest.raises(PermissionError, match="allow-failed-rescore"):
        mod.cmd_promote(_promote_args(spec_path, variant))
    assert [r["status"] for r in registry.read_records(reg)] == [
        "oos_read_attempt", "failed",
    ]

    # with the flag the retry proceeds and produces a scored record
    mod.cmd_promote(_promote_args(spec_path, variant, allow_failed_rescore=True))
    assert [r["status"] for r in registry.read_records(reg)] == [
        "oos_read_attempt", "failed", "oos_read_attempt", "ok",
    ]
    # once scored, even the flag cannot re-read the holdout
    with pytest.raises(PermissionError, match="already has"):
        mod.cmd_promote(_promote_args(spec_path, variant, allow_failed_rescore=True))


# ── (d) launch resume skips already-collected variants ───────────────────
def test_launch_resume_skips_scored_variant(harness) -> None:
    mod, tmp, state = harness
    spec_path = _write_spec(tmp, spec_id="exp_e", tier=3, seeds=[0])
    variant = "exp_e__seed0__W4"
    reg = tmp / "Runs" / "exp_e" / "registry.jsonl"
    registry.append_record(
        reg,
        {"variant_id": variant, "run_id": variant, "evaluation_tier": 3, "status": "ok"},
    )
    mod.cmd_launch(_launch_args(spec_path))
    assert state["calls"] == []  # resumed: nothing re-run
    assert len(registry.read_records(reg)) == 1  # no duplicate record


def test_launch_runs_unscored_variant(harness) -> None:
    """Inverse of the resume test: without an ok record the variant trains and collects."""
    mod, tmp, state = harness
    spec_path = _write_spec(tmp, spec_id="exp_f", tier=3, seeds=[0])
    variant = "exp_f__seed0__W4"
    reg = tmp / "Runs" / "exp_f" / "registry.jsonl"
    # a failed record does NOT count as collected at this tier
    registry.append_record(
        reg,
        {"variant_id": variant, "run_id": variant, "evaluation_tier": 3, "status": "failed"},
    )
    mod.cmd_launch(_launch_args(spec_path))
    assert state["calls"] == [("train", variant)]  # tier 3: no backtest, no OOS
    records = registry.read_records(reg)
    assert [r["status"] for r in records] == ["failed", "ok"]
    assert records[-1]["git_commit"] == "deadbeef"  # collected from fabricated manifest


# ── Phase C: backend command construction, queue guardrail, screen ranking ──
def test_train_cmd_modal_backend_constructs_modal_invocation() -> None:
    import scripts.research as research

    entry = {
        # _materialize emits ABSOLUTE paths; the remote container needs repo-relative
        "config_path": str(research.PROJECT_ROOT / "Runs" / "c" / "configs" / "v.yaml"),
        "run_id": "v", "seed": 7,
        "window": {"train_end": "2021-12-31", "holdout_start": "2022-01-01",
                   "holdout_end": "2023-12-31"},
    }
    spec = research.load_spec(research.PROJECT_ROOT / "specs" / "feature_split_ab.yaml")
    cmd = research._train_cmd(entry, spec, backend="modal", modal_gpu="H100")
    # modal_app.py has several local entrypoints — the ::train one must be named,
    # else `modal run` refuses to dispatch
    runnable = next(a for a in cmd if a.endswith("modal_app.py::train"))
    assert runnable
    assert "run" in cmd and "--" in cmd
    assert "--modal-gpu" in cmd and "H100" in cmd
    # config path must be repo-relative so it resolves inside the container
    cfg_i = cmd.index("--config") + 1
    assert cmd[cfg_i] == "Runs/c/configs/v.yaml", cmd[cfg_i]
    # --modal-gpu and train flags all come AFTER the `--` separator
    sep = cmd.index("--")
    assert cmd.index("--modal-gpu") > sep and cmd.index("--config") > sep
    # timesteps override beats spec timesteps
    cmd2 = research._train_cmd(entry, spec, timesteps_override=123)
    assert "123" in cmd2


def test_train_cmd_modal_never_hard_requires_path_binary(monkeypatch) -> None:
    """modal_cli() degrades which() -> venv sibling -> python -m modal; _train_cmd
    must never SystemExit just because `modal` is not on PATH."""
    import rlbot.modal_cloud as mc
    import scripts.research as research

    monkeypatch.setattr(mc.shutil, "which", lambda b: None)
    entry = {"config_path": "c.yaml", "run_id": "v", "seed": 1, "window": {}}
    spec = research.load_spec(research.PROJECT_ROOT / "specs" / "feature_split_ab.yaml")
    cmd = research._train_cmd(entry, spec, backend="modal")
    head = cmd[: cmd.index("run")]
    assert head[-1].endswith("modal") or head[-2:] == ["-m", "modal"], head


def test_run_queue_refuses_oos_tiers(harness, tmp_path) -> None:
    mod, tmp, state = harness
    qdir = tmp / "queue"
    qdir.mkdir()
    spec_path = _write_spec(tmp, spec_id="exp_q4", tier=4, seeds=[0])
    shutil.copy2(spec_path, qdir / "exp_q4.yaml")
    mod.cmd_run_queue(argparse.Namespace(queue_dir=str(qdir), backend="local",
                                         modal_gpu=None, window_budget=None))
    assert not (qdir / "exp_q4.yaml").exists()
    assert (qdir / "failed" / "exp_q4.yaml").exists()
    assert state["calls"] == []  # nothing trained, nothing backtested


def test_run_queue_survives_malformed_yaml(harness) -> None:
    mod, tmp, state = harness
    qdir = tmp / "queue"
    qdir.mkdir()
    (qdir / "bad.yaml").write_text("id: [unclosed\n  ::: not yaml", encoding="utf-8")
    mod.cmd_run_queue(argparse.Namespace(queue_dir=str(qdir), backend="local",
                                         modal_gpu=None, window_budget=None))
    assert not (qdir / "bad.yaml").exists()
    assert (qdir / "failed" / "bad.yaml").exists()


def test_screen_ranking_orders_and_advances_top_fraction() -> None:
    import scripts.research as research

    records = [
        {"group_id": "a", "best_eval_nav": 100.0},
        {"group_id": "a", "best_eval_nav": 110.0},
        {"group_id": "b", "best_eval_nav": 300.0},
        {"group_id": "c", "best_eval_nav": 200.0},
        {"group_id": "d", "best_eval_nav": None},  # never trained far enough
    ]
    ranked, advance = research.screen_ranking(records, keep_top=0.25)
    assert [g for g, _ in ranked][:3] == ["b", "c", "a"]
    assert advance == ["b"]  # round(4 * 0.25) = 1
    assert ranked[-1][1] == float("-inf")  # nav-less group sinks to the bottom
    # at least one group always advances
    _, adv2 = research.screen_ranking(records[:2], keep_top=0.01)
    assert adv2 == ["a"]
    assert research.screen_ranking([], keep_top=0.5) == ([], [])


def test_screen_uses_isolated_run_ids() -> None:
    import scripts.research as research

    src = (research.PROJECT_ROOT / "scripts" / "research.py").read_text(encoding="utf-8")
    screen_body = src.split("def cmd_screen", 1)[1].split("\ndef ", 1)[0]
    assert '"__screen"' in screen_body, (
        "screen must train under <variant>__screen run ids — reusing the variant id "
        "would overwrite full-budget artifacts and let collect stamp screen runs at "
        "the full tier"
    )


def test_run_queue_refuses_tier5_specs(harness) -> None:
    """Tier 5 no longer 'touches OOS' taxonomically, but starting shadow evaluation
    is still a human promotion decision — the queue must refuse it too."""
    mod, tmp, state = harness
    qdir = tmp / "queue"
    qdir.mkdir()
    spec_path = _write_spec(tmp, spec_id="exp_q5", tier=5, seeds=[0])
    shutil.copy2(spec_path, qdir / "exp_q5.yaml")
    mod.cmd_run_queue(argparse.Namespace(queue_dir=str(qdir), backend="local",
                                         modal_gpu=None, window_budget=None))
    assert (qdir / "failed" / "exp_q5.yaml").exists()
    assert state["calls"] == []


def test_promote_always_records_at_tier_4(harness) -> None:
    """Registry tier>=4 rows mean exactly 'holdout reads': a tier-5 spec's promote
    must record at tier 4, never 5 (shadow evidence lives in execution/, not here)."""
    mod, tmp, state = harness
    spec_path = _write_spec(tmp, spec_id="exp_p5", tier=5, seeds=[0])
    variant = "exp_p5__seed0__W4"
    _fabricate_manifest(tmp, variant)
    mod._materialize(mod.load_spec(spec_path))
    mod.cmd_promote(_promote_args(spec_path, variant))
    records = registry.read_records(tmp / "Runs" / "exp_p5" / "registry.jsonl")
    assert [int(r["evaluation_tier"]) for r in records] == [4, 4]
