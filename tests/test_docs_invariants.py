"""Tripwire: keep CLAUDE.md / AGENTS.md and the load-bearing code invariants in sync.

The docs were corrected on 2026-06-05 after badly drifting (118 vs 128 obs dim, 0.50 vs
0.35 cap, a nonexistent sync_trading_env_aliases, references to a missing windows/ dir).
This test fails if code and the agent docs disagree again. It is torch-free.
"""

from __future__ import annotations

import inspect

import rlbot.rl_config as rl_config
from rlbot.data_utils import train_test_split_alternating
from rlbot.rl_config import get_config, observation_dim_for_universe
from rlbot.run_artifacts import PROJECT_ROOT

AGENT_DOCS = ("CLAUDE.md", "AGENTS.md")


def _doc_text(name: str) -> str:
    return (PROJECT_ROOT / name).read_text(encoding="utf-8")


def test_obs_dim_is_128_for_default_universe() -> None:
    assert observation_dim_for_universe(10) == 128


def test_default_cap_is_025() -> None:
    assert get_config().environment.max_single_asset_weight == 0.25


def test_split_supports_feature_split_mode() -> None:
    params = inspect.signature(train_test_split_alternating).parameters
    assert "feature_split_mode" in params


def test_no_sync_trading_env_aliases_symbol() -> None:
    assert not hasattr(rl_config, "sync_trading_env_aliases")


def test_referenced_paths_match_layout() -> None:
    assert (PROJECT_ROOT / "scripts" / "train.py").is_file()
    assert (PROJECT_ROOT / "scripts" / "backtest.py").is_file()
    assert not (PROJECT_ROOT / "windows").exists()
    assert not (PROJECT_ROOT / "train.py").exists()  # no top-level entrypoints


def test_agent_docs_quote_code_derived_values() -> None:
    """Docs must mention the values the code actually computes (couples doc↔code)."""
    obs = str(observation_dim_for_universe(10))
    cap = str(get_config().environment.max_single_asset_weight)
    for name in AGENT_DOCS:
        text = _doc_text(name)
        assert obs in text, f"{name} must mention obs dim {obs}"
        assert cap in text, f"{name} must mention cap {cap}"
        assert "feature_split_mode" in text, f"{name} must document feature_split_mode"
        assert "Runs/" in text, f"{name} must use the Runs/ layout"
        assert "scripts/train.py" in text and "scripts/backtest.py" in text


def test_agent_docs_in_sync_with_each_other() -> None:
    """The two agent docs share an identical body (only the attribution line differs)."""
    claude = _doc_text("CLAUDE.md").splitlines()
    agents = _doc_text("AGENTS.md").splitlines()
    # Skip the first 3 lines (title + attribution) on each; bodies must match.
    assert claude[3:] == agents[3:]


# ── user docs (README / RESEARCH) — claims corrected twice get a tripwire ─


def test_readme_does_not_overclaim_purge() -> None:
    """README.md:111 once claimed 'join purge — no cross-block leakage' for the default
    mode, where the purge is not applied. The split-mode behavior must be described
    via feature_split_mode instead."""
    text = _doc_text("README.md")
    assert "join purge — no cross-block leakage" not in text
    assert "feature_split_mode" in text
    assert "feature_preroll_bars" in text


def test_research_md_states_run_local_config_binding() -> None:
    """RESEARCH.md once claimed backtest uses the *current global* config; it binds the
    run-local snapshot by default since the evolution-roadmap branch."""
    text = _doc_text("docs/RESEARCH.md")
    assert "uses the **current** global config" not in text
    assert "run-local" in text.lower()


def test_user_docs_mention_canonical_windows() -> None:
    """README/RESEARCH window tables must agree with the enforced canonical table."""
    from rlbot.research.spec import CANONICAL_WINDOWS

    readme = _doc_text("README.md")
    research = _doc_text("docs/RESEARCH.md")
    for name, w in CANONICAL_WINDOWS.items():
        assert w["train_end"] in readme, f"README window table missing {name} {w['train_end']}"
        assert w["train_end"] in research, f"RESEARCH.md registry missing {name} {w['train_end']}"


def test_published_metric_examples_use_best_checkpoint() -> None:
    """Examples in user docs must model --checkpoint best (latest/both touch OOS)."""
    for name in ("README.md", "docs/TRAINING.md"):
        text = _doc_text(name)
        assert "--checkpoint both" not in text.replace(
            "`--checkpoint latest|both`", ""
        ), f"{name} example still models --checkpoint both"


def test_agent_docs_do_not_claim_untracked_execution_readme() -> None:
    """CLAUDE/AGENTS once claimed execution/README.md is tracked (it is not) and that
    paper_trade/ is absent (it is tracked)."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True
    ).stdout.splitlines()
    assert "paper_trade/README.md" in tracked
    assert not any(p.startswith("execution/") for p in tracked)
    for name in AGENT_DOCS:
        text = _doc_text(name)
        assert "except a tracked `execution/README.md`" not in text
