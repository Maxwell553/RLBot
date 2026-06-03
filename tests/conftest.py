"""Pytest fixtures: repo root on path and config loaded for env helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def _load_rl_config() -> None:
    from rlbot.rl_config import load_config, set_config

    set_config(load_config(ROOT / "config" / "config.yaml"))
