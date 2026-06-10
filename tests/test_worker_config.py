"""Worker-side config propagation (C1): SubprocVecEnv workers spawn fresh
interpreters where the config singleton is unset, so without an explicit installer
every ``--config``/``--n-assets`` override silently falls back to the default
``config/config.yaml`` inside the training envs. Torch-free: uses plain
``multiprocessing`` spawn (the same start semantics SB3 uses) instead of SubprocVecEnv.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
from pathlib import Path

from rlbot.rl_config import (
    DEFAULT_CONFIG_PATH,
    WorkerConfigInstaller,
    _parse_config,
    get_config,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PATCHED_REWARD_SCALE = 1234.5


def _probe_with_installer(installer: WorkerConfigInstaller, queue) -> None:
    installer()
    from rlbot.rl_config import get_config as worker_get_config

    cfg = worker_get_config()
    queue.put((cfg.reward.reward_scale, cfg.environment.max_single_asset_weight))


def _probe_without_installer(queue) -> None:
    from rlbot.rl_config import get_config as worker_get_config

    cfg = worker_get_config()
    queue.put(cfg.reward.reward_scale)


def _patched_config():
    raw = copy.deepcopy(get_config().raw)
    raw["reward"]["reward_scale"] = _PATCHED_REWARD_SCALE
    return _parse_config(raw, DEFAULT_CONFIG_PATH)


def test_worker_config_installer_round_trips_in_spawned_process() -> None:
    patched = _patched_config()
    installer = WorkerConfigInstaller(patched)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_probe_with_installer, args=(installer, queue))
    proc.start()
    reward_scale, max_w = queue.get(timeout=60)
    proc.join(timeout=60)
    assert proc.exitcode == 0
    assert reward_scale == _PATCHED_REWARD_SCALE
    assert max_w == patched.environment.max_single_asset_weight
    # the parent's installed config is untouched by building the installer
    assert get_config().reward.reward_scale != _PATCHED_REWARD_SCALE


def test_spawned_process_without_installer_falls_back_to_default_config() -> None:
    """Documents the C1 failure mode the installer exists to prevent."""
    default_scale = load_config(DEFAULT_CONFIG_PATH).reward.reward_scale
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_probe_without_installer, args=(queue,))
    proc.start()
    reward_scale = queue.get(timeout=60)
    proc.join(timeout=60)
    assert proc.exitcode == 0
    assert reward_scale == default_scale


def test_train_factory_threads_config_installer() -> None:
    """Source guard: the env factory installs the config first thing inside the
    worker, and BOTH SubprocVecEnv construction sites pass the installer."""
    src = (PROJECT_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "config_installer: WorkerConfigInstaller | None" in src
    body = src.split("def _init():", 1)[1].split("return _init", 1)[0]
    install_pos = body.find("config_installer()")
    env_pos = body.find("MultiAssetPortfolioEnv(")
    assert install_pos != -1, "factory never calls the worker config installer"
    assert env_pos != -1
    assert install_pos < env_pos, "config must be installed before env construction"
    assert src.count("config_installer=worker_config") >= 2, (
        "both train and eval SubprocVecEnv factories must pass the installer"
    )
