"""
Per-training-run paths: plots/, models/, logs/, tb_logs/, runs/<id>/manifest + eval npz.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_DATA_CACHE = CACHE_DIR / "data_cache.npz"
_LEGACY_DATA_CACHE = PROJECT_ROOT / "data_cache.npz"


def resolve_data_cache() -> Path:
    """Return the canonical panel cache path, migrating a legacy root ``data_cache.npz`` once."""
    if not DEFAULT_DATA_CACHE.is_file() and _LEGACY_DATA_CACHE.is_file():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(_LEGACY_DATA_CACHE), str(DEFAULT_DATA_CACHE))
    return DEFAULT_DATA_CACHE
RUNS_ROOT = PROJECT_ROOT / "runs"
LATEST_RUN_FILE = RUNS_ROOT / "LATEST.txt"


def _human_steps(n: int) -> str:
    if n >= 1_000_000_000:
        v = n / 1_000_000_000
        return f"{v:.0f}B" if v == int(v) else f"{v:.1f}B"
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.0f}k" if v == int(v) else f"{v:.1f}k"
    return str(n)


def new_run_id(timesteps: int = 0) -> str:
    """
    Generate a run ID like '60M_4_14_26'.
    If a folder with that name already exists, appends _a, _b, _c, ...
    """
    now = datetime.now(timezone.utc)
    date_part = f"{now.month}_{now.day}_{now.strftime('%y')}"

    if timesteps > 0:
        base = f"{_human_steps(timesteps)}_{date_part}"
    else:
        base = f"{now.strftime('%H%M')}_{date_part}"

    candidate = base
    suffix = 0
    while (PROJECT_ROOT / "runs" / candidate).exists():
        suffix += 1
        candidate = f"{base}_{'abcdefghijklmnopqrstuvwxyz'[suffix - 1] if suffix <= 26 else suffix}"
    return candidate


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root: Path = PROJECT_ROOT

    @property
    def run_meta_dir(self) -> Path:
        """Manifest, optional data snapshot, eval npz directory root."""
        return self.root / "runs" / self.run_id

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots" / self.run_id

    @property
    def models_dir(self) -> Path:
        return self.root / "models" / self.run_id

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs" / self.run_id

    @property
    def tb_dir(self) -> Path:
        return self.root / "tb_logs" / self.run_id

    @property
    def eval_log_dir(self) -> Path:
        return self.run_meta_dir / "eval_logs"

    @property
    def training_plot(self) -> Path:
        return self.plots_dir / "training.png"

    @property
    def eval_npz(self) -> Path:
        return self.eval_log_dir / "evaluations.npz"

    @property
    def eval_nav_history(self) -> Path:
        return self.eval_log_dir / "eval_nav_history.npz"

    @property
    def final_model(self) -> Path:
        return self.models_dir / "ppo_portfolio_final.zip"

    @property
    def best_model_dir(self) -> Path:
        return self.models_dir / "best"

    @property
    def manifest_path(self) -> Path:
        return self.run_meta_dir / "manifest.json"

    @property
    def data_snapshot(self) -> Path:
        return self.run_meta_dir / "data_cache.npz"

    def mkdirs(self) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.run_meta_dir.mkdir(parents=True, exist_ok=True)
        self.eval_log_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")


def write_latest_pointer(run_id: str) -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_FILE.write_text(run_id.strip() + "\n", encoding="utf-8")


def read_latest_run_id() -> str | None:
    if not LATEST_RUN_FILE.is_file():
        return None
    text = LATEST_RUN_FILE.read_text(encoding="utf-8").strip()
    return text or None


def read_run_manifest(run_id: str) -> dict[str, Any] | None:
    """Load ``runs/<run_id>/manifest.json`` if present."""
    path = RunPaths(run_id=run_id).manifest_path
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_data_cache(src: Path, dest: Path) -> None:
    if src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
