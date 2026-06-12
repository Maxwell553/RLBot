"""
Per-training-run layout under ``Runs/<run_id>/``:

  Runs/<id>/manifest.json, config.yaml, data_cache.npz, eval_logs/
  Runs/<id>/models/, plots/, logs/, tb_logs/

Legacy roots (``runs/``, ``models/``, ``plots/``, ``logs/``, ``tb_logs/``) are still
resolved for **read** when a run has not been migrated yet.
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

RUNS_ROOT = PROJECT_ROOT / "Runs"
_LEGACY_RUNS_META_ROOT = PROJECT_ROOT / "runs"


def resolve_data_cache() -> Path:
    """Return the canonical panel cache path, migrating a legacy root ``data_cache.npz`` once."""
    if not DEFAULT_DATA_CACHE.is_file() and _LEGACY_DATA_CACHE.is_file():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(_LEGACY_DATA_CACHE), str(DEFAULT_DATA_CACHE))
    return DEFAULT_DATA_CACHE


def _run_exists(run_id: str, root: Path = PROJECT_ROOT) -> bool:
    """True if a run directory already exists (new or legacy layout)."""
    rid = run_id.strip()
    if not rid:
        return False
    return (root / "Runs" / rid).exists() or (root / "runs" / rid).exists()


def new_run_id(
    window: int,
    *,
    root: Path = PROJECT_ROOT,
    when: datetime | None = None,
) -> str:
    """
    Generate ``W{window}_{month}{day:02d}`` (e.g. ``W1_604`` for window 1 on June 4).

    If that id already exists, append ``_a``, ``_b``, … (e.g. ``W1_604_a``).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    now = when or datetime.now(timezone.utc)
    date_part = f"{now.month}{now.day:02d}"
    base = f"W{window}_{date_part}"
    candidate = base
    suffix = 0
    while _run_exists(candidate, root):
        suffix += 1
        letter = (
            "abcdefghijklmnopqrstuvwxyz"[suffix - 1]
            if suffix <= 26
            else str(suffix)
        )
        candidate = f"{base}_{letter}"
    return candidate


def _pick_existing(new_path: Path, legacy_path: Path) -> Path:
    """Prefer ``new_path`` when it exists; else legacy; else ``new_path`` for writes."""
    if new_path.exists():
        return new_path
    if legacy_path.exists():
        return legacy_path
    return new_path


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root: Path = PROJECT_ROOT

    @property
    def run_dir(self) -> Path:
        """Canonical run root (all new artifacts are written here)."""
        return self.root / "Runs" / self.run_id

    @property
    def run_meta_dir(self) -> Path:
        """Manifest, config snapshot, eval npz, optional data snapshot."""
        new = self.run_dir
        legacy = self.root / "runs" / self.run_id
        if (new / "manifest.json").is_file() or not (legacy / "manifest.json").is_file():
            return new
        return legacy

    @property
    def plots_dir(self) -> Path:
        return _pick_existing(self.run_dir / "plots", self.root / "plots" / self.run_id)

    @property
    def models_dir(self) -> Path:
        return _pick_existing(self.run_dir / "models", self.root / "models" / self.run_id)

    @property
    def logs_dir(self) -> Path:
        return _pick_existing(self.run_dir / "logs", self.root / "logs" / self.run_id)

    @property
    def tb_dir(self) -> Path:
        return _pick_existing(self.run_dir / "tb_logs", self.root / "tb_logs" / self.run_id)

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
    def eval_portfolio_diagnostics_jsonl(self) -> Path:
        return self.eval_log_dir / "eval_portfolio_diagnostics.jsonl"

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

    @property
    def config_snapshot(self) -> Path:
        return self.run_meta_dir / "config.yaml"

    def mkdirs(self) -> None:
        """Create the unified ``Runs/<run_id>/`` tree (always new layout)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("plots", "models", "logs", "tb_logs", "eval_logs"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        (self.run_dir / "models" / "best").mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")


def merge_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Write manifest fields, preserving keys from an existing file not in ``payload``."""
    merged = dict(payload)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, val in existing.items():
            if key not in merged:
                merged[key] = val
    write_manifest(path, merged)


def read_run_manifest(run_id: str) -> dict[str, Any] | None:
    """Load ``Runs/<run_id>/manifest.json`` (or legacy ``runs/<run_id>/``)."""
    path = RunPaths(run_id=run_id).manifest_path
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_run_ids_with_models() -> list[str]:
    """Run ids that have a ``models/`` tree (new or legacy layout)."""
    import re

    found: set[str] = set()
    pat = re.compile(r"^(.+)_seed_(\d+)$")
    if RUNS_ROOT.is_dir():
        for p in RUNS_ROOT.iterdir():
            if p.is_dir() and (p / "models").is_dir():
                found.add(p.name)
    legacy_models = PROJECT_ROOT / "models"
    if legacy_models.is_dir():
        for p in legacy_models.iterdir():
            if p.is_dir():
                found.add(p.name)
    return sorted(found, key=lambda x: (0 if pat.match(x) else 1, x))


def snapshot_data_cache(src: Path, dest: Path) -> None:
    if src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def resolve_run_data_cache(
    run_id: str,
    override: str | Path = "",
    *,
    root: Path = PROJECT_ROOT,
    default: Path | None = None,
) -> Path:
    """Panel cache for a run: ``override`` > run-local snapshot > ``default``/global.

    Lets a backtest stay reproducible from the run's own ``data_cache.npz`` even after
    the global cache is refreshed. Raises if an explicit ``override`` is missing.
    """
    ov = str(override).strip()
    if ov:
        p = Path(ov)
        if not p.is_file():
            raise FileNotFoundError(f"data cache override not found: {p}")
        return p
    snap = RunPaths(run_id=run_id, root=root).data_snapshot
    if snap.is_file():
        return snap
    return default if default is not None else resolve_data_cache()


def sha256_file(path: Path | str) -> str | None:
    """SHA-256 of a file's bytes (None if missing). Used for run provenance."""
    import hashlib

    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sha256(config_dict: Mapping[str, Any]) -> str:
    """Stable SHA-256 of a config dict (sorted keys) for run provenance."""
    import hashlib

    payload = json.dumps(dict(config_dict), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_provenance(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Best-effort ``{git_commit, git_dirty}`` (tolerates a non-git checkout)."""
    import subprocess

    def _run(cmd: list[str]) -> str | None:
        # Best-effort: git may be absent or this may not be a repo. Narrow to OS / subprocess
        # errors so a real bug here is not silently swallowed.
        try:
            out = subprocess.run(
                cmd, cwd=str(root), capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
    }


def check_holdout_window_against_manifest(
    got_start,
    got_end,
    chronological_holdout: dict | None,
    *,
    cli_override: bool = False,
) -> str | None:
    """Cross-check a resolved OOS window against the training manifest's recorded one.

    ``got_start``/``got_end`` are the first/last resolved holdout timestamps (anything
    ``pd.Timestamp`` accepts). Returns ``None`` when the windows match (or the manifest
    recorded no dates). On mismatch: raises ``ValueError`` unless ``cli_override``
    (explicit window flags = a deliberate cross-window evaluation), in which case the
    warning text is returned for the caller to print.

    Tail-based (``holdout_days``) runs recompute their window from the CURRENT cache's
    last bar — a refreshed/extended cache silently shifts the OOS window away from
    what training reserved; this check is what makes that loud.
    """
    import pandas as pd

    ch = chronological_holdout or {}
    rec_start_raw, rec_end_raw = ch.get("date_start"), ch.get("date_end")
    if not rec_start_raw or not rec_end_raw:
        return None
    rec_start = pd.Timestamp(rec_start_raw).date()
    rec_end = pd.Timestamp(rec_end_raw).date()
    got_start_d = pd.Timestamp(got_start).date()
    got_end_d = pd.Timestamp(got_end).date()
    if (got_start_d, got_end_d) == (rec_start, rec_end):
        return None
    msg = (
        f"OOS window mismatch: training recorded {rec_start}..{rec_end} but this "
        f"backtest resolved {got_start_d}..{got_end_d}. The evaluated holdout is NOT "
        "the one training reserved (refreshed cache or window overrides)."
    )
    if cli_override:
        return msg
    raise ValueError(
        msg + " Re-run with the run-local data snapshot, or pass explicit "
        "--train-end/--holdout-start/--holdout-end to acknowledge a "
        "deliberate cross-window evaluation."
    )
