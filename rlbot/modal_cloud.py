"""
Modal cloud training: volume commits, artifact sync, and launch-broker helpers.

Training hooks (:func:`commit_modal_volumes`, :func:`mark_plot_saved`) avoid importing
``modal`` at module load so local ``train.py`` runs do not require the Modal SDK.

Sync CLI: ``python scripts/modal_app.py sync --run-id W2_605 --watch``
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rlbot.run_artifacts import PROJECT_ROOT, RunPaths

APP_NAME = "rlbot-train"
VOLUME_RUNS = "rlbot-runs"
VOLUME_CACHE = "rlbot-cache"
WORKSPACE = "/workspace"
DEFAULT_GPU = "A10G"
MODAL_MAX_TIMEOUT_SEC = 86_400
DEFAULT_TIMEOUT_SEC = MODAL_MAX_TIMEOUT_SEC
def resolve_n_steps_for_argv(train_argv: list[str] | None = None) -> int:
    """hyperparameters.n_steps for a launch: from the forwarded ``--config`` when
    present (research variants may patch n_steps), else the default config. Called
    at launch time, never frozen at import. Falls back to 4096 only when no config
    is parseable (skeleton containers)."""
    cfg_path = None
    argv = list(train_argv or [])
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 < len(argv):
            cfg_path = argv[i + 1]
    else:
        cfg_path = next(
            (a.split("=", 1)[1] for a in argv if a.startswith("--config=")), None
        )
    try:
        from rlbot.rl_config import get_config, load_config

        cfg = load_config(cfg_path) if cfg_path else get_config()
        return int(cfg.hyperparameters.n_steps)
    except Exception:
        return 4096


# Default-config value, kept for callers without an argv context (prints, sizing
# fallbacks). Prefer resolve_n_steps_for_argv at launch time.
N_STEPS = resolve_n_steps_for_argv()
TARGET_MINIBATCHES_PER_EPOCH = 4
MODAL_GPU_FLAG = "--modal-gpu"

VOLUME_RUNS_ENV = "RLBOT_MODAL_VOLUME_RUNS"
VOLUME_CACHE_ENV = "RLBOT_MODAL_VOLUME_CACHE"
MODAL_FLAG_ENV = "RLBOT_MODAL"

DEFAULT_RUNS_VOLUME = VOLUME_RUNS
DEFAULT_CACHE_VOLUME = VOLUME_CACHE

GPU_PROFILES: dict[str, dict[str, int]] = {
    "T4": {"cpus": 4, "n_envs": 8},
    "A10G": {"cpus": 16, "n_envs": 16},
    "L4": {"cpus": 16, "n_envs": 16},
    "A100": {"cpus": 32, "n_envs": 32},
    "H100": {"cpus": 64, "n_envs": 64},
}
DEFAULT_PROFILE = {"cpus": 16, "n_envs": 16}

PLOT_FILES = ("training.png", "training_episodes.npz")
EVAL_FILES = ("eval_nav_history.npz", "evaluations.npz")
VIZ_FREQ_DEFAULT = 500_000


# ── Remote volume commits (called from train.py / visualize.py on Modal) ─────


def is_modal_remote() -> bool:
    return os.environ.get(MODAL_FLAG_ENV) == "1"


def commit_modal_volumes(*, reason: str = "") -> None:
    """Flush Modal volume writes so local sync and web endpoints see new artifacts."""
    if not is_modal_remote():
        return
    names: list[str] = []
    for key in (VOLUME_RUNS_ENV, VOLUME_CACHE_ENV):
        name = os.environ.get(key, "").strip()
        if name:
            names.append(name)
    if not names:
        names = [DEFAULT_RUNS_VOLUME, DEFAULT_CACHE_VOLUME]
    try:
        import modal

        for name in names:
            modal.Volume.from_name(name).commit()
        if reason:
            print(f"[modal] committed volumes ({reason})", flush=True)
    except Exception as exc:
        print(f"[modal] volume commit warning: {exc}", flush=True)


def mark_plot_saved(path: Path, *, reason: str = "plot") -> None:
    if not is_modal_remote():
        return
    commit_modal_volumes(reason=f"{reason}: {path.name}")


# ── Launch broker helpers ────────────────────────────────────────────────────


def normalize_gpu_name(gpu: str) -> str:
    key = gpu.strip().upper()
    aliases = {"A10": "A10G", "A100-80GB": "A100", "H100-80GB": "H100"}
    return aliases.get(key, key)


def gpu_profile(gpu: str) -> dict[str, int]:
    return dict(GPU_PROFILES.get(normalize_gpu_name(gpu), DEFAULT_PROFILE))


def rollout_batch_size(n_envs: int, *, n_steps: int = N_STEPS) -> int:
    rollout = n_steps * n_envs
    batch = max(rollout // TARGET_MINIBATCHES_PER_EPOCH, 1)
    while batch > 1 and rollout % batch != 0:
        batch //= 2
    return batch


def parse_entrypoint_argv(
    arglist: list[str],
    *,
    default_gpu: str = DEFAULT_GPU,
) -> tuple[str, list[str]]:
    args = list(arglist)
    gpu = default_gpu
    i = 0
    while i < len(args):
        if args[i] == MODAL_GPU_FLAG:
            if i + 1 >= len(args):
                raise SystemExit(f"{MODAL_GPU_FLAG} requires a GPU name (e.g. A10G, A100)")
            gpu = args[i + 1]
            del args[i : i + 2]
            continue
        i += 1
    return gpu, args


def extract_run_id(argv: list[str]) -> str | None:
    args = list(argv)
    for i, tok in enumerate(args):
        if tok == "--run-id" and i + 1 < len(args):
            return args[i + 1].strip() or None
        if tok.startswith("--run-id="):
            return tok.split("=", 1)[1].strip() or None
    for i, tok in enumerate(args):
        if tok == "--window" and i + 1 < len(args):
            try:
                from rlbot.run_artifacts import new_run_id

                return new_run_id(int(args[i + 1]))
            except (ValueError, ImportError):
                return None
    return None


# ── Local artifact sync (Modal CLI subprocess) ─────────────────────────────


def modal_cli() -> list[str]:
    venv_modal = Path(sys.executable).resolve().parent / "modal"
    if venv_modal.is_file():
        return [str(venv_modal)]
    found = shutil.which("modal")
    if found:
        return [found]
    return [sys.executable, "-m", "modal"]


def _is_missing_remote_error(err: str) -> bool:
    low = err.lower()
    return "not found" in low or "no such file or directory" in low or "does not exist" in low


def volume_get(
    volume: str,
    remote_path: str,
    local_path: Path,
    *,
    verbose: bool = False,
) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [*modal_cli(), "volume", "get", volume, remote_path, str(local_path), "--force"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if verbose and err and not _is_missing_remote_error(err):
            print(err, file=sys.stderr)
        return False
    return local_path.is_file() or local_path.is_dir()


def volume_get_dir(
    volume: str,
    remote_dir: str,
    local_dir: Path,
    *,
    verbose: bool = False,
) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [*modal_cli(), "volume", "get", volume, remote_dir, str(local_dir), "--force"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if verbose and err and not _is_missing_remote_error(err):
            print(err, file=sys.stderr)
        return False
    return local_dir.exists()


def volume_has_run(volume: str, run_id: str) -> bool:
    proc = subprocess.run(
        [*modal_cli(), "volume", "ls", volume, run_id],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and bool((proc.stdout or "").strip())


def sync_run_meta(
    run_id: str,
    *,
    root: Path = PROJECT_ROOT,
    volume: str = DEFAULT_RUNS_VOLUME,
) -> bool:
    paths = RunPaths(run_id=run_id, root=root)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    return volume_get(volume, f"{run_id}/manifest.json", paths.manifest_path)


def sync_plot_artifacts(
    run_id: str,
    *,
    root: Path = PROJECT_ROOT,
    volume: str = DEFAULT_RUNS_VOLUME,
) -> Path | None:
    paths = RunPaths(run_id=run_id, root=root)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    got_any = False
    for name in PLOT_FILES:
        if volume_get(volume, f"{run_id}/plots/{name}", paths.plots_dir / name):
            got_any = True
    return paths.training_plot if got_any and paths.training_plot.is_file() else None


def sync_eval_logs(
    run_id: str,
    *,
    root: Path = PROJECT_ROOT,
    volume: str = DEFAULT_RUNS_VOLUME,
) -> bool:
    paths = RunPaths(run_id=run_id, root=root)
    paths.eval_log_dir.mkdir(parents=True, exist_ok=True)
    return any(
        volume_get(volume, f"{run_id}/eval_logs/{name}", paths.eval_log_dir / name)
        for name in EVAL_FILES
    )


def pull_all(
    run_id: str,
    *,
    root: Path = PROJECT_ROOT,
    volume: str = DEFAULT_RUNS_VOLUME,
) -> Path:
    """Download Runs/<run_id> from the volume. The modal CLI re-creates the remote
    directory NAME under the destination (paths are relative to the remote parent),
    so the destination must be the Runs/ parent — passing Runs/<run_id> would nest
    everything one level too deep (Runs/<id>/<id>/...)."""
    local_run = root / "Runs" / run_id
    ok = volume_get_dir(volume, run_id, root / "Runs")
    if not ok or not (local_run / "manifest.json").is_file():
        raise RuntimeError(
            f"pull_all({run_id!r}) did not produce {local_run}/manifest.json — the "
            "remote run may not exist on the volume or the download failed."
        )
    return local_run


def watch_run(
    run_id: str,
    *,
    interval_sec: float,
    open_plot: bool,
    root: Path = PROJECT_ROOT,
    volume: str = DEFAULT_RUNS_VOLUME,
) -> None:
    from rlbot.visualize import open_plot_file

    paths = RunPaths(run_id=run_id, root=root)
    last_mtime: float | None = None
    opened = False
    waiting_logged = False
    print(
        f"Watching Modal volume {volume!r} for run {run_id!r} "
        f"(every {interval_sec:g}s) → {paths.run_dir}",
        flush=True,
    )
    if not volume_has_run(volume, run_id):
        print(
            f"[modal] Run {run_id!r} not on volume {volume!r} yet. "
            "Check Modal logs for the run id.",
            flush=True,
        )
    else:
        sync_run_meta(run_id, root=root, volume=volume)
        print(
            f"[modal] Run on volume. training.png after first viz (~{VIZ_FREQ_DEFAULT:,} steps).",
            flush=True,
        )
        waiting_logged = True
    print("View in IDE: Runs/<run_id>/plots/training.png", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            if volume_has_run(volume, run_id):
                sync_run_meta(run_id, root=root, volume=volume)
            plot_path = sync_plot_artifacts(run_id, root=root, volume=volume)
            sync_eval_logs(run_id, root=root, volume=volume)
            if plot_path and plot_path.is_file():
                mtime = plot_path.stat().st_mtime
                if last_mtime is None or mtime > last_mtime:
                    last_mtime = mtime
                    print(f"[modal] updated {plot_path}", flush=True)
                    if open_plot and not opened:
                        open_plot_file(plot_path)
                        opened = True
            elif volume_has_run(volume, run_id) and not waiting_logged:
                print(
                    f"[modal] Waiting for training.png (~{VIZ_FREQ_DEFAULT:,} steps).",
                    flush=True,
                )
                waiting_logged = True
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\nStopped.")


def sync_cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sync Modal run artifacts to local Runs/")
    parser.add_argument("--run-id", required=True, help="Run id on rlbot-runs volume")
    parser.add_argument("--volume", default=DEFAULT_RUNS_VOLUME)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--open", dest="open_plot", action="store_true")
    parser.add_argument("--pull-all", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.pull_all:
        try:
            print(f"Pulled run → {pull_all(args.run_id, volume=args.volume)}")
        except RuntimeError as exc:
            raise SystemExit(f"sync --pull-all failed: {exc}")
        return
    if args.watch:
        watch_run(
            args.run_id,
            interval_sec=args.interval,
            open_plot=args.open_plot,
            volume=args.volume,
        )
        return

    plot_path = sync_plot_artifacts(args.run_id, volume=args.volume)
    sync_eval_logs(args.run_id, volume=args.volume)
    if plot_path:
        print(f"Synced plot → {plot_path}")
        if args.open_plot:
            from rlbot.visualize import open_plot_file

            open_plot_file(plot_path)
    elif volume_has_run(args.volume, args.run_id):
        sync_run_meta(args.run_id, volume=args.volume)
        print(
            f"Run {args.run_id!r} on volume; training.png not ready yet "
            f"(~{VIZ_FREQ_DEFAULT:,} steps). Use --watch."
        )
    else:
        print(f"No plot for {args.run_id!r} on volume {args.volume!r} yet.")
