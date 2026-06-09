"""
Modal GPU training for RLBot — single entry point for cloud train, sync, and utilities.

Setup:
  pip install -e ".[modal]" && modal setup

Train (pass train.py flags after ``--``):
  modal run scripts/modal_app.py -- --modal-gpu H100 --window 2 --run-id W2_605 ...

Sync plots locally while training:
  python scripts/modal_app.py sync --run-id W2_605 --watch

Pull full run tree after training:
  python scripts/modal_app.py sync --run-id W2_605 --pull-all

Upload local OHLCV cache:
  modal run scripts/modal_app.py::upload_cache
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Sync uses only the Modal CLI subprocess — no SDK import (matches old modal_sync.py).
if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == "sync":
    from rlbot.modal_cloud import sync_cli_main

    sync_cli_main(sys.argv[2:])
    raise SystemExit(0)

import modal

from rlbot.modal_cloud import (
    APP_NAME,
    DEFAULT_GPU,
    DEFAULT_TIMEOUT_SEC,
    MODAL_FLAG_ENV,
    N_STEPS,
    VOLUME_CACHE,
    VOLUME_CACHE_ENV,
    VOLUME_RUNS,
    VOLUME_RUNS_ENV,
    WORKSPACE,
    rollout_batch_size,
    extract_run_id,
    gpu_profile,
    normalize_gpu_name,
    parse_entrypoint_argv,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)

runs_volume = modal.Volume.from_name(VOLUME_RUNS, create_if_missing=True)
cache_volume = modal.Volume.from_name(VOLUME_CACHE, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("fastapi>=0.100.0")
    .add_local_dir(
        str(_REPO_ROOT),
        remote_path=WORKSPACE,
        ignore=[
            ".venv",
            "venv",
            "Runs",
            ".cache",
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "*.egg-info",
            "dist",
            "build",
            "archive",
            "models",
            "plots",
            "logs",
            "tb_logs",
            "runs",
            "ibkr_paper",
            "execution",
        ],
    )
)

_volume_mounts = {
    f"{WORKSPACE}/Runs": runs_volume,
    f"{WORKSPACE}/.cache": cache_volume,
}


@app.function(
    image=image,
    timeout=DEFAULT_TIMEOUT_SEC,
    volumes=_volume_mounts,
    retries=modal.Retries(max_retries=1, initial_delay=60.0),
)
def train_remote(
    train_argv: list[str],
    n_envs_override: int,
    batch_size_override: int,
) -> dict[str, Any]:
    """Run ``scripts/train.py`` with GPU-tuned n_envs and batch_size."""
    os.environ[MODAL_FLAG_ENV] = "1"
    os.environ[VOLUME_RUNS_ENV] = VOLUME_RUNS
    os.environ[VOLUME_CACHE_ENV] = VOLUME_CACHE
    os.chdir(WORKSPACE)

    # Broker values are *defaults*: prepend them so a user-passed --n-envs/--batch-size
    # (argparse last-wins) overrides the GPU profile instead of being silently ignored.
    def _user_passed(flag: str) -> bool:
        return any(a == flag or a.startswith(flag + "=") for a in train_argv)

    if _user_passed("--n-envs"):
        n_envs_override = int(
            train_argv[train_argv.index("--n-envs") + 1]
            if "--n-envs" in train_argv
            else next(a.split("=", 1)[1] for a in train_argv if a.startswith("--n-envs="))
        )
    if _user_passed("--batch-size"):
        batch_size_override = int(
            train_argv[train_argv.index("--batch-size") + 1]
            if "--batch-size" in train_argv
            else next(a.split("=", 1)[1] for a in train_argv if a.startswith("--batch-size="))
        )
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--n-envs",
        str(n_envs_override),
        "--batch-size",
        str(batch_size_override),
        *train_argv,
    ]
    rollout = N_STEPS * n_envs_override
    minibatches = rollout // max(batch_size_override, 1)
    print(f"[modal] cwd={os.getcwd()}", flush=True)
    print(
        f"[modal] rollout={rollout:,}  batch_size={batch_size_override:,}  "
        f"mini-batches/epoch≈{minibatches}",
        flush=True,
    )
    print(f"[modal] command: {' '.join(cmd)}", flush=True)

    proc = subprocess.run(cmd, check=False)
    runs_volume.commit()
    cache_volume.commit()

    return {
        "exit_code": int(proc.returncode),
        "run_id": extract_run_id(train_argv),
        "n_envs": n_envs_override,
        "batch_size": batch_size_override,
        "runs_volume": VOLUME_RUNS,
        "remote_runs_root": f"{WORKSPACE}/Runs",
    }


@app.function(
    image=image,
    volumes={f"{WORKSPACE}/Runs": runs_volume},
)
@modal.concurrent(max_inputs=100)
@modal.fastapi_endpoint(method="GET", label="plot")
def training_plot(run_id: str):
    from fastapi import Response
    from fastapi.responses import FileResponse

    runs_volume.reload()
    plot_path = Path(WORKSPACE) / "Runs" / run_id / "plots" / "training.png"
    if not plot_path.is_file():
        return Response(
            content=f"Plot not ready for run_id={run_id!r}.",
            status_code=404,
            media_type="text/plain",
        )
    return FileResponse(
        str(plot_path),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.function(
    image=image,
    volumes={f"{WORKSPACE}/Runs": runs_volume},
)
@modal.concurrent(max_inputs=100)
@modal.fastapi_endpoint(method="GET", label="status")
def run_status(run_id: str):
    import json

    from fastapi import Response

    runs_volume.reload()
    run_dir = Path(WORKSPACE) / "Runs" / run_id
    manifest_path = run_dir / "manifest.json"
    plot_path = run_dir / "plots" / "training.png"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "plot_ready": plot_path.is_file(),
        "manifest_ready": manifest_path.is_file(),
    }
    if manifest_path.is_file():
        try:
            payload["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            payload["manifest_error"] = str(exc)
    return Response(
        content=json.dumps(payload, indent=2, default=str),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@app.local_entrypoint()
def train(*arglist: str):
    """Launch training on Modal. Optional ``--modal-gpu H100`` in forwarded args."""
    gpu, train_argv = parse_entrypoint_argv(list(arglist))
    if not train_argv:
        print(
            "Pass train.py arguments after --\n"
            "  modal run scripts/modal_app.py -- --modal-gpu H100 --window 2 --run-id W2_605 ...\n"
            "Sync: python scripts/modal_app.py sync --run-id W2_605 --watch"
        )
        raise SystemExit(1)

    gpu_key = normalize_gpu_name(gpu)
    profile = gpu_profile(gpu)
    target_envs = profile["n_envs"]
    batch_size = rollout_batch_size(target_envs)
    rollout = N_STEPS * target_envs

    print("=" * 60)
    print("MODAL LAUNCH BROKER")
    print(f"  GPU:        {gpu_key}")
    print(f"  vCPUs:      {profile['cpus']}")
    print(f"  n_envs:     {target_envs}")
    print(f"  batch_size: {batch_size:,}  (rollout {rollout:,})")
    print(f"  timeout:    {DEFAULT_TIMEOUT_SEC // 3600}h")
    print("=" * 60)

    run_id = extract_run_id(train_argv)
    fn = train_remote.with_options(gpu=gpu_key, cpu=profile["cpus"])
    if run_id:
        print(
            f"[modal] Watch: python scripts/modal_app.py sync --run-id {run_id} --watch"
        )
    result = fn.remote(train_argv, target_envs, batch_size)
    print(f"[modal] Finished: {result}")
    if int(result.get("exit_code", 1)) != 0:
        raise SystemExit(int(result["exit_code"]))
    if run_id:
        print(
            f"\nPull artifacts: python scripts/modal_app.py sync --run-id {run_id} --pull-all\n"
            f"Backtest:         python scripts/backtest.py --run-id {run_id} --checkpoint best --plot-tag best"
        )


@app.local_entrypoint()
def upload_cache(local_path: str = ".cache/data_cache.npz", remote_name: str = "data_cache.npz"):
    path = Path(local_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Cache file not found: {path}")
    with cache_volume.batch_upload() as batch:
        batch.put_file(str(path), remote_name)
    print(f"Uploaded {path} → volume {VOLUME_CACHE!r} as /{remote_name}")


@app.local_entrypoint()
def list_runs():
    for entry in runs_volume.listdir("/"):
        name = entry.path.strip("/")
        if name and re.match(r"^W\d+_", name):
            print(name)


@app.local_entrypoint()
def serve_plot(run_id: str):
    url = training_plot.get_web_url()
    sep = "&" if "?" in url else "?"
    print(f"{url}{sep}run_id={run_id}")


if __name__ == "__main__":
    print("Usage:")
    print("  modal run scripts/modal_app.py -- [train.py flags]")
    print("  python scripts/modal_app.py sync --run-id W2_605 --watch")
    raise SystemExit(1)
