# Training on Modal

Run long GPU training jobs on [Modal](https://modal.com) while keeping the same artifact layout as local training: `Runs/<run_id>/` (models, plots, logs, TensorBoard, manifest). Note: runs are long and training can be potentially very expensive. 

## Code layout


| File                   | Role                                                                                                |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| `scripts/modal_app.py` | Modal App: remote train, web endpoints, `upload_cache` / `list_runs` / `serve_plot`, and `sync` CLI |
| `rlbot/modal_cloud.py` | Volume commit hooks (used by `train.py` on Modal) and artifact sync helpers                         |


`modal_cloud` stays separate from the App file so local `train.py` can import volume-commit hooks without loading the Modal SDK at import time.

## Setup

```bash
source .venv/bin/activate
pip install -e ".[modal]"
modal setup   # once per machine; links your Modal account
```

Optional: upload a local data cache so the remote job skips yfinance download:

```bash
# After a local --refresh-data to create the cache
modal run scripts/modal_app.py::upload_cache
```

## Launch training

Pass the same flags you would give `scripts/train.py` after `--`:

```bash
modal run scripts/modal_app.py::train -- \
  --window 2 \
  --timesteps 50000000 \
  --since 2006-01-01 \
  --train-end 2017-12-31 \
  --holdout-start 2018-01-01 \
  --holdout-end 2019-12-31 \
  --until 2019-12-31
```

Pick a GPU (default `A10G`). The launch broker scales vCPUs, `n_envs`, and `batch_size` to match:


| GPU       | vCPUs | `n_envs` | `batch_size` (auto) |
| --------- | ----- | -------- | ------------------- |
| T4        | 4     | 8        | 8,192               |
| A10G / L4 | 16    | 16       | 16,384              |
| A100      | 32    | 32       | 32,768              |
| H100      | 64    | 64       | 65,536              |


GPU choice is fixed at launch (`--modal-gpu`); SB3 cannot rescale `n_envs` mid-run. To train faster, pick a bigger card before starting (or stop and `--resume` from the latest checkpoint on a faster tier). Rough wall-clock vs ~6h on a Mac:


| `--modal-gpu`  | Parallel envs | Typical speedup |
| -------------- | ------------- | --------------- |
| A10G (default) | 16            | ~1×             |
| A100           | 32            | ~1.5–2×         |
| H100           | 64            | ~2–3×           |


`config.yaml` sets `n_epochs: 3` and a baseline `batch_size: 16384` for local 16-env training (~12 backprop loops per PPO pause). Modal overrides `n_envs` and `batch_size` at launch; `n_epochs` stays in config.

```bash
modal run scripts/modal_app.py::train -- --modal-gpu A100 --window 2 --timesteps 50000000 ...
```

Maximum throughput (highest cost):

```bash
modal run scripts/modal_app.py::train -- --modal-gpu H100 --window 1 --run-id <RUN_ID> --timesteps 50000000 ...
```

Use an explicit run id when you plan to watch or sync artifacts:

```bash
modal run scripts/modal_app.py::train -- --run-id <RUN_ID> --timesteps 50000000 ...
```

Remote writes go to Modal volumes:


| Volume        | Mount               | Contents                                               |
| ------------- | ------------------- | ------------------------------------------------------ |
| `rlbot-runs`  | `/workspace/Runs`   | Per-run tree (`manifest.json`, `models/`, `plots/`, …) |
| `rlbot-cache` | `/workspace/.cache` | Shared `data_cache.npz`                                |


After each training plot refresh, the job commits the runs volume so local sync and web endpoints can see updates.

## Watch plots while training

In a second terminal on your device:

```bash
python scripts/modal_app.py sync --run-id <RUN_ID> --watch
```

This polls the `rlbot-runs` volume every 30 seconds (change with `--interval`) and writes into local `Runs/<run_id>/plots/training.png`. Open that file in Cursor/your IDE — it updates in place without spawning Preview. Omit `--open` (default); `--open` only launches the OS viewer once when the first plot arrives.

One-shot sync (no watch):

```bash
python scripts/modal_app.py sync --run-id <RUN_ID>
```

Pull the full run after training finishes (models, logs, TensorBoard, etc.):

```bash
python scripts/modal_app.py sync --run-id <RUN_ID> --pull-all
python scripts/backtest.py --run-id <RUN_ID> --checkpoint best --plot-tag best
```

List run folders on the volume:

```bash
modal run scripts/modal_app.py::list_runs
```

## Live plot URL (browser)

Deploy the app once:

```bash
modal deploy scripts/modal_app.py
```

Open the plot endpoint (replace host and run id):

```
https://<your-workspace>--rlbot-train-plot.modal.run?run_id=<RUN_ID>
```

Run status JSON:

```
https://<your-workspace>--rlbot-train-status.modal.run?run_id=<RUN_ID>
```

For ephemeral dev URLs:

```bash
modal serve scripts/modal_app.py
```

Then open the plot route with `?run_id=<RUN_ID>`.

## Resume on Modal

Checkpoints land on the runs volume under `Runs/<run_id>/models/checkpoints/`. After a **crash or preemption**, continue with `--resume` (restores curriculum + entropy schedule). Use `--finetune` only for an intentional second-stage experiment (lower LR/entropy, no curriculum).

```bash
modal run scripts/modal_app.py::train -- \
  --run-id <RUN_ID> \
  --resume Runs/<RUN_ID>/models/checkpoints/ppo_<steps>_steps.zip \
  --timesteps 50000000
```

(Paths are inside the container at `/workspace/Runs/...`.)

## Tips

- **Run id:** Auto ids (`--window N` → `W{N}_MMDD`, month/day at launch) are generated on the remote host. Check Modal logs for `Run id:` or pass `--run-id <RUN_ID>` for predictable sync.
- **Timeout:** Modal caps each container at **24 hours** (we set the max as headroom). A 50M-step run that takes ~5h locally should finish in one session on an A10G/A100 — use `--resume` (not `--finetune`) if the job crashes or is preempted (checkpoints every 1M steps).
- **Costs:** 50M-step jobs are long; pick GPU in `rlbot/modal_cloud.py` (`DEFAULT_GPU` or `--modal-gpu`) to match your budget.
- **n_envs:** Linux Modal containers usually spawn `SubprocVecEnv` faster than macOS; you can still tune `--n-envs` if memory is tight.
- **Local vs remote:** Local `Runs/` is gitignored. After `--pull-all`, local backtest uses the same paths as a local training run.

