# Training Automation

This folder contains the three-week, training-only ablation queue. It is intentionally
separate from `scripts/research.py` and does not run `scripts/backtest.py`.

## What It Runs

`hypotheses.yaml` defines 25 variants across canonical windows W1-W5:

- `25 variants * 5 windows = 125 training runs`
- each run calls `python scripts/train.py --config <generated> --window N --run-id <id> --seed <seed>`
- run IDs use the repo's default style: `W<window>_<month><day>`, with `_a`,
  `_b`, ... suffixes for multiple jobs on the same window/day
- generated configs are written to `Automation/generated_configs/`
- stdout/stderr logs are written to `Automation/logs/`
- queue state is appended to `Automation/queue_state.jsonl`
- a full queue manifest is written to `Automation/queue_manifest.json`

The queue trains only. You can backtest/analyze after returning.

## Quick Sanity Check

From the repo root:

```bash
.venv/bin/python Automation/run_training_queue.py --dry-run
```

Count selected jobs:

```bash
.venv/bin/python Automation/run_training_queue.py --dry-run | wc -l
```

The first line is the job count, the second line is the manifest path, then 125 commands.

## Launch

Use a persistent terminal/session. From the repo root:

```bash
.venv/bin/python Automation/run_training_queue.py --keep-going
```

If the machine has no `.venv`, use whichever Python environment has the project installed:

```bash
python Automation/run_training_queue.py --keep-going
```

## Resume After Crash Or Reboot

If a run was interrupted and has checkpoints:

```bash
.venv/bin/python Automation/run_training_queue.py --resume-incomplete --keep-going
```

The runner skips completed runs, and with `--resume-incomplete` it resumes an incomplete
run from the highest `Runs/<run_id>/models/checkpoints/ppo_*_steps.zip` checkpoint.

If `Automation/queue.lock` remains after a confirmed-dead process:

```bash
.venv/bin/python Automation/run_training_queue.py --resume-incomplete --keep-going --force-lock
```

## Smoke Test

To verify the machinery without launching full runs:

```bash
.venv/bin/python Automation/run_training_queue.py --dry-run --limit 3
```

For a tiny real smoke run, use an intentionally tiny timestep override and a throwaway
variant/window selection:

```bash
.venv/bin/python Automation/run_training_queue.py \
  --limit 1 \
  --timesteps 10000 \
  --no-viz
```

Do not use `--timesteps` for the real three-week campaign.

## Editing The Queue

Edit `Automation/hypotheses.yaml`, then rerun with `--dry-run`. The runner regenerates
`Automation/generated_configs/` and `Automation/queue_manifest.json` each time.

Run IDs use the normal training style and are recorded in `Automation/queue_manifest.json`:

```text
W1_625
W1_625_a
W1_625_b
```

The manifest maps each run ID back to its variant, seed, window, config path, and patch.
If a completed run with that ID exists, it is skipped. If you rerun the queue on a later
day, the runner reuses the already assigned run IDs from `Automation/queue_manifest.json`.
