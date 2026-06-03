#!/usr/bin/env bash
# Walk-forward window 4 — OOS 2021-07-01 → 2022-12-31
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
ARGS=(
  --since 2006-01-01
  --until 2022-12-31
  --train-end 2021-06-30
  --holdout-start 2021-07-01
  --holdout-end 2022-12-31
  --timesteps 65000000
)
[[ -n "${RUN_ID:-}" ]] && ARGS+=(--run-id "$RUN_ID")
exec "$PYTHON" scripts/train.py "${ARGS[@]}" "$@"
