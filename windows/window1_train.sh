#!/usr/bin/env bash
# Thin launcher — all logic lives in train.py. See windows/README.md for the same flags.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3

ARGS=(
  --since 2006-01-01
  --until 2022-12-31
  --train-end 2020-12-31
  --holdout-start 2021-01-01
  --holdout-end 2022-12-31
  --timesteps 65000000
)
if [[ -n "${RUN_ID:-}" ]]; then
  ARGS+=(--run-id "$RUN_ID")
fi

exec "$PYTHON" train.py "${ARGS[@]}" "$@"
