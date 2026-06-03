#!/usr/bin/env bash
# Walk-forward window 3 — OOS 2020-01-01 → 2021-06-30
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
ARGS=(
  --since 2006-01-01
  --until 2021-06-30
  --train-end 2019-12-31
  --holdout-start 2020-01-01
  --holdout-end 2021-06-30
  --timesteps 65000000
)
[[ -n "${RUN_ID:-}" ]] && ARGS+=(--run-id "$RUN_ID")
exec "$PYTHON" scripts/train.py "${ARGS[@]}" "$@"
