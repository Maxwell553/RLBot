#!/usr/bin/env bash
# Walk-forward window 2 — OOS 2018-01-01 → 2019-12-31
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
ARGS=(
  --since 2006-01-01
  --until 2019-12-31
  --train-end 2017-12-31
  --holdout-start 2018-01-01
  --holdout-end 2019-12-31
  --timesteps 65000000
)
[[ -n "${RUN_ID:-}" ]] && ARGS+=(--run-id "$RUN_ID")
exec "$PYTHON" scripts/train.py "${ARGS[@]}" "$@"
