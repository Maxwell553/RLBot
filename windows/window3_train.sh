#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3

ARGS=(--since 2006-01-01 --holdout-days 365 --timesteps 65000000)
if [[ -n "${RUN_ID:-}" ]]; then
  ARGS+=(--run-id "$RUN_ID")
fi

exec "$PYTHON" train.py "${ARGS[@]}" "$@"
