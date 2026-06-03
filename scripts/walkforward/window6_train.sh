#!/usr/bin/env bash
# Walk-forward window 6 — OOS 2025-01-01 → latest bar
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
ARGS=(
  --since 2006-01-01
  --train-end 2024-12-31
  --holdout-start 2025-01-01
  --timesteps 65000000
)
[[ -n "${RUN_ID:-}" ]] && ARGS+=(--run-id "$RUN_ID")
exec "$PYTHON" scripts/train.py "${ARGS[@]}" "$@"
