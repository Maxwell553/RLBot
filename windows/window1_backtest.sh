#!/usr/bin/env bash
# Window 1 OOS backtest on 2021–2022 (must match window1_train.sh holdout).
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${1:?Usage: $0 <RUN_ID>}"
PYTHON="${PYTHON:-.venv/bin/python}"

exec "$PYTHON" backtest.py \
  --run-id "$RUN_ID" \
  --until 2022-12-31 \
  --train-end 2020-12-31 \
  --holdout-start 2021-01-01 \
  --holdout-end 2022-12-31 \
  --detailed \
  "$@"
