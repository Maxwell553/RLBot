#!/usr/bin/env bash
# Window 1 OOS backtest — holdout must match window1_train.sh (2016–2017).
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN_ID="${1:?Usage: $0 <RUN_ID>}"
PYTHON="${PYTHON:-.venv/bin/python}"

exec "$PYTHON" scripts/backtest.py \
  --run-id "$RUN_ID" \
  --until 2017-12-31 \
  --train-end 2015-12-31 \
  --holdout-start 2016-01-01 \
  --holdout-end 2017-12-31 \
  --detailed \
  --stochastic-paths 30 \
  --plot-tag best \
  "$@"
