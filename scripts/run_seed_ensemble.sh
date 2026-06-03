#!/usr/bin/env bash
# Train a walk-forward window under multiple RNG seeds (sequential).
# Run IDs: <COHORT>_seed_<SEED>  (e.g. my_cohort_seed_42)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

WINDOW=""
COHORT=""
SEEDS="42 101 777 2026 999"
EXTRA=()

usage() {
  echo "Usage: $0 --window {1..6} --cohort PREFIX [--seeds \"42 101 ...\"]"
  echo "  Example: $0 --window 3 --cohort my_cohort"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --window) WINDOW="$2"; shift 2 ;;
    --cohort) COHORT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

[[ -n "$WINDOW" && -n "$COHORT" ]] || usage

case "$WINDOW" in
  1)
    TRAIN_ARGS=(
      --since 2006-01-01 --until 2017-12-31
      --train-end 2015-12-31 --holdout-start 2016-01-01 --holdout-end 2017-12-31
    )
    ;;
  2)
    TRAIN_ARGS=(
      --since 2006-01-01 --until 2019-12-31
      --train-end 2017-12-31 --holdout-start 2018-01-01 --holdout-end 2019-12-31
    )
    ;;
  3)
    TRAIN_ARGS=(
      --since 2006-01-01 --until 2021-06-30
      --train-end 2019-12-31 --holdout-start 2020-01-01 --holdout-end 2021-06-30
    )
    ;;
  4)
    TRAIN_ARGS=(
      --since 2006-01-01 --until 2022-12-31
      --train-end 2021-06-30 --holdout-start 2021-07-01 --holdout-end 2022-12-31
    )
    ;;
  5)
    TRAIN_ARGS=(
      --since 2006-01-01 --until 2024-12-31
      --train-end 2022-12-31 --holdout-start 2023-01-01 --holdout-end 2024-12-31
    )
    ;;
  6)
    TRAIN_ARGS=(
      --since 2006-01-01
      --train-end 2024-12-31 --holdout-start 2025-01-01
    )
    ;;
  *) echo "Invalid --window $WINDOW (use 1–6)"; exit 1 ;;
esac

for SEED in $SEEDS; do
  RUN_ID="${COHORT}_seed_${SEED}"
  echo "========== Window ${WINDOW} | seed ${SEED} | run-id ${RUN_ID} =========="
  "$PY" scripts/train.py \
    "${TRAIN_ARGS[@]}" \
    --timesteps 65000000 \
    --seed "$SEED" \
    --run-id "$RUN_ID" \
    "${EXTRA[@]}"
done

echo "Done. Backtest cohort:"
echo "  $PY scripts/backtest.py --ensemble-prefix ${COHORT} --detailed"
