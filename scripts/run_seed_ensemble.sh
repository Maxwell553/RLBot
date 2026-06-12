#!/usr/bin/env bash
# Train multiple RNG seeds with the same calendar/holdout flags (sequential).
# Run IDs: <COHORT>_seed_<SEED>  (e.g. my_cohort_seed_42)
# Pass holdout dates on the CLI (see docs/RESEARCH.md); they are stored in each run manifest.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

COHORT=""
SEEDS="42 101 777 2026 999"
EXTRA=()

usage() {
  echo "Usage: $0 --cohort PREFIX [--seeds \"42 101 ...\"] -- [train.py flags]"
  echo "  Example:"
  echo "    $0 --cohort wf3 -- --train-end 2019-12-31 --holdout-start 2020-01-01 \\"
  echo "      --holdout-end 2021-06-30 --until 2021-06-30 --timesteps 50000000"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cohort) COHORT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    -h|--help) usage ;;
    --) shift; EXTRA=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

[[ -n "$COHORT" ]] || usage
[[ ${#EXTRA[@]} -gt 0 ]] || usage

for SEED in $SEEDS; do
  RUN_ID="${COHORT}_seed_${SEED}"
  if [[ -f "Runs/${RUN_ID}/training_summary.json" ]]; then
    echo "========== seed ${SEED} | run-id ${RUN_ID} already trained; skipping =========="
    continue
  fi
  EXTRA_FLAGS=()
  if [[ -f "Runs/${RUN_ID}/manifest.json" ]]; then
    echo "(stale crashed run dir for ${RUN_ID} — retraining with --overwrite-run)"
    EXTRA_FLAGS+=(--overwrite-run)
  fi
  echo "========== seed ${SEED} | run-id ${RUN_ID} =========="
  "$PY" scripts/train.py \
    --seed "$SEED" \
    --run-id "$RUN_ID" \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    "${EXTRA[@]}"
done

echo "Done. Backtest cohort:"
echo "  $PY scripts/backtest.py --ensemble-prefix ${COHORT} --detailed"
