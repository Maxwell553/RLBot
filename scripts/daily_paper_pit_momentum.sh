#!/usr/bin/env bash
# Daily paper loop for locked PIT S&P momentum (FINALMODEL → /ops/forward).
#
# After the US cash close:
#   1. Refresh daily adj-close cache (PIT members + SPY)
#   2. Month-end → compute targets (no orders that day)
#   3. Next session → paper orders to hit targets (lag=1)
#   4. Write shadow ledger + forward mark; activate FINALMODEL
#
# Manual:
#   bash scripts/daily_paper_pit_momentum.sh
#   bash scripts/daily_paper_pit_momentum.sh --as-of 2026-06-30

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
LOG_DIR="$ROOT/execution/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/daily_paper_pit_momentum_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

echo "[daily_paper_pit] start utc=$STAMP"

EXTRA=()
if [[ "${1:-}" == "--as-of" && -n "${2:-}" ]]; then
  EXTRA+=(--as-of "$2")
fi

python scripts/paper_pit_momentum.py run-day --refresh-data "${EXTRA[@]}"

echo "[daily_paper_pit] done → /ops/forward (FINALMODEL)"
echo "[daily_paper_pit] state: execution/paper_pit_momentum/state.json"
echo "[daily_paper_pit] ledger: execution/shadow_ledger_FINALMODEL.jsonl"
