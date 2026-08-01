#!/usr/bin/env bash
# Daily paper loop for prod_return_alpha_v1 (1360pctAlgo → /ops/forward).
#
#   1. Refresh daily OHLC (yfinance)
#   2. Compute weekly TQQQ + month-end dual targets
#   3. Paper-rebalance when due
#   4. Write shadow ledger + forward mark; activate PROD_RETURN_ALPHA
#
# Usage:
#   bash scripts/daily_paper_prod_return_alpha.sh
#   bash scripts/daily_paper_prod_return_alpha.sh --as-of 2026-06-30
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
if [[ -f .venv/bin/activate ]]; then
  # Prefer project venv when present.
  source .venv/bin/activate
fi
LOG_DIR="${MARKETTRAINER_LOG_DIR:-$ROOT/execution/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/daily_paper_prod_return_alpha_${STAMP}.log"
EXTRA=("$@")
{
  echo "[daily_paper_1360] start ${STAMP}"
  python scripts/paper_prod_return_alpha.py run-day --refresh-data "${EXTRA[@]}"
  echo "[daily_paper_1360] done → /ops/forward (PROD_RETURN_ALPHA)"
  echo "[daily_paper_1360] state: execution/paper_prod_return_alpha/state.json"
  echo "[daily_paper_1360] ledger: execution/shadow_ledger_PROD_RETURN_ALPHA.jsonl"
} | tee "$LOG"
