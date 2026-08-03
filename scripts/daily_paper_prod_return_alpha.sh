#!/usr/bin/env bash
# Daily paper loop for GeneralEquity1 pack → /ops/forward.
#
#   1. Refresh daily OHLC (yfinance) for MTM
#   2. Pull targets from locked GeneralEquity1/ pack
#   3. Paper-rebalance when due
#   4. Write shadow ledger + forward mark; activate GENERAL_EQUITY1
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
LOG="$LOG_DIR/daily_paper_general_equity1_${STAMP}.log"
EXTRA=("$@")
{
  echo "[daily_paper_ge1] start ${STAMP}"
  python scripts/paper_prod_return_alpha.py run-day --refresh-data "${EXTRA[@]}"
  echo "[daily_paper_ge1] done → /ops/forward (GENERAL_EQUITY1)"
  echo "[daily_paper_ge1] state: execution/paper_general_equity1/state.json"
  echo "[daily_paper_ge1] ledger: execution/shadow_ledger_GENERAL_EQUITY1.jsonl"
} | tee "$LOG"
