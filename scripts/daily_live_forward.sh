#!/usr/bin/env bash
# Daily LIVE forward loop: one decision/day + dashboard mark.
#
# After the US cash close (~18:00 America/New_York on weekdays):
#   1. Refresh global yfinance cache
#   2. shadow_trade record  — append today's target weights (one trade)
#   3. shadow_trade reconcile — mark prior decisions when bars are available
#   4. forward_mark export  — rebuild NAV chart (model / EW / SPY) for /ops/forward
#
# Install (launchd, macOS):
#   bash scripts/install_live_forward_launchd.sh
#
# Manual:
#   bash scripts/daily_live_forward.sh [--run-id LIVE_MODEL]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="LIVE_MODEL"
if [[ "${1:-}" == "--run-id" && -n "${2:-}" ]]; then
  RUN_ID="$2"
elif [[ -n "${MARKETTRAINER_LIVE_RUN_ID:-}" ]]; then
  RUN_ID="$MARKETTRAINER_LIVE_RUN_ID"
fi

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
LOG_DIR="$ROOT/execution/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/daily_live_forward_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

echo "[daily_live] start run_id=$RUN_ID utc=$STAMP"

if [[ ! -f "Runs/$RUN_ID/models/best/best_model.zip" ]]; then
  echo "[daily_live] ERROR: missing Runs/$RUN_ID/models/best/best_model.zip" >&2
  exit 1
fi

# One refresh for the whole loop (shadow + forward mark share the global cache).
echo "[daily_live] refreshing global data cache..."
python - <<'PY'
from rlbot.data_utils import fetch_aligned_daily, save_cache
from rlbot.rl_config import get_config, load_config, set_config
from rlbot.run_artifacts import PROJECT_ROOT, resolve_data_cache

set_config(load_config(PROJECT_ROOT / "config" / "config.yaml"))
cfg = get_config()
cache_path = resolve_data_cache()
print(f"[daily_live] cache → {cache_path}")
idx, ohlcv, rsi, macd, macro, fd, fdm, trend, avol, mvol, live = fetch_aligned_daily(
    symbols_dict=cfg.universe.assets,
    since=cfg.data.since,
    until=None,
    fracdiff_d=cfg.data.fracdiff_d,
)
save_cache(
    str(cache_path),
    idx, ohlcv, rsi, macd, macro, fd, fdm, trend, avol, mvol,
    asset_live=live,
    fracdiff_d=cfg.data.fracdiff_d,
    tickers=list(cfg.universe.tickers),
)
print(f"[daily_live] cache bars={len(idx)} through {idx[-1].date()}")
PY

CACHE="$(python - <<'PY'
from rlbot.run_artifacts import resolve_data_cache
print(resolve_data_cache())
PY
)"

echo "[daily_live] shadow record (one trade / decision bar)..."
python scripts/shadow_trade.py record --run-id "$RUN_ID" --checkpoint best --data-cache "$CACHE"

echo "[daily_live] shadow reconcile..."
python scripts/shadow_trade.py reconcile --run-id "$RUN_ID" --checkpoint best --data-cache "$CACHE"

echo "[daily_live] shadow report..."
python scripts/shadow_trade.py report --run-id "$RUN_ID" --checkpoint best || true

echo "[daily_live] forward mark (dashboard series)..."
python scripts/forward_mark.py --run-id "$RUN_ID"

echo "[daily_live] done → /ops/forward (active=$RUN_ID)"
echo "[daily_live] ledger: execution/shadow_ledger_${RUN_ID}.jsonl"
echo "[daily_live] mark:   Runs/$RUN_ID/forward_mark.json"
