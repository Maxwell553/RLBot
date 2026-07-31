#!/usr/bin/env bash
# One-command local UI: research API (:8787) + workflow API (:8790) + Vite (:5173).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/frontend"

# Finder/iCloud duplicates ("node_modules 2") make Vite crawl forever. Never
# block startup on a move — kick it to /tmp in the background (may take minutes).
for d in "node_modules 2" "node_modules 3"; do
  if [[ -e "$d" ]]; then
    junk="/tmp/markettrainer-nm-junk-$$-${RANDOM}"
    echo "[start_ui] Quarantining '$d' in background (iCloud may be slow)…"
    (mv "$d" "$junk" && rm -rf "$junk") >/dev/null 2>&1 &
  fi
done

# Clear stale listeners from a previous crashed start (orphan Vite on :5173).
for port in 5173 8787 8790; do
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "[start_ui] Freeing port $port (pids: $pids)"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 0.15
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
    fi
  fi
done

if [[ ! -e node_modules ]]; then
  echo "[start_ui] Installing frontend dependencies…"
  npm install
fi
exec npm run dev "$@"
