#!/usr/bin/env bash
# Install a KeepAlive LaunchAgent for the headless forward collector.
#
# Writes 5m NAV marks + paper/shadow ledgers to execution/ even when the
# frontend is closed. Replaces the old weekday-18:15-only job.
#
#   bash scripts/install_live_forward_launchd.sh
#   python scripts/live_forward_loop.py --once     # one tick now
#   python scripts/live_forward_loop.py --status
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.markettrainer.live-forward"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
SUPPORT="$HOME/Library/Application Support/MarketTrainer"
TRAMPOLINE="$SUPPORT/live_forward_loop.sh"
LOG_DIR="$ROOT/execution/logs"
mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents" "$SUPPORT"

RUN_ID="${MARKETTRAINER_LIVE_RUN_ID:-RLModel}"

# Copy the stdlib collector off iCloud Desktop — LaunchAgents get
# "Operation not permitted" opening files under ~/Desktop.
cp "$ROOT/scripts/frontend_api_lite.py" "$SUPPORT/frontend_api_lite.py"
cat >"$TRAMPOLINE" <<EOF
#!/bin/bash
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MARKETTRAINER_LIVE_RUN_ID=${RUN_ID}
export MARKETTRAINER_ROOT="${ROOT}"
cd /tmp || exit 1
exec /usr/bin/python3 "${SUPPORT}/frontend_api_lite.py" --collect-loop --interval 300
EOF
chmod +x "$TRAMPOLINE"

cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>WorkingDirectory</key>
  <string>/tmp</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${TRAMPOLINE}</string>
  </array>
  <key>KeepAlive</key>
  <true/>
  <key>RunAtLoad</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>30</integer>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/forward_loop_stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/forward_loop_stderr.log</string>
</dict>
</plist>
EOF

echo "Unload:     launchctl bootout gui/\$UID $PLIST"

# Start the TCC-holding collector first so it owns the lock, then KeepAlive
# LaunchAgent (waits on the lock; takes over after reboot / collector exit).
bash "$ROOT/scripts/start_forward_collector.sh"

launchctl bootout "gui/${UID}" "$PLIST" 2>/dev/null || launchctl unload "$PLIST" 2>/dev/null || true
if ! launchctl bootstrap "gui/${UID}" "$PLIST" 2>/dev/null; then
  launchctl load "$PLIST"
fi
echo "Installed $PLIST"
echo "Trampoline: $TRAMPOLINE"
echo "KeepAlive collector every 5 minutes (Yahoo 5m + paper + RLModel shadow)."
echo "Active RL shadow run_id=$RUN_ID (after 18:00 ET, or immediately if the ledger is a cash reset)."
echo "Status:     python $ROOT/scripts/live_forward_loop.py --status"
echo "One tick:   python $ROOT/scripts/live_forward_loop.py --once"
echo "Unload:     launchctl bootout gui/\$UID $PLIST"
