#!/usr/bin/env bash
# Install a weekday 15:45 local-time LaunchAgent for LiveTrader run-if-due.
#
# Uses the machine timezone (set macOS to America/New_York for the cash close).
# Does NOT arm live submit unless LIVE_TRADER_LAUNCHD_LIVE=1 is in LiveTrader/.env
# and config allow_live is true.
#
#   bash LiveTrader/install_run_if_due_launchd.sh
#   python LiveTrader/trader.py run-if-due --preview-only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.markettrainer.livetrader-run-if-due"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
SUPPORT="$HOME/Library/Application Support/MarketTrainer"
TRAMPOLINE="$SUPPORT/livetrader_run_if_due.sh"
LOG_DIR="$ROOT/execution/logs"
mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents" "$SUPPORT"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="/usr/bin/python3"
fi

cat >"$TRAMPOLINE" <<EOF
#!/bin/bash
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/LiveTrader:${ROOT}"
cd /tmp || exit 1
if [[ -f "${ROOT}/LiveTrader/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/LiveTrader/.env"
  set +a
fi
exec "${PYTHON}" "${ROOT}/LiveTrader/trader.py" run-if-due --refresh-data
EOF
chmod +x "$TRAMPOLINE"

# launchd Weekday: 0=Sun … 5=Fri. 15:45 local (use ET on this Mac).
INTERVALS=""
for wd in 1 2 3 4 5; do
  INTERVALS="${INTERVALS}
    <dict>
      <key>Weekday</key><integer>${wd}</integer>
      <key>Hour</key><integer>15</integer>
      <key>Minute</key><integer>45</integer>
    </dict>"
done

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
  <key>RunAtLoad</key>
  <false/>
  <key>StartCalendarInterval</key>
  <array>
${INTERVALS}
  </array>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/livetrader_run_if_due_stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/livetrader_run_if_due_stderr.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/${UID}" "$PLIST" 2>/dev/null || launchctl unload "$PLIST" 2>/dev/null || true
if ! launchctl bootstrap "gui/${UID}" "$PLIST" 2>/dev/null; then
  launchctl load "$PLIST"
fi
echo "Installed $PLIST"
echo "Trampoline: $TRAMPOLINE"
echo "Weekdays 15:45 local → python LiveTrader/trader.py run-if-due"
echo "Unload:     launchctl bootout gui/\$UID $PLIST"
echo "One shot:   python $ROOT/LiveTrader/trader.py run-if-due --preview-only"
