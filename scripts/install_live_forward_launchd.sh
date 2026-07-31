#!/usr/bin/env bash
# Install a weekday LaunchAgent that runs scripts/daily_live_forward.sh at 18:15 ET.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.markettrainer.live-forward"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="$ROOT/execution/logs"
mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"

RUN_ID="${MARKETTRAINER_LIVE_RUN_ID:-LIVE_MODEL}"
PYTHON_BIN="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>WorkingDirectory</key>
  <string>${ROOT}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${ROOT}/scripts/daily_live_forward.sh</string>
    <string>--run-id</string>
    <string>${RUN_ID}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>${ROOT}/.venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>MARKETTRAINER_LIVE_RUN_ID</key>
    <string>${RUN_ID}</string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>15</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>15</integer></dict>
  </array>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/launchd_stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/launchd_stderr.log</string>
  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "Installed $PLIST"
echo "Runs weekdays 18:15 local time for run_id=$RUN_ID"
echo "Test now:  bash $ROOT/scripts/daily_live_forward.sh --run-id $RUN_ID"
echo "Unload:    launchctl unload $PLIST"
