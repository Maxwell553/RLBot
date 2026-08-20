#!/usr/bin/env bash
# Start the headless 5m collector if it is not already running.
# Must be launched from a process that may read ~/Desktop (Terminal, start_ui).
# Double-forks so Python.app is not killed when this shell exits.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/execution/logs"
mkdir -p "$LOG_DIR"
PIDFILE="$LOG_DIR/forward_loop.pid"
SCRIPT="$ROOT/scripts/frontend_api_lite.py"
OUT="$LOG_DIR/forward_loop_stdout.log"
ERR="$LOG_DIR/forward_loop_stderr.log"

already=""
if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    if ps -p "$old" -o command= 2>/dev/null | grep -q 'scripts/frontend_api_lite.py --collect-loop'; then
      already="$old"
    fi
  fi
fi
if [[ -z "$already" ]]; then
  already="$(pgrep -f 'scripts/frontend_api_lite.py --collect-loop' | head -1 || true)"
fi
if [[ -n "$already" ]]; then
  echo "[forward-collector] already running pid=$already"
  echo "$already" >"$PIDFILE"
  exit 0
fi

/usr/bin/python3 - "$SCRIPT" "$OUT" "$ERR" "$PIDFILE" <<'PY'
import os, sys
script, out_path, err_path, pid_path = sys.argv[1:]
if os.fork():
    os._exit(0)
os.setsid()
if os.fork():
    os._exit(0)
os.chdir("/tmp")
os.umask(0)
devnull = os.open("/dev/null", os.O_RDWR)
os.dup2(devnull, 0)
out = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
err = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(out, 1)
os.dup2(err, 2)
with open(pid_path, "w", encoding="utf-8") as fh:
    fh.write(str(os.getpid()))
os.execv("/usr/bin/python3", ["python3", script, "--collect-loop", "--interval", "300"])
PY
sleep 0.3
pid="$(cat "$PIDFILE" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "[forward-collector] started pid=$pid (5m Yahoo + GE1/Crest paper + RLModel shadow)"
else
  echo "[forward-collector] failed to start; see $ERR" >&2
  exit 1
fi
