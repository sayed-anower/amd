#!/usr/bin/env bash
# Start the retrieval API. Logs stay inside this project directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
LOG="$ROOT/server.log"
PID_FILE="$ROOT/server.pid"

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[start_api] already running on http://127.0.0.1:$PORT"
  exit 0
fi

echo "[start_api] starting on http://$HOST:$PORT"
nohup python -m amd_websearch --serve --host "$HOST" --port "$PORT" >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[start_api] ready (log: $LOG)"
    exit 0
  fi
  sleep 0.5
done

echo "[start_api] did not become healthy; see $LOG" >&2
tail -n 20 "$LOG" >&2 || true
exit 1
