#!/usr/bin/env bash
# Start llama.cpp's OpenAI-compatible server for Qwen2.5-0.5B-Instruct.
# All logs stay inside this project directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
HOST="${LLM_HOST:-127.0.0.1}"
PORT="${LLM_PORT:-8099}"
THREADS="${LLM_THREADS:-$(nproc)}"
CONTEXT="${LLM_CONTEXT:-2048}"
LOG="$ROOT/llama-server.log"
PID_FILE="$ROOT/llama-server.pid"

if [ ! -f "$MODEL" ]; then
  echo "[start_llama] model not found: $MODEL" >&2
  exit 1
fi

if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "[start_llama] already running on http://$HOST:$PORT"
  exit 0
fi

echo "[start_llama] starting $MODEL (threads=$THREADS, ctx=$CONTEXT)"
nohup ./llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -c "$CONTEXT" \
  -t "$THREADS" \
  --no-warmup \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

for _ in $(seq 1 120); do
  if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "[start_llama] ready on http://$HOST:$PORT (log: $LOG)"
    exit 0
  fi
  sleep 0.5
done

echo "[start_llama] did not become healthy; see $LOG" >&2
tail -n 20 "$LOG" >&2 || true
exit 1
