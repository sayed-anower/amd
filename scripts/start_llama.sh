#!/usr/bin/env bash
# Start llama.cpp's OpenAI-compatible server for Qwen2.5-0.5B-Instruct.
# Uses the prebuilt binary (no compilation). Logs stay inside this project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL_PATH:-${MODEL:-model/qwen-2.5-0.5b.gguf}}"
HOST="${LLM_HOST:-127.0.0.1}"
PORT="${LLM_PORT:-8001}"
THREADS="${LLM_THREADS:-$(nproc)}"
CONTEXT="${LLM_CONTEXT:-2048}"
LOG="${LLAMA_LOG:-$ROOT/llama-server.log}"
PID_FILE="${LLAMA_PID_FILE:-$ROOT/llama-server.pid}"

if [ ! -f "$MODEL" ]; then
  echo "[start_llama] model not found: $MODEL" >&2
  exit 1
fi

if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "[start_llama] already running on http://$HOST:$PORT"
  exit 0
fi

BIN="${LLAMA_SERVER:-}"
if [ -z "$BIN" ]; then
  BIN="$(command -v llama-server || true)"
fi
if [ -z "$BIN" ]; then
  BIN="$(find "$ROOT/llama-bin" -type f -name llama-server 2>/dev/null | head -n 1 || true)"
fi
if [ -z "$BIN" ]; then
  BIN="$ROOT/llama.cpp/build/bin/llama-server"
fi
if [ ! -x "$BIN" ]; then
  echo "[start_llama] llama-server not found; run scripts/fetch_llama.sh" >&2
  exit 1
fi

echo "[start_llama] starting $MODEL (threads=$THREADS, ctx=$CONTEXT, bin=$BIN)"
nohup "$BIN" \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -c "$CONTEXT" \
  -t "$THREADS" \
  --no-warmup \
  ${LLAMA_SERVER_EXTRA_ARGS:-} \
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
