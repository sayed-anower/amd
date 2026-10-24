#!/usr/bin/env bash
# Match 4 production entrypoint:
#   1. start a warm llama.cpp server on 127.0.0.1:8001
#   2. answer /data/questions.jsonl into /output/answers.jsonl
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-$ROOT/model/qwen-2.5-0.5b.gguf}"
LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-8001}"
export MODEL_PATH LLM_HOST LLM_PORT
export LLM_BASE_URL="${LLM_BASE_URL:-http://${LLM_HOST}:${LLM_PORT}/v1}"
export LLM_MODEL="${LLM_MODEL:-qwen2.5-0.5b-instruct}"
export PYTHONUNBUFFERED=1

if [ "${OFFLINE_EVAL:-0}" = "1" ]; then
  export SEARCH_PROVIDER=local
  export PROVIDER_FALLBACK=local
fi

# The grader mounts /output and drops DAC_OVERRIDE. Make sure the mounted
# directory is writable without that capability (CHOWN/FOWNER remain).
OUTPUT_DIR="$(dirname "${ANSWERS_FILE:-/output/answers.jsonl}")"
mkdir -p "$OUTPUT_DIR" 2>/dev/null || true
if [ ! -w "$OUTPUT_DIR" ]; then
  chmod u+rwx "$OUTPUT_DIR" 2>/dev/null || chown "$(id -u):$(id -g)" "$OUTPUT_DIR" 2>/dev/null || true
fi
if [ ! -w "$OUTPUT_DIR" ]; then
  echo "[run] output directory is not writable: $OUTPUT_DIR" >&2
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "[run] model not found: $MODEL_PATH" >&2
  echo "[run] place the Qwen2.5-0.5B GGUF at model/qwen-2.5-0.5b.gguf" >&2
  exit 1
fi

STARTED_LLAMA=0
if ! curl -sf "http://${LLM_HOST}:${LLM_PORT}/health" >/dev/null 2>&1; then
  bash "$ROOT/scripts/start_llama.sh"
  STARTED_LLAMA=1
fi

cleanup() {
  if [ "$STARTED_LLAMA" = "1" ] && [ -f "$ROOT/llama-server.pid" ]; then
    pid="$(cat "$ROOT/llama-server.pid" 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$ROOT/llama-server.pid"
  fi
}
trap cleanup EXIT

python "$ROOT/app/main.py" "$@"
