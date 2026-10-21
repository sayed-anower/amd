#!/usr/bin/env bash
# Stop the API and llama.cpp servers started by the scripts in this folder.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for name in server llama-server; do
  pid_file="$ROOT/$name.pid"
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      echo "[stop] stopped $name (pid $pid)"
    fi
    rm -f "$pid_file"
  fi
done
