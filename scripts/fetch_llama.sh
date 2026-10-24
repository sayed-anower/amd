#!/usr/bin/env bash
# Download a prebuilt llama.cpp release. No compilation, ever.
#
# Default: Ubuntu x64 CPU build (works in the ROCm image and on any CI box).
# Override: LLAMA_ASSET=llama-b11849-bin-ubuntu-rocm-10.0-x64.tar.gz
#           LLAMA_BUILD=b11849
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${LLAMA_BUILD:-b11849}"
ASSET="${LLAMA_ASSET:-llama-${BUILD}-bin-ubuntu-x64.tar.gz}"
URL="${LLAMA_CPP_URL:-https://github.com/ggml-org/llama.cpp/releases/download/${BUILD}/${ASSET}}"
DEST="${LLAMA_BIN_DIR:-$ROOT/llama-bin}"

mkdir -p "$DEST"
if [ -x "$DEST/llama-$BUILD/llama-server" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[fetch_llama] already installed: $DEST/llama-$BUILD (set FORCE=1 to redownload)"
  exit 0
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[fetch_llama] downloading $URL"
curl -fL --retry 3 --connect-timeout 20 "$URL" -o "$TMP/llama.tar.gz"
tar -xzf "$TMP/llama.tar.gz" -C "$TMP"

SRC="$(find "$TMP" -maxdepth 2 -type f -name llama-server -printf '%h\n' | head -n 1)"
if [ -z "$SRC" ]; then
  echo "[fetch_llama] llama-server missing from archive" >&2
  exit 1
fi

rm -rf "$DEST/llama-$BUILD"
mv "$SRC" "$DEST/llama-$BUILD"
echo "[fetch_llama] installed $DEST/llama-$BUILD"
"$DEST/llama-$BUILD/llama-server" --version 2>&1 | head -n 2 || true
