#!/usr/bin/env bash
# Download the AGBE model weight file.
#
# Rules:
#   - Must be idempotent (safe to run multiple times).
#   - Must download without any credentials (public URL only).
#   - The output path must match `_runtime.model_path` in metadata.json.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/agbe-1b-q4_k_m.gguf"

MODEL_URL="https://huggingface.co/NEVODESIGN/agbe-1b/resolve/main/agbe-1b-q4_k_m.gguf"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_FILE" ]]; then
  echo "model already present at $MODEL_FILE — skipping download"
  exit 0
fi

echo "downloading $MODEL_URL → $MODEL_FILE (~0.8 GB)…"

# Resume rather than restart. The target user is on a rural connection where an
# 800 MB transfer routinely drops part way, and starting from zero each time can
# mean never finishing. -C - continues from whatever is already on disk, and we
# retry a few times before giving up.
if command -v curl > /dev/null 2>&1; then
  curl -L --fail --progress-bar -C - --retry 5 --retry-delay 5 \
       -o "$MODEL_FILE.partial" "$MODEL_URL" || true
elif command -v wget > /dev/null 2>&1; then
  wget --continue --tries=5 --show-progress -O "$MODEL_FILE.partial" "$MODEL_URL" || true
else
  echo "error: neither curl nor wget found" >&2
  exit 1
fi

# Verify before committing the name. A truncated GGUF keeps a valid magic number
# at byte 0, so a header check passes on a broken file; only the length catches it.
# We hit exactly this during development: curl exited 0 on three partial models
# and llama.cpp reported only "failed to load model".
ACTUAL=$(stat -c%s "$MODEL_FILE.partial" 2>/dev/null || stat -f%z "$MODEL_FILE.partial" 2>/dev/null || echo 0)
if [[ "$ACTUAL" -lt 800000000 ]]; then
  echo "error: got $ACTUAL bytes, expected ~814 MB. The transfer was cut short." >&2
  echo "The partial file is kept at $MODEL_FILE.partial — re-run this script and" >&2
  echo "it will resume from where it stopped." >&2
  exit 1
fi

mv "$MODEL_FILE.partial" "$MODEL_FILE"
echo "done: $MODEL_FILE ($ACTUAL bytes)"
