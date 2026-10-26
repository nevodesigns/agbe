#!/usr/bin/env bash
# Fetches the AGBE v16 model file (GGUF Q4_K_M) from Hugging Face.
# Safe to run again: if the file is already there it does nothing.
# No credentials are needed, and the URL is pinned to one commit so the
# file cannot change under a reader. Keep MODEL_FILE equal to
# _runtime.model_path in metadata.json.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"

MODEL_FILE="$MODEL_DIR/agbe-1b-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/NEVODESIGN/agbe-1b/resolve/545554ab173c363a03ed3672ace9693cec617d9d/agbe-1b-q4_k_m.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
  echo "model already present at $MODEL_FILE, nothing to download"
  exit 0
fi

echo "downloading $MODEL_URL to $MODEL_FILE"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --progress-bar -o "$MODEL_FILE.partial" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
  wget --show-progress -O "$MODEL_FILE.partial" "$MODEL_URL"
else
  echo "error: curl or wget is required" >&2
  exit 1
fi

mv "$MODEL_FILE.partial" "$MODEL_FILE"
echo "done: $MODEL_FILE"
