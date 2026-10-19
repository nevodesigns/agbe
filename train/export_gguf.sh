#!/usr/bin/env bash
# Convert a merged Hugging Face model to GGUF and quantise to Q4_K_M with llama.cpp. UNTESTED until run in a session.
# GGUF conversion depends on transformers 4.57.x (see BUILDS.md): install it before running.
#   bash train/export_gguf.sh merged-1b agbe-v2-1b
set -euo pipefail
MERGED="$1"; NAME="$2"; LC="${LLAMA_CPP_DIR:-llama.cpp}"
[ -d "$LC" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LC"
# a separate virtualenv: llama.cpp's converter pins a CPU torch and an older transformers, and must not replace the GPU stack
VENV="${GGUF_VENV:-gguf-venv}"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" 2>/dev/null || { rm -rf "$VENV"; python3 -m pip -q install virtualenv && python3 -m virtualenv -q "$VENV"; }  # Kaggle's python has no ensurepip
fi
"$VENV/bin/pip" -q install -r "$LC/requirements/requirements-convert_hf_to_gguf.txt" "transformers==4.57.*"
[ -x "$LC/build/bin/llama-quantize" ] || (cmake -S "$LC" -B "$LC/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null && cmake --build "$LC/build" --target llama-quantize -j 4)
"$VENV/bin/python" "$LC/convert_hf_to_gguf.py" "$MERGED" --outfile "$NAME-f16.gguf" --outtype f16
"$LC/build/bin/llama-quantize" "$NAME-f16.gguf" "$NAME-q4_k_m.gguf" Q4_K_M
sha256sum "$NAME-q4_k_m.gguf" | tee "$NAME-q4_k_m.gguf.sha256"
ls -la "$NAME"-*.gguf
