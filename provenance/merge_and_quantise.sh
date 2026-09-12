#!/usr/bin/env bash
# How the LoRA adapter becomes the shipped GGUF.
#
# Gate 2 section 3.1 asks for "the merge/quantization script showing how the
# adapter was merged into the base weights and converted to GGUF". This is that
# path, as three commands, in the order the Kaggle notebook runs them.
#
# The merge itself is peft's merge_and_unload(), called inside
# train/train_lora.py under --merge rather than in a separate script, because
# the adapter has to be merged into a base model that is already loaded in the
# same dtype it was trained against. Running it separately reloads the base in
# whatever dtype the second script happens to choose, and a merge across dtypes
# is a silent quality change. Step 1 below is the exact call.

set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-/kaggle/working/llama.cpp}"
OUT="${OUT:-/kaggle/working/out}"
WORK="${WORK:-/kaggle/working}"

# 1. Train, then merge the adapter into the base weights.
#
#    train_lora.py --merge does, in this order:
#      base   = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=float16)
#      merged = PeftModel.from_pretrained(base, out/adapter).merge_and_unload()
#      merged.save_pretrained(out/merged, safe_serialization=True)
#
#    It also writes out/provenance/{training_log.json,training_log.csv,
#    run_manifest.json,checksums.json} before the merge, so a merge failure does
#    not take the evidence of the training run with it.
python train/train_lora.py \
    --train corpus/build/train.jsonl \
    --out   "$OUT" \
    --merge

# 2. Convert the merged HF model to GGUF at f16.
#
#    Pinned llama.cpp, and transformers must be on 4.57.x here. Under 5.0.0
#    GemmaTokenizer injects <image_soft_token> at id 262144 while the text-only
#    1B declares vocab_size 262144, and conversion dies on
#    `assert max(tokenizer.vocab.values()) < vocab_size` AFTER writing all 340
#    tensors, leaving no file behind. BUILDS.md records the build lost to this.
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$OUT/merged" \
    --outfile "$WORK/agbe-f16.gguf" \
    --outtype f16

# 3. Quantise to Q4_K_M, the preset every measurement in REPORT.md was taken on.
"$LLAMA_CPP/build/bin/llama-quantize" \
    "$WORK/agbe-f16.gguf" \
    "$WORK/agbe-1b-q4_k_m.gguf" \
    Q4_K_M

sha256sum "$WORK/agbe-1b-q4_k_m.gguf"
ls -l "$WORK/agbe-1b-q4_k_m.gguf"
