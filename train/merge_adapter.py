#!/usr/bin/env python3
"""Merge a LoRA adapter into its base and save a plain Hugging Face model, ready for GGUF conversion.
UNTESTED until run on a GPU session.

    python3 train/merge_adapter.py --base google/gemma-3-1b-it --adapter adapter-1b --out merged-1b
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
# the converter wants the original tokenizer files, and the older config key; newer transformers saves both differently
import json, os, shutil
from huggingface_hub import snapshot_download
snap = snapshot_download(a.base, allow_patterns=["tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*", "generation_config.json"])
for f in os.listdir(snap):
    shutil.copy2(os.path.join(snap, f), os.path.join(a.out, f))
cfgp = os.path.join(a.out, "config.json")
cfg = json.load(open(cfgp))
if "torch_dtype" not in cfg:
    cfg["torch_dtype"] = cfg.get("dtype", "float16")
    json.dump(cfg, open(cfgp, "w"), indent=2)
print("merged model saved to", a.out)
