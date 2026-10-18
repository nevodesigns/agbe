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
print("merged model saved to", a.out)
