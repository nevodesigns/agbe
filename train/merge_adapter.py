#!/usr/bin/env python3
"""Merge a LoRA adapter into its base and save a plain Hugging Face model, ready for GGUF conversion.
UNTESTED until run on a GPU session.

    python3 train/merge_adapter.py --base google/gemma-3-1b-it --adapter adapter-1b --out merged-1b
"""
import argparse, importlib.metadata as md, subprocess, sys

# Kaggle ships an old torchao that peft refuses to import next to; it is not needed here, so remove it
try:
    if tuple(int(x) for x in md.version("torchao").split(".")[:2]) < (0, 16):
        subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchao"], check=False)
except md.PackageNotFoundError:
    pass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.base)
try:  # float32 so the small LoRA changes are not rounded away when added; newer transformers calls the argument dtype
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32, device_map="cpu")
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32, device_map="cpu")
probe_name = next(n for n, _ in model.named_parameters() if "layers.0.self_attn.q_proj.weight" in n)
before = dict(model.named_parameters())[probe_name].detach().clone()
pm = PeftModel.from_pretrained(model, a.adapter)

SOURCE = "[1] (Maize guide, p.3) Plant maize at the onset of rains. Space rows 75 cm apart and plants 25 cm apart within the row. Thin to one plant per stand two weeks after planting."
MSG = [{"role": "system", "content": "You are AGBE, an agricultural advisor. Use only the numbered sources provided. Cite them like [1].\n\nSources:\n" + SOURCE},
       {"role": "user", "content": "How far apart should I plant maize?"}]


def sample(m):
    enc = tok.apply_chat_template(MSG, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=60, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


try:  # three-way sanity check: base, adapter applied but not merged, merged
    with pm.disable_adapter():
        s_base = sample(pm)
    s_adapter = sample(pm)
except Exception as e:
    s_base = s_adapter = "sanity check failed: %r" % (e,)
model = pm.merge_and_unload()
after = dict(model.named_parameters())[probe_name].detach()
delta = float((after - before).abs().max())
print("MERGE CHECK: largest change in", probe_name, "=", delta)
try:
    s_merged = sample(model)
except Exception as e:
    s_merged = "sanity check failed: %r" % (e,)
print("SANITY base    :", s_base)
print("SANITY adapter :", s_adapter)
print("SANITY merged  :", s_merged)
if delta == 0.0:
    sys.exit("the merge changed nothing: the adapter was not applied")
model = model.to(torch.float16)
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
