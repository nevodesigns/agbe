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


import json as _json, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROBES = []
try:
    from serve.guard import build_messages
    ev = _pl.Path(a.adapter).parent / "rag_eval.jsonl"
    recs = [_json.loads(l) for l in open(ev) if l.strip()]
    PROBES = [build_messages(r["question"], r["sources"]) for r in recs if r["kind"] == "grounded"][:3]
except Exception as e:
    print("real-prompt probes unavailable:", repr(e))


def probe(m, tag):
    for i, msgs in enumerate(PROBES):
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=70, do_sample=False)
        print(f"PROBE{i} {tag}:", tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")[:260])


try:
    with pm.disable_adapter():
        probe(pm, "base   ")
    probe(pm, "adapter")
except Exception as e:
    print("probe failed:", repr(e))
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
try:
    probe(model, "merged ")
except Exception as e:
    print("probe failed:", repr(e))
if delta == 0.0:
    sys.exit("the merge changed nothing: the adapter was not applied")
model = model.to(torch.float16)
for attr in ("dtype", "torch_dtype"):  # newer transformers saves weights in the dtype named in the config
    try:
        setattr(model.config, attr, torch.float16)
    except Exception:
        pass
model.save_pretrained(a.out, safe_serialization=True)
# the evaluation and the GGUF both start from the saved file, so prove the file still behaves like the merged model
try:
    del model, pm
    try:
        reloaded = AutoModelForCausalLM.from_pretrained(a.out, dtype=torch.float16, device_map="cpu")
    except TypeError:
        reloaded = AutoModelForCausalLM.from_pretrained(a.out, torch_dtype=torch.float16, device_map="cpu")
    s_saved = sample(reloaded)
    print("SANITY saved   :", s_saved)
    print("SAVED FILE KEEPS THE TRAINING:", s_saved != s_base and s_saved == s_merged, "(same as merged:", s_saved == s_merged, "| differs from base:", s_saved != s_base, ")")
except Exception as e:
    print("saved-file check failed:", repr(e))
tok.save_pretrained(a.out)
# the converter wants the original tokenizer files, and the older config key; newer transformers saves both differently
import json, os, shutil
from huggingface_hub import snapshot_download
import fnmatch
KEEP = ["tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*", "generation_config.json"]
snap = snapshot_download(a.base, allow_patterns=KEEP)
for f in os.listdir(snap):  # only the tokenizer files: the snapshot folder also holds the BASE weights and config, which must never overwrite the merge
    if any(fnmatch.fnmatch(f, pat) for pat in KEEP):
        shutil.copy2(os.path.join(snap, f), os.path.join(a.out, f))
cfgp = os.path.join(a.out, "config.json")
cfg = json.load(open(cfgp))
if "torch_dtype" not in cfg:
    cfg["torch_dtype"] = cfg.get("dtype", "float16")
    json.dump(cfg, open(cfgp, "w"), indent=2)
print("merged model saved to", a.out)
try:  # final proof on the finished folder, after every file copy
    try:
        final = AutoModelForCausalLM.from_pretrained(a.out, dtype=torch.float16, device_map="cpu")
    except TypeError:
        final = AutoModelForCausalLM.from_pretrained(a.out, torch_dtype=torch.float16, device_map="cpu")
    tok_final = AutoTokenizer.from_pretrained(a.out)
    enc = tok_final.apply_chat_template(PROBES[0] if PROBES else MSG, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = final.generate(**enc, max_new_tokens=60, do_sample=False)
    txt = tok_final.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
    print("FINAL FOLDER ANSWER:", txt[:240])
    print("FINAL FOLDER IS THE TUNED MODEL:", "[1]" in txt or "[2]" in txt or "[3]" in txt)
except Exception as e:
    print("final folder check failed:", repr(e))
