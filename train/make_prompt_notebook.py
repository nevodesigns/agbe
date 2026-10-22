#!/usr/bin/env python3
"""Writes train/AGBE_v2_6_prompt_and_guard.ipynb: does a shorter prompt keep the quality, and how good is the model behind the guard?

    python3 train/make_prompt_notebook.py
"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md("""
# AGBE v2: a shorter prompt, and the model behind the guard

Two questions, both on the CPU, with the Qwen3 1.7B file that is the edge candidate:

1. **Shorter prompt.** Most of an answer's time is the model reading about 750 tokens of passages. Does keeping the best two passages and trimming each to the sentences that match the question keep the quality? It runs three settings on all 140 held-out cases and records speed and quality together.
2. **The guard.** The product is the model behind the guard. This applies the guard's own sentence check to the answers the three models already gave, and scores the result.

About 2 hours. **Before you start:** Session options, **Accelerator: None**, **Internet: On**, secrets attached. Save Version, **Save & Run All**. Then save the log (Download Logs) in your Downloads and tell me its name.
""")
code("""
import os, json, pathlib, subprocess
from kaggle_secrets import UserSecretsClient
_s = UserSecretsClient()
try: os.environ["HF_TOKEN"] = _s.get_secret("HF_WRITE_TOKEN")
except Exception: os.environ["HF_TOKEN"] = _s.get_secret("HF_TOKEN")
HF_REPO = "NEVODESIGN/agbe-v2-work"; DATA_RUN = "run1"; RUN = "prompt"
THREADS = 4; CTX = 2048
VARIANTS = [("k3w100", 3, 100), ("k2w80", 2, 80), ("k2w50", 2, 50)]   # name, passages kept, words per passage
!git clone -q --depth 1 -b agbe2 https://github.com/nevodesigns/agbe.git /kaggle/working/agbe
%cd /kaggle/working/agbe
!pip -q install huggingface_hub transformers
!nproc
WORK = pathlib.Path("/kaggle/working/work"); WORK.mkdir(exist_ok=True)
TMP = pathlib.Path("/kaggle/temp"); TMP.mkdir(exist_ok=True)
from huggingface_hub import hf_hub_download
get = lambda name: hf_hub_download(HF_REPO, name, repo_type="dataset", local_dir=str(TMP))
EVAL = get(f"{DATA_RUN}/rag_eval.jsonl")
ROWS = {k: get(f"gguf/rageval_gguf_{k}.json") for k in ("gemma1b", "qwen17", "qwen4b")}     # answers the three 4-bit files already gave
GGUF = get(f"{DATA_RUN}/agbe-v2-qwen17-q4_k_m.gguf")
""")
md("## 1. The model behind the guard (no model run needed)")
code("""
for key, path in ROWS.items():
    print("=" * 100)
    !python train/guard_eval.py --eval {EVAL} --rows {path} --out {WORK}/guard_{key}.json
""")
md("## 2. A shorter prompt")
code("""
LC = TMP / "llama.cpp"
if not (LC / "build/bin/llama-server").exists():
    !git clone -q --depth 1 https://github.com/ggml-org/llama.cpp {LC}
    !cmake -S {LC} -B {LC}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
    !cmake --build {LC}/build --target llama-server -j 4 2>&1 | tail -2
""")
code("""
for name, k, w in VARIANTS:
    out = WORK / f"prompt_{name}.json"
    if out.exists():
        continue
    cmd = f"python train/gguf_eval.py --server {LC}/build/bin/llama-server --gguf {GGUF} --tokenizer Qwen/Qwen3-1.7B --eval {EVAL} --out {out} --threads {THREADS} --ctx {CTX} --sources {k} --max-words {w}"
    print(cmd, flush=True)
    !{cmd}
    !python train/guard_eval.py --eval {EVAL} --rows {out} --out {WORK}/guard_prompt_{name}.json --sources {k} --max-words {w}
""")
md("## 3. The table")
code("""
def line(tag, summ, sp=None, g=None):
    s = f"{tag:30} pass {summ['grounded_pass']:.3f} | supported {summ['supported_ratio']:.3f} | cite {summ['valid_citation']:.3f} | right source {summ['right_source']:.3f} | wrong abstain {summ['wrongly_abstained']:.3f} | abstain {summ['abstain_pass']:.3f} (clean {summ.get('abstain_clean_pass')})"
    if sp: s += f" | {sp['mean_prompt_tokens']} prompt tokens | {sp['mean_seconds_per_answer']} s per answer"
    if g: s += f" | WITH GUARD pass {g['grounded_pass']:.3f}, abstain {g['abstain_pass']:.3f}, supported {g['supported_ratio']:.3f}"
    print(s)
full = json.load(open(ROWS["qwen17"])); gfull = json.load(open(WORK / "guard_qwen17.json"))
line("full prompt (3 passages, whole)", full["summary"], full["speed"], gfull["with_guard"])
for name, k, w in VARIANTS:
    r = json.load(open(WORK / f"prompt_{name}.json")); g = json.load(open(WORK / f"guard_prompt_{name}.json"))
    line(f"{k} passages, {w} words each", r["summary"], r["speed"], g["with_guard"])
""")
code("""
!python train/hf_persist.py push {HF_REPO} {WORK} --path-in-repo {RUN}
""")
md("Done. Save the log. The question is whether a setting keeps the pass rate close to the full prompt while cutting the seconds per answer.")

meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
(HERE / "AGBE_v2_6_prompt_and_guard.ipynb").write_text(json.dumps({"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}, indent=1))
print("wrote AGBE_v2_6_prompt_and_guard.ipynb")
