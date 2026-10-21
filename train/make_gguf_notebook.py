#!/usr/bin/env python3
"""Writes train/AGBE_v2_4_gguf_cpu.ipynb: scores the three shipped-format GGUF files and times them on a 4 core CPU.

    python3 train/make_gguf_notebook.py
"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md("""
# AGBE v2: score and time the 4-bit GGUF files on a CPU

The three trained models were scored in float16 on a GPU. This notebook scores the **files that would actually ship** (Q4_K_M GGUF) on the same 140 held-out cases, with llama.cpp itself, and records speed and peak memory on a 4 core CPU. About 2 to 3 hours.

**Before you start:** Session options, **Accelerator: None** (this is a CPU run on purpose, a laptop has no GPU), **Internet: On**. Secrets `HF_WRITE_TOKEN` and `HF_TOKEN` attached. Then Save Version, **Save & Run All**, and close the browser. When it finishes, save the log (Download Logs) in your Downloads and tell me its name.
""")
code('''
import os, json, pathlib, subprocess
from kaggle_secrets import UserSecretsClient
_s = UserSecretsClient()
try: os.environ["HF_TOKEN"] = _s.get_secret("HF_WRITE_TOKEN")
except Exception: os.environ["HF_TOKEN"] = _s.get_secret("HF_TOKEN")
HF_REPO = "NEVODESIGN/agbe-v2-work"; DATA_RUN = "run1"; RUN = "gguf"
THREADS = 4          # the competition machine had 4 cores
LIMIT = 0            # 0 means all 140 held-out cases
MODELS = {"gemma1b": "google/gemma-3-1b-it", "qwen17": "Qwen/Qwen3-1.7B", "qwen4b": "Qwen/Qwen3-4B-Instruct-2507"}   # key: base model (for its chat template)
!git clone -q --depth 1 -b agbe2 https://github.com/nevodesigns/agbe.git /kaggle/working/agbe
%cd /kaggle/working/agbe
!pip -q install huggingface_hub transformers
!nproc
!lscpu | grep -i "model name"
!free -g | head -2
WORK = pathlib.Path("/kaggle/working/work"); WORK.mkdir(exist_ok=True)
TMP = pathlib.Path("/kaggle/temp"); TMP.mkdir(exist_ok=True)
''')
md("## 1. Build llama.cpp (the same source that made the GGUF files)")
code('''
LC = TMP / "llama.cpp"
if not (LC / "build/bin/llama-server").exists():
    !git clone -q --depth 1 https://github.com/ggml-org/llama.cpp {LC}
    !cmake -S {LC} -B {LC}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
    !cmake --build {LC}/build --target llama-server -j 4 2>&1 | tail -3
!ls -la {LC}/build/bin/llama-server
''')
md("## 2. Download the held-out cases and the three GGUF files from the private repository")
code('''
from huggingface_hub import hf_hub_download
def get(name, to):
    return hf_hub_download(HF_REPO, f"{DATA_RUN}/{name}", repo_type="dataset", local_dir=str(to))
EVAL = get("rag_eval.jsonl", TMP)
for key in MODELS:
    get(f"agbe-v2-{key}-q4_k_m.gguf", TMP)
    get(f"rageval_tuned_{key}.json", TMP)      # the float16 scores, for comparison
GG = TMP / DATA_RUN
!ls -la {GG}
''')
md("## 3. Score and time each file (one process per model, so memory is per model)")
code('''
for key, base in MODELS.items():
    out = WORK / f"rageval_gguf_{key}.json"
    if out.exists():
        continue
    cmd = f"python train/gguf_eval.py --server {LC}/build/bin/llama-server --gguf {GG}/agbe-v2-{key}-q4_k_m.gguf --tokenizer {base} --eval {EVAL} --out {out} --threads {THREADS}"
    if LIMIT: cmd += f" --limit {LIMIT}"
    print(cmd, flush=True)
    !{cmd}
''')
md("## 4. The table: float16 on the GPU against the shipped 4-bit file on the CPU")
code('''
rows = []
for key in MODELS:
    t = json.load(open(GG / f"rageval_tuned_{key}.json"))["summary"]
    g = json.load(open(WORK / f"rageval_gguf_{key}.json"))
    s, sp = g["summary"], g["speed"]
    f = lambda d: f"{d['grounded_pass']:.3f} / {d['supported_ratio']:.3f} / {d['valid_citation']:.3f} / {d['right_source']:.3f} / {d['wrongly_abstained']:.3f} / {d['abstain_pass']:.3f}"
    print(f"{key:8} float16 (pass/supported/cite/right/wrongabstain/abstain): {f(t)}")
    print(f"{key:8} Q4_K_M  (same measures)                                : {f(s)}")
    print(f"{'':8} file {sp['file_mb']} MB | peak memory {sp['peak_memory_mb']} MB | generation {sp['generation_tokens_per_s']} tokens/s | prompt {sp['prompt_tokens_per_s']} tokens/s | {sp['mean_seconds_per_answer']} s per answer | {sp['threads']} threads | {sp['cpu']}")
    print()
''')
code('''
!python train/hf_persist.py push {HF_REPO} {WORK} --path-in-repo {RUN}
''')
md("Done. The two numbers that matter for the product: **peak memory** (must fit the 8 GB laptop with room to spare) and **generation tokens per second**. Send the log.")

meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
(HERE / "AGBE_v2_4_gguf_cpu.ipynb").write_text(json.dumps({"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}, indent=1))
print("wrote AGBE_v2_4_gguf_cpu.ipynb")
