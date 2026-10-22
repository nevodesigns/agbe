#!/usr/bin/env python3
"""Writes train/AGBE_v2_5_speed_memory.ipynb: speed and peak memory of the shipped-format GGUF files on a 4 core CPU,
with the v16 competition file as a same-machine control and llama.cpp's prompt caches switched off.

    python3 train/make_speed_notebook.py
"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md("""
# AGBE v2: speed and memory of the 4-bit files on a CPU

The first CPU run showed the 4-bit files keep their quality, but its memory figures (10 to 13 GB for files of 1 to 2.5 GB) were inflated by llama.cpp's own prompt cache. This run switches those caches off, uses the competition's 2,048-token context, and also times the **v16 competition file on the same machine**, so the speeds can be compared fairly. About 30 to 45 minutes.

**Before you start:** Session options, **Accelerator: None**, **Internet: On**, secrets attached. Save Version, **Save & Run All**. Then save the log (Download Logs) in your Downloads and tell me its name.
""")
code("""
import os, json, pathlib, subprocess
from kaggle_secrets import UserSecretsClient
_s = UserSecretsClient()
try: os.environ["HF_TOKEN"] = _s.get_secret("HF_WRITE_TOKEN")
except Exception: os.environ["HF_TOKEN"] = _s.get_secret("HF_TOKEN")
HF_REPO = "NEVODESIGN/agbe-v2-work"; DATA_RUN = "run1"; RUN = "speed"
THREADS = 4
LIMIT = 30          # speed and memory need only a few cases; quality was scored on all 140 already
CTX = 2048          # the context the competition used
!git clone -q --depth 1 -b agbe2 https://github.com/nevodesigns/agbe.git /kaggle/working/agbe
%cd /kaggle/working/agbe
!pip -q install huggingface_hub transformers
!nproc
!lscpu | grep -i "model name"
!free -g | head -2
WORK = pathlib.Path("/kaggle/working/work"); WORK.mkdir(exist_ok=True)
TMP = pathlib.Path("/kaggle/temp"); TMP.mkdir(exist_ok=True)
""")
code("""
LC = TMP / "llama.cpp"
if not (LC / "build/bin/llama-server").exists():
    !git clone -q --depth 1 https://github.com/ggml-org/llama.cpp {LC}
    !cmake -S {LC} -B {LC}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
    !cmake --build {LC}/build --target llama-server -j 4 2>&1 | tail -2
!{LC}/build/bin/llama-server --help 2>&1 | grep -E "cache-ram|ctx-checkpoints" | head -4
""")
code("""
from huggingface_hub import hf_hub_download
EVAL = hf_hub_download(HF_REPO, f"{DATA_RUN}/rag_eval.jsonl", repo_type="dataset", local_dir=str(TMP))
MODELS = {   # key: (gguf file, base model for the chat template)
    "v16":     (hf_hub_download("NEVODESIGN/agbe-1b", "agbe-1b-q4_k_m.gguf", local_dir=str(TMP / "v16")), "google/gemma-3-1b-it"),
    "gemma1b": (hf_hub_download(HF_REPO, f"{DATA_RUN}/agbe-v2-gemma1b-q4_k_m.gguf", repo_type="dataset", local_dir=str(TMP)), "google/gemma-3-1b-it"),
    "qwen17":  (hf_hub_download(HF_REPO, f"{DATA_RUN}/agbe-v2-qwen17-q4_k_m.gguf", repo_type="dataset", local_dir=str(TMP)), "Qwen/Qwen3-1.7B"),
    "qwen4b":  (hf_hub_download(HF_REPO, f"{DATA_RUN}/agbe-v2-qwen4b-q4_k_m.gguf", repo_type="dataset", local_dir=str(TMP)), "Qwen/Qwen3-4B-Instruct-2507"),
}
for k, (f, b) in MODELS.items(): print(k, round(os.path.getsize(f) / 1e6), "MB", f)
""")
code("""
for key, (gguf, base) in MODELS.items():
    out = WORK / f"speed_{key}.json"
    if out.exists():
        continue
    cmd = f"python train/gguf_eval.py --server {LC}/build/bin/llama-server --gguf {gguf} --tokenizer {base} --eval {EVAL} --out {out} --threads {THREADS} --ctx {CTX} --limit {LIMIT}"
    print(cmd, flush=True)
    !{cmd}
""")
code("""
base = None
for key in MODELS:
    sp = json.load(open(WORK / f"speed_{key}.json"))["speed"]
    if key == "v16": base = sp["generation_tokens_per_s"]
    rel = f" ({sp['generation_tokens_per_s'] / base:.2f}x v16)" if base and key != "v16" else ""
    print(f"{key:8} file {sp['file_mb']:>5} MB | peak memory {sp['peak_memory_mb']} MB | generation {sp['generation_tokens_per_s']} tokens/s{rel} | prompt {sp['prompt_tokens_per_s']} tokens/s | {sp['mean_seconds_per_answer']} s per answer | {sp['mean_prompt_tokens']} prompt tokens | {sp['threads']} threads")
print(sp["cpu"])
""")
code("""
!python train/hf_persist.py push {HF_REPO} {WORK} --path-in-repo {RUN}
""")
md("Done. Save the log. The competition measured v16 at 24.08 tokens per second on a different machine; the v16 line above is the same file on this machine, which is what the other lines should be compared with.")

meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
(HERE / "AGBE_v2_5_speed_memory.ipynb").write_text(json.dumps({"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}, indent=1))
print("wrote AGBE_v2_5_speed_memory.ipynb")
