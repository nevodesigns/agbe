#!/usr/bin/env python3
"""Writes CPU notebooks that score one trained model's 4-bit GGUF under the full prompt and the short prompt, with speed, memory and the guard.

    python3 train/make_eval_notebook.py
"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent


def build(fname, run, title, variants):
    cells = []
    def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
    def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})
    md(f"""
# AGBE v2: {title}

Scores the 4-bit GGUF saved in the folder `{run}` of the private repository on all 140 held-out cases, once per prompt setting, with llama.cpp itself on a 4 core CPU, and applies the guard. About {35 * len(variants)} to {50 * len(variants)} minutes.

**Before you start:** Session options, **Accelerator: None**, **Internet: On**, secrets attached. Save Version, **Save & Run All**. Then save the log (Download Logs) in your Downloads and tell me its name.
""")
    code(f"""
import os, json, pathlib, subprocess
from kaggle_secrets import UserSecretsClient
_s = UserSecretsClient()
try: os.environ["HF_TOKEN"] = _s.get_secret("HF_WRITE_TOKEN")
except Exception: os.environ["HF_TOKEN"] = _s.get_secret("HF_TOKEN")
HF_REPO = "NEVODESIGN/agbe-v2-work"; RUN_IN = "{run}"; RUN_OUT = "eval_{run}"
THREADS = 4; CTX = 2048
VARIANTS = {variants!r}   # name, passages kept (0 = all), words per passage (0 = whole)
!git clone -q --depth 1 -b agbe2 https://github.com/nevodesigns/agbe.git /kaggle/working/agbe
%cd /kaggle/working/agbe
!pip -q install huggingface_hub transformers
WORK = pathlib.Path("/kaggle/working/work"); WORK.mkdir(exist_ok=True)
TMP = pathlib.Path("/kaggle/temp"); TMP.mkdir(exist_ok=True)
from huggingface_hub import hf_hub_download
get = lambda name: hf_hub_download(HF_REPO, name, repo_type="dataset", local_dir=str(TMP))
EVAL = get("run1/rag_eval.jsonl")
GGUF = get(f"{{RUN_IN}}/agbe-v2-qwen17-q4_k_m.gguf")
LC = TMP / "llama.cpp"
if not (LC / "build/bin/llama-server").exists():
    !git clone -q --depth 1 https://github.com/ggml-org/llama.cpp {{LC}}
    !cmake -S {{LC}} -B {{LC}}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
    !cmake --build {{LC}}/build --target llama-server -j 4 2>&1 | tail -2
""")
    code("""
for name, k, w in VARIANTS:
    out = WORK / f"eval_{name}.json"
    if out.exists():
        continue
    cmd = f"python train/gguf_eval.py --server {LC}/build/bin/llama-server --gguf {GGUF} --tokenizer Qwen/Qwen3-1.7B --eval {EVAL} --out {out} --threads {THREADS} --ctx {CTX} --sources {k} --max-words {w}"
    print(cmd, flush=True)
    !{cmd}
    !python train/guard_eval.py --eval {EVAL} --rows {out} --out {WORK}/guard_{name}.json --sources {k} --max-words {w}
""")
    code("""
def line(tag, summ, sp, g):
    print(f"{tag:26} pass {summ['grounded_pass']:.3f} | supported {summ['supported_ratio']:.3f} | cite {summ['valid_citation']:.3f} | right source {summ['right_source']:.3f} | wrong abstain {summ['wrongly_abstained']:.3f} | clean abstain {summ.get('abstain_clean_pass')} | {sp['mean_prompt_tokens']} tokens | {sp['mean_seconds_per_answer']} s | memory {sp['peak_memory_mb']} MB | WITH GUARD pass {g['grounded_pass']:.3f}, clean abstain {g.get('abstain_clean_pass')}")
for name, k, w in VARIANTS:
    r = json.load(open(WORK / f"eval_{name}.json")); g = json.load(open(WORK / f"guard_{name}.json"))
    line(name, r["summary"], r["speed"], g["with_guard"])
""")
    code("""
!python train/hf_persist.py push {HF_REPO} {WORK} --path-in-repo {RUN_OUT}
""")
    md("Done. Save the log.")
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    (HERE / fname).write_text(json.dumps({"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}, indent=1))
    print("wrote", fname)


build("AGBE_v2_7_eval_run2.ipynb", "run2", "score the retrained Qwen3 1.7B (clean abstain data) as a 4-bit file, full prompt and short prompt", [["full", 0, 0], ["k3w100", 3, 100]])
build("AGBE_v2_8_eval_run3.ipynb", "run3", "score the Qwen3 1.7B trained on the short prompt, as a 4-bit file, short prompt and full prompt", [["k3w100", 3, 100], ["full", 0, 0]])
