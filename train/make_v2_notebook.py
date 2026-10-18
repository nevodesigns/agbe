#!/usr/bin/env python3
"""Writes train/AGBE_v2_Kaggle.ipynb: one notebook that runs the whole v2 pipeline on a free Kaggle GPU.

    python3 train/make_v2_notebook.py
"""
import json, pathlib

OUT = pathlib.Path(__file__).resolve().parent / "AGBE_v2_Kaggle.ipynb"
cells = []


def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


md("""
# AGBE v2: the whole pipeline on a free Kaggle GPU

Run the cells from top to bottom. Each stage saves its result to a private Hugging Face dataset repo, so if Kaggle disconnects you
can start the notebook again and it picks up where it stopped (cells skip work that is already done).

**How to run it (three sessions, each one click):** the whole pipeline is too long for one free GPU session, so it has stage switches. In the first code cell set `STAGES` and press **Save Version, Save & Run All**. Kaggle then runs it in the background and you can close the browser.
- Session 1: `STAGES = ["A"]` (bake-off, about 2 to 3 hours). Read the result table.
- Session 2: `STAGES = ["B"]` (claims, questions, training data, about 3 to 4 hours).
- Session 3: `STAGES = ["C", "D", "E"]` (pick base, train, export, re-test, about 3 hours).
Each session starts by restoring the previous ones from your Hugging Face repo.

**Before you start (once):**
1. Right side, **Session options**: Accelerator **GPU T4 x2**, Internet **On**.
2. **Add-ons, Secrets**: a secret named `HF_WRITE_TOKEN` (or `HF_TOKEN`) with **write** access. If both exist, the write one is used.
3. On Hugging Face, open the pages of `google/gemma-3-1b-it`, `google/gemma-3-4b-it` and `meta-llama/Llama-3.2-1B-Instruct` and accept their terms (a model you did not accept is skipped, not fatal).

Stages: A bake-off of base models (with retrieved sources), B claim extraction + questions + training data, C training, D export to GGUF,
E check the new model against the same tests. Nothing here is a claim of quality until the numbers in stage E say so.
""")

md("## 0. Setup")
code('''
import os, json, pathlib, subprocess
from kaggle_secrets import UserSecretsClient
_s = UserSecretsClient()
try: os.environ["HF_TOKEN"] = _s.get_secret("HF_WRITE_TOKEN")   # your write token wins
except Exception: os.environ["HF_TOKEN"] = _s.get_secret("HF_TOKEN")
STAGES = ["A"]                         # session 1: ["A"], session 2: ["B"], session 3: ["C", "D", "E"]
HF_REPO = "NEVODESIGN/agbe-v2-work"   # private dataset repo that keeps your work between sessions (change if your token cannot write there)
RUN = "run1"
BAKEOFF_SAMPLE = 60                    # prompts per model in the bake-off; raise it later for a fuller test
EXTRACT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXTRACT_PASSAGES = 1000                # best passages to extract claims from; raise it if you have GPU hours left
PROFILE = "edge"                       # "edge" trains the best model of about 2B parameters or less, "server" the best model overall
''')
code('''
!git clone -q --depth 1 -b agbe2 https://github.com/nevodesigns/agbe.git /kaggle/working/agbe
%cd /kaggle/working/agbe
!pip -q install -U "transformers>=4.46" accelerate bitsandbytes peft trl datasets huggingface_hub
WORK = pathlib.Path("/kaggle/working/work"); WORK.mkdir(exist_ok=True)
PASSAGES = WORK / "passages.jsonl"
!gunzip -c data/fulltext_v1/passages.jsonl.gz > {PASSAGES}
!wc -l {PASSAGES}
!python train/hf_persist.py pull {HF_REPO} {WORK} --path-in-repo {RUN}
def save(): subprocess.run(["python", "train/hf_persist.py", "push", HF_REPO, str(WORK), "--path-in-repo", RUN])
''')

md("""
## A. Bake-off: which base model follows sources best?
Every candidate answers the same prompts with retrieved passages in front of it, and is scored by the competition scorer plus a check of how many of its sentences the sources do not support.
Small models run in full precision, 3B and 4B models in 4-bit. A model that cannot be loaded (terms not accepted) is skipped. About 2 to 3 hours in total.
""")
code('''
if "A" in STAGES and not (WORK / "bakeoff_small" / "summary.json").exists():
    !python train/bakeoff.py --modes rag --sample {BAKEOFF_SAMPLE} --max-new 160 --passages {PASSAGES} --out {WORK}/bakeoff_small --models google/gemma-3-1b-it Qwen/Qwen3-1.7B meta-llama/Llama-3.2-1B-Instruct
    save()
if "A" in STAGES and not (WORK / "bakeoff_big" / "summary.json").exists():
    !python train/bakeoff.py --modes rag --sample {BAKEOFF_SAMPLE} --max-new 160 --load-4bit --passages {PASSAGES} --out {WORK}/bakeoff_big --models google/gemma-3-4b-it Qwen/Qwen3-4B-Instruct-2507 HuggingFaceTB/SmolLM3-3B meta-llama/Llama-3.2-3B-Instruct
    save()
''')
code('''
if set(STAGES) & set("ACDE"):
    !python tools/pick_base.py {WORK}/bakeoff_*/summary.json --out {WORK}/chosen_base.json
    choice = json.load(open(WORK / "chosen_base.json"))
    BASE = (choice["best_small"] or choice["best"]) if PROFILE == "edge" else choice["best"]
    print("\\nTraining base for profile", PROFILE, "->", BASE)
    save()
''')

md("""
## B. Training data: verified claims, questions, conversations
A 7B model proposes facts from the passages. Code keeps only facts whose quote is verbatim in the passage and that are not chemical rates or study language.
Then the model writes farmer questions, and the builder composes grounded, abstain and refusal examples. Resumable. About 2 to 4 hours for 1,000 passages.
""")
code('''
if "B" in STAGES:
  !python tools/extract_claims.py --backend transformers --model {EXTRACT_MODEL} --load-4bit --batch 8 --prioritise --limit {EXTRACT_PASSAGES} --resume --passages {PASSAGES} --out {WORK}/claims.jsonl
  save()
  import collections
  print(collections.Counter(json.loads(l)["status"] for l in open(WORK / "claims.jsonl")))
''')
code('''
if "B" in STAGES:
  !python tools/build_grounded_sft.py questions --passages {PASSAGES} --claims {WORK}/claims.jsonl --out {WORK}/questions.jsonl --backend transformers --model {EXTRACT_MODEL} --load-4bit --batch 16 --resume
  !python tools/build_grounded_sft.py compose --passages {PASSAGES} --claims {WORK}/claims.jsonl --questions {WORK}/questions.jsonl --out {WORK}/sft.jsonl
  save()
  n = sum(1 for _ in open(WORK / "sft.jsonl")); print(n, "training examples")
  if n < 600: print("WARNING: fewer than 600 examples. Raise EXTRACT_PASSAGES and rerun stage B before training.")
''')

md("""
## C. Training
First a data check, then a 20-step smoke test, then the real run (QLoRA, one GPU). The trainer script was written without a GPU: if the smoke test fails, stop and send the error.
""")
code('''
if "C" in STAGES:
  !CUDA_VISIBLE_DEVICES=0 python train/train_grounded.py --data {WORK}/sft.jsonl --base {BASE} --out {WORK}/adapter_smoke --dry-data
  !CUDA_VISIBLE_DEVICES=0 python train/train_grounded.py --data {WORK}/sft.jsonl --base {BASE} --out {WORK}/adapter_smoke --max-steps 20
''')
code('''
if "C" in STAGES:
  !CUDA_VISIBLE_DEVICES=0 python train/train_grounded.py --data {WORK}/sft.jsonl --base {BASE} --out {WORK}/adapter --rank 32 --epochs 2
  save()
''')

md("## D. Merge and export to GGUF (Q4_K_M)")
code('''
if "D" in STAGES:
  !python train/merge_adapter.py --base {BASE} --adapter {WORK}/adapter --out {WORK}/merged
  !bash train/export_gguf.sh {WORK}/merged {WORK}/agbe-v2
  save()
''')

md("""
## E. Does the new model beat its own base?
Same prompts, same sources, same scorer. Compare the two rows. Only a better row, with zero leaks, is a result.
""")
code('''
if "E" in STAGES:
  !python train/bakeoff.py --modes rag --sample {BAKEOFF_SAMPLE} --max-new 160 --passages {PASSAGES} --out {WORK}/eval_v2 --models {WORK}/merged
  save()
for f in sorted(WORK.glob("*/summary.md")): print(f.parent.name); print(f.read_text())
''')
md("""
## Done: what to download
In the right panel, **Output**, under `work/`: `agbe-v2-q4_k_m.gguf` (the model), `adapter/` (the LoRA adapter), `claims.jsonl`, `sft.jsonl`, and every `summary.md`. Everything is also in your private Hugging Face repo.
Send the stage E table back, and the model is next measured on the real target (CPU speed and memory).
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT, len(cells), "cells")
