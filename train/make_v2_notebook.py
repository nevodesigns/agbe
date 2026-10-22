#!/usr/bin/env python3
"""Writes train/AGBE_v2_Kaggle.ipynb: one notebook that runs the whole v2 pipeline on a free Kaggle GPU.

    python3 train/make_v2_notebook.py
"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
cells = []


def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


md("""
# AGBE v2: the whole pipeline on a free Kaggle GPU

Run the cells from top to bottom. Each stage saves its result to a private Hugging Face dataset repo, so if Kaggle disconnects you
can start the notebook again and it picks up where it stopped (cells skip work that is already done).

**This is session @@N@@: @@WHAT@@.** Nothing to edit. Press **Save Version, then Save & Run All**: Kaggle runs it in the background and you can close the browser. When it finishes, open the next notebook (the three share their work through your Hugging Face repo, so each one starts by restoring the earlier ones).

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
STAGES = @@STAGES@@                    # already set for this notebook, nothing to edit
HF_REPO = "NEVODESIGN/agbe-v2-work"   # private dataset repo that keeps your work between sessions
DATA_RUN = "run1"                      # the folder this notebook starts from
FRESH_EVAL = @@FRESH@@                  # False when retraining against the held-out cases that already exist
RUN = "@@RUN@@"                        # the folder this notebook saves to (the smoke test uses its own, so it cannot disturb run1)
BASE_KEYS = @@KEYS@@                   # which bases to train; None means all in train/session_config.json
EPOCHS_OVERRIDE = @@EPOCHS@@           # None means the epochs in train/session_config.json
PROFILE = "edge"
''')
code('''
!git clone -q --depth 1 -b agbe2 https://github.com/nevodesigns/agbe.git /kaggle/working/agbe
%cd /kaggle/working/agbe
!pip -q install -U "transformers>=4.46" accelerate bitsandbytes peft trl datasets huggingface_hub
CFG = json.load(open("train/session_config.json"))   # every tunable number lives in this file, in the repo
BAKEOFF_SAMPLE, EXTRACT_MODEL, EXTRACT_PASSAGES = CFG["bakeoff_sample"], CFG["extract_model"], CFG["extract_passages"]
EPOCHS = EPOCHS_OVERRIDE or CFG["epochs"]
BASES = {k: tuple(v) for k, v in CFG["bases"].items() if BASE_KEYS is None or k in BASE_KEYS}
print("config:", CFG, "| bases this run:", list(BASES), "| epochs:", EPOCHS)
WORK = pathlib.Path("/kaggle/working/work"); WORK.mkdir(exist_ok=True)
PASSAGES = WORK / "passages.jsonl"
!gunzip -c data/fulltext_v1/passages.jsonl.gz > {PASSAGES}
!wc -l {PASSAGES}
!python train/hf_persist.py pull {HF_REPO} {WORK} --path-in-repo {DATA_RUN} @@IGNORE@@
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
if "A" in STAGES:
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
  !python tools/extract_claims.py --backend transformers --model {EXTRACT_MODEL} --load-4bit --batch 8 --prioritise --max-study-hits 2 --min-signal 2 --limit {EXTRACT_PASSAGES} --resume --passages {PASSAGES} --out {WORK}/claims.jsonl
  save()
  import collections
  print(collections.Counter(json.loads(l)["status"] for l in open(WORK / "claims.jsonl")))
''')
code('''
if "B" in STAGES:
  !python tools/build_grounded_sft.py questions --passages {PASSAGES} --claims {WORK}/claims.jsonl --out {WORK}/questions.jsonl --backend transformers --model {EXTRACT_MODEL} --load-4bit --batch 16 --resume
  !python tools/build_grounded_sft.py compose --passages {PASSAGES} --claims {WORK}/claims.jsonl --questions {WORK}/questions.jsonl --out {WORK}/sft.jsonl --variants {CFG["variants"]}
  save()
  n = sum(1 for _ in open(WORK / "sft.jsonl")); print(n, "training examples")
  if n < 600: print("WARNING: fewer than 600 examples. Raise extract_passages in train/session_config.json and rerun stage B before training.")
  !python train/diagnose_data.py --claims {WORK}/claims.jsonl --sft {WORK}/sft.jsonl
''')

md("""
## C. Held-out test, baselines, training
Three candidates are trained on the same data: **gemma-3-1b-it** (smallest, the continuity choice), **Qwen3-1.7B** (what the bake-off picked for the edge profile) and **Qwen3-4B-Instruct** (best in the bake-off, zero leaks, the server profile).
The bake-off showed the small models are too close to call on a 60-prompt test, so the real choice is made after training, on passages the models never saw.
First the training set is rebuilt with 10% of passages held out for testing, then every base is scored on the held-out test, then each is trained (a 20-step smoke test first).
""")
code('''
if "C" in STAGES:
  FLAGS = f'--variants {CFG["variants"]} --holdout {CFG["holdout"]}'
  if FRESH_EVAL: FLAGS += f' --eval-out {WORK}/rag_eval.jsonl'                                             # first time: write the held-out test cases
  else: FLAGS += f' --abstain-share {CFG["abstain_share"]} --hard-negatives'                               # retraining: keep the SAME held-out cases, harder abstain examples
  !python tools/build_grounded_sft.py compose --passages {PASSAGES} --claims {WORK}/claims.jsonl --questions {WORK}/questions.jsonl --out {WORK}/sft.jsonl {FLAGS}
  save()
  print(sum(1 for _ in open(WORK / "sft.jsonl")), "training examples;", sum(1 for _ in open(WORK / "rag_eval.jsonl")), "held-out test cases")
  !python train/diagnose_data.py --sft {WORK}/sft.jsonl
''')
code('''
if "C" in STAGES:
  for key, (base, lr) in BASES.items():
    if not (WORK / f"rageval_base_{key}.json").exists():
      !python train/rag_eval.py --model {base} --eval {WORK}/rag_eval.jsonl --out {WORK}/rageval_base_{key}.json
  save()
''')
code('''
if "C" in STAGES:
  !CUDA_VISIBLE_DEVICES=0 python train/train_grounded.py --data {WORK}/sft.jsonl --base google/gemma-3-1b-it --out {WORK}/adapter_smoke --dry-data
  for i, (key, (base, lr)) in enumerate(BASES.items()):
    if (WORK / f"adapter_{key}" / "adapter_config.json").exists():
      continue
    if i == 0:
      !CUDA_VISIBLE_DEVICES=0 python train/train_grounded.py --data {WORK}/sft.jsonl --base {base} --out {WORK}/adapter_smoke --max-steps 20
    !CUDA_VISIBLE_DEVICES=0 python train/train_grounded.py --data {WORK}/sft.jsonl --base {base} --out {WORK}/adapter_{key} --rank 32 --epochs {EPOCHS} --lr {lr}
    save()
''')

md("""
## D and E. Merge, export to GGUF, and test each trained model
For each model: merge the adapter, convert to GGUF (Q4_K_M), score it on the held-out test, then delete the big temporary files (Kaggle's disk is limited). The table at the end puts every base next to its trained version.
""")
code('''
TMP = "/kaggle/temp"; os.makedirs(TMP, exist_ok=True)   # big temporary files go here, outside the 20 GB output limit
if "D" in STAGES:
  for key, (base, lr) in BASES.items():
    if (WORK / f"agbe-v2-{key}-q4_k_m.gguf").exists():
      continue
    !python train/merge_adapter.py --base {base} --adapter {WORK}/adapter_{key} --out {TMP}/merged_{key}
    !bash train/export_gguf.sh {TMP}/merged_{key} {TMP}/agbe-v2-{key}
    !cp {TMP}/agbe-v2-{key}-q4_k_m.gguf {TMP}/agbe-v2-{key}-q4_k_m.gguf.sha256 {WORK}/
    !python train/rag_eval.py --model {TMP}/merged_{key} --eval {WORK}/rag_eval.jsonl --out {WORK}/rageval_tuned_{key}.json
    !rm -rf {TMP}/merged_{key} {TMP}/agbe-v2-{key}-*.gguf
    save()
''')
code('''
if "E" in STAGES or "D" in STAGES:
  !python train/rag_eval.py --table {WORK}/rageval_base_*.json {WORK}/rageval_tuned_*.json
for f in sorted(WORK.glob("bakeoff_*/summary.md")): print(f.parent.name); print(f.read_text())
''')

md("""
## Done: what to download
In the right panel, **Output**, under `work/`: `agbe-v2-<name>-q4_k_m.gguf` for each trained model, the `adapter_*` folders, `claims.jsonl`, `sft.jsonl`, `rag_eval.jsonl` and the `rageval_*.json` results. Everything is also in your private Hugging Face repo.
Send the table printed at the end, and the models are next measured on the real target (CPU speed and memory).
""")

SESSIONS = [  # (label, stages, what, file, run folder, base keys, epochs)
    ("1", '["A"]', "the bake-off of base models, about 2 to 3 hours", "AGBE_v2_1_bakeoff.ipynb", "run1", "None", "None"),
    ("2", '["B"]', "claims, questions and training data, about 3 to 4 hours (already done once; this is the first pass)", "AGBE_v2_2_data.ipynb", "run1", "None", "None"),
    ("2b", '["B"]', "MORE training data: it resumes from the passages already done and adds about 1,150 more, about 3 to 4 hours (the first run took 7.7 s per passage)", "AGBE_v2_2b_more_data.ipynb", "run1", "None", "None"),
    ("3s", '["C", "D", "E"]', "SMOKE TEST of training, export and testing on the data so far, one base and one epoch, about 1 to 2 hours. Its results go to a separate folder and are not the final models", "AGBE_v2_3s_smoketest.ipynb", "smoke", '["gemma1b"]', "1"),
    ("3a", '["C", "D", "E"]', "RETRAIN Qwen3 1.7B with harder abstain examples (same-document passages that look relevant but do not answer, 30% abstain share), scored on the SAME held-out cases as before, about 1.5 hours. Saves to folder run2, so run1 is untouched", "AGBE_v2_3a_retrain_abstain.ipynb", "run2", '["qwen17"]', "None"),
    ("3", '["C", "D", "E"]', "train the three bases on the full data, export the GGUFs and re-test, about 4 to 6 hours. Run it after 2b has finished", "AGBE_v2_3_train.ipynb", "run1", "None", "None"),
]
meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
for n, stages, what, fname, run, keys, epochs in SESSIONS:
    out_cells = []
    for c in cells:
        src = "".join(c["source"]).replace("@@STAGES@@", stages).replace("@@N@@", n).replace("@@RUN@@", run).replace("@@KEYS@@", keys).replace("@@FRESH@@", "False" if n == "3a" else "True").replace("@@IGNORE@@", '--ignore "*.gguf" "adapter_*"' if n == "3a" else "").replace("@@EPOCHS@@", epochs).replace("@@WHAT@@", what)
        out_cells.append(dict(c, source=src.splitlines(keepends=True)))
    (HERE / fname).write_text(json.dumps({"cells": out_cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}, indent=1))
    print("wrote", fname)
