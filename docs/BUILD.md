# How AGBE v2 is built

The pipeline in order. Every tunable number is in `train/session_config.json`. Tests run in CI (`.github/workflows/tests.yml`).

## 0. Principle

The model is trained to answer only from numbered passages it is shown, to cite them, and to say "I do not have a source" when they do not cover the question. A guard around the model adds fixed safety rules, retrieval, a sentence-level support check and a stop rule. Training data are built only from facts that were checked against their passage.

## 1. Sources (`tools/cgspace_harvest.py`, `tools/fetch_fulltext.py`, `tools/licence_manifest.py`)

1. List items with a stated open licence. Tier A is CC-BY-4.0, CC-BY-3.0, CC0-1.0 or public domain; only Tier A is trained on. Tier B (evaluation only) and Tier C (needs written permission) are listed in `data/sources/registry.json`. Result: 7,890 Tier A items listed (`data/sources/cgspace_tier_a.jsonl.gz`).
2. Re-check each item's licence live, download the PDF, convert it in reading order, cut it into short passages with an id, page and source link. Result: 8,062 passages from 279 documents, all CC-BY-4.0 (`data/fulltext_v1/`).
3. Write the licence manifest and `ATTRIBUTION.md`.

## 2. Claims (`tools/extract_claims.py`)

A 7B model (Qwen2.5-7B-Instruct, Apache-2.0) proposes one practical claim per passage. A verifier accepts a claim only if the quote is verbatim in the passage, states no chemical rate, and reads as advice and not as study language. Result: 1,765 proposed, 1,205 accepted (`data/run1_kaggle/claims.jsonl`).

## 3. Training examples (`tools/build_grounded_sft.py`)

- A farmer-style question is written for each accepted claim (`questions` command).
- `compose` builds three kinds of example: grounded (the answer passage plus distractors; the answer cites it), abstain (retrieved passages that do not support any accepted claim; the answer is "I do not have a source"), and refusal (doses, medical advice, illegal crops, self-harm).
- 10% of passages are held out for the test set, by passage hash. Abstain candidates that still support an accepted claim of the removed passage are dropped from training and flagged in the test set.
- Current set: 868 grounded, 200 abstain, 196 refusal examples; test set 140 cases (70 grounded, 70 abstain; 41 clean abstain cases).

## 4. Training (`train/train_grounded.py`, `train/make_v2_notebook.py`)

QLoRA on a Kaggle T4: rank 32, 2 epochs, learning rate 1.5e-4 (Qwen3-1.7B). The T4 has no native bfloat16: the base loads in float16 and the adapter weights stay float32. The notebook generator writes stage-gated notebooks (`AGBE_v2_*.ipynb`); each run clones the repository fresh, and saves to a private Hugging Face dataset repository through `train/hf_persist.py`. Tokens are Kaggle secrets only.

## 5. Merge and export (`train/merge_adapter.py`, `train/export_gguf.sh`)

Merge in float32, save float16, then check that the final folder differs from the base on real prompts. Export to GGUF Q4_K_M with llama.cpp built from git main. The merge check exists because one run once shipped base weights under a tuned name (see the engineering log).

## 6. Evaluation (`train/rag_eval.py`, `train/gguf_eval.py`, `train/guard_eval.py`)

Each held-out case is scored for: pass (answers, cites validly, at least 80% of sentences supported), supported share, right source, wrong abstention, and correct abstention on the clean cases. `gguf_eval.py` scores the shipping GGUF file on CPU through llama-server (flags: `--cache-ram 0 --ctx-checkpoints 0` to avoid an inflated prompt cache). `guard_eval.py` applies the guard to recorded answers. Note: the guard and the scorer share one claim checker, so the guard's own support figure is circular; only a human review is independent.

## 7. Serving (`serve/`)

`serve/app.py` is a FastAPI service with `/advise` and an OpenAI-compatible `/v1/chat/completions` (optional API keys). `serve/guard.py` runs hard rules, BM25 retrieval (`serve/retrieval.py`), the model, then the sentence check and answer-then-stop. By default the prompt carries the best 3 passages trimmed to about 100 words (`serve/prompt.py`; `AGBE_COMPACT=3,100`, `off` disables).

```bash
pip install -r serve/requirements.txt
gunzip -k data/fulltext_v1/passages.jsonl.gz
AGBE_GGUF=model.gguf AGBE_PASSAGES=data/fulltext_v1/passages.jsonl uvicorn serve.app:app --port 8080
```

## 8. Tests

```bash
cd eval && python3 -m unittest test_claim_check
cd ../tools && python3 -m unittest test_extract_claims test_build_grounded_sft
cd ../train && python3 -m unittest test_rag_eval test_gguf_eval test_guard_eval
cd .. && AGBE_AUTOSTART=0 python3 -m unittest serve.test_guard serve.test_app serve.test_prompt
python3 eval/measure_guard.py
```

## 9. The competition build (v16)

Built from `corpus/` (hand-written facts composed by `corpus/generate.py`, 1,237 training conversations, 107 held out), trained with `train/train_lora.py` (3 epochs, 117 optimiser steps, LoRA rank 32; record in `provenance/`), scored with `eval/run_eval.py`, packaged by `download_model.sh`. `tools/check_submission.sh` and `tools/check_docs.py` check the documents against `FINAL.json`. `BUILDS.md` is the ledger of every candidate.
