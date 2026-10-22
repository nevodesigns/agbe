# AGBE recovery guide

One document to rebuild the project from nothing, or to pick it up cold after a long break. Written 23 October 2026. It states what exists, where it lives, what was measured, how to rebuild each part, what went wrong on the way, and what is next. The detail behind each section is in `docs/V2_ENGINEERING_LOG.md` (chronology, faults, tables) and `docs/V2_HANDOFF.md` (status board). Private matters (people, accounts, business arrangements) are deliberately not in this repository.

## 1. What this is

AGBE (Àgbẹ̀, Yoruba for farmer) is an offline agricultural advisor for smallholder farmers and extension officers in West and Central Africa.

- **v16, the competition build.** Gemma 3 1B fine-tuned (LoRA), GGUF Q4_K_M, 814 MB, 24.08 tokens per second and 1,039 MB of memory on the competition machine. It reached the final of the Africa Deep Tech Challenge 2026 as **one of the top 10 finalists among 291 submissions** (it did not place in the top three). Frozen at the git tag `adtc-2026-final`. It answers from memory and ignores retrieved sources.
- **v2, the current line (branch `agbe2`).** Answers come from numbered, openly licensed source passages, every sentence is checked against them, and the model says so when no source covers the question. Not shipped yet.

## 2. Where everything is

| What | Where |
|---|---|
| Code, data samples, docs | GitHub `nevodesigns/agbe`, branch `agbe2` (default). `main` and tag `adtc-2026-final` hold the competition build. Tag `v2-checkpoint-2026-10-23` marks the state this guide describes |
| Passages and claims | `data/fulltext_v1/` (8,062 passages, 279 open-licence documents, `ATTRIBUTION.md`); `data/run1_kaggle/` (claims, questions, training examples, logs); `data/sources/`, `data/LICENCE_MANIFEST.md` |
| Trained models, adapters, scores | Private Hugging Face dataset repository `NEVODESIGN/agbe-v2-work`: `run1/` (three bases: adapters, three Q4_K_M GGUF files, scores), `run2/` (retrained Qwen3 1.7B, clean abstain data), `gguf/` (4-bit scores), `speed/` (speed and memory with the v16 control), `prompt/` (prompt-length experiment), `smoke/` (the smoke test). About 7.7 GB in total |
| Competition weights | Public `NEVODESIGN/agbe-1b` on Hugging Face (v16) |
| Kaggle notebooks | Generated into `train/`, copies in `~/Downloads/agbe-kaggle/` on the author's laptop. Run slugs under the account `nevodm`: `notebook2b13bd5043` (bake-off), `notebook0423bb70f5` (data run), `agbe-v2-2b-more-data`, `agbe-v2-3s-smoketest`, `agbe-v2-3-train`, `agbe-v2-4-gguf-cpu`, `notebook92134dcc52` (speed and memory), `notebook0192936801` (prompt and guard), `notebook82ff15355b` (first retraining, abandoned), `notebook36fefae435` (corrected retraining) |
| Product site | https://agbe-omega.vercel.app (repo `nevodesigns/agbe-site`; home is the current AGBE, `/adtc/` is the full competition record). Deployed with `vercel deploy --prod --yes` |
| Private record and backup | Outside the repository: a master record, an archive of memory notes, notebooks and docs. Keep a copy off the laptop |

## 3. Key numbers (all on 140 held-out cases, held out by passage not by document)

**Data.** 7,890 openly licensed items (CC-BY-4.0, CC-BY-3.0, CC0, public domain) listed; 8,062 passages from 279 documents; Qwen2.5-7B proposed 1,765 claims and a verifier accepted 1,205 (quote verbatim in the passage, no chemical rate, advice not study language). Training set: 868 grounded, 200 abstain, 196 refusal examples. Test set: 70 grounded and 70 abstain cases; 29 of the 70 abstain cases turned out to be answerable and are reported apart, leaving **41 clean abstain cases**.

**Models** (answers correctly from the sources, untrained to trained; "no source" is on the 41 clean cases):

| Model | Pass | Says "no source" | Memory | Speed vs v16 (same CPU) |
|---|---|---|---|---|
| Gemma 3 1B (Q4) | 7.1% to 72.9% | 43.9% | 1.0 GB | 1.00x |
| **Qwen3 1.7B** (edge) | 24.3% to **94.3%** (retrained; 87.1% before) | 48.8% | 2.0 GB | 0.73x |
| **Qwen3 4B** (server, Q4) | 31.4% to 90.0% | **85.4%** | 4.5 GB | 0.35x |

The 94.3% for the retrained Qwen3 1.7B is a merged float16 score; its 4-bit file is saved but not yet scored. Shrinking to 4-bit cost nothing measurable for the earlier models. v16 on the same source-grounded test scores 0%, because it ignores sources.

**Prompt.** Keeping the best 3 passages trimmed to about 100 words keeps the pass rate (87.1%) and cuts the answer time by 39% (30.6 s to 18.7 s on a slow shared cloud CPU); it is the app default (`AGBE_COMPACT=3,100`, `off` to disable). **Guard.** The guard's sentence check raises grounded pass by 1 to 3 points without hurting abstention; the hard rules stop 53 of 62 development attacks and 28 of 30 holdout attacks with none of the legitimate questions blocked.

## 4. Rebuild from nothing

1. `git clone -b agbe2 https://github.com/nevodesigns/agbe.git`.
2. Run the tests exactly as CI does (`.github/workflows/tests.yml`): `cd eval && python -m unittest test_claim_check`; `cd tools && python -m unittest test_extract_claims test_build_grounded_sft`; `cd train && python -m unittest test_rag_eval test_gguf_eval test_guard_eval`; `AGBE_AUTOSTART=0 python -m unittest serve.test_guard serve.test_app serve.test_prompt`; `python eval/measure_guard.py`.
3. Regenerate the notebooks: `python3 train/make_v2_notebook.py`, `make_speed_notebook.py`, `make_prompt_notebook.py`, `make_eval_notebook.py`, `make_gguf_notebook.py`. Every tunable number is in `train/session_config.json`.
4. On Kaggle: create a private Hugging Face dataset repository for the work, add the secrets `HF_WRITE_TOKEN` and `HF_TOKEN` (tokens belong only in Kaggle secrets and your own environment, never in a file or the repository), import a notebook, set the accelerator (GPU T4 x2 for training, None for the CPU notebooks), Internet on, Save and Run All, then Download Logs.
5. Order of the data and training notebooks: `AGBE_v2_2b_more_data` (claims, questions, training data, resumable), then `AGBE_v2_3_train` (three bases), then `AGBE_v2_3a_retrain_abstain` (retrain Qwen3 1.7B with clean abstain data), then the CPU notebooks `4`, `5`, `6`, `7`. The data are also in `data/run1_kaggle/` if the work repository is lost; the passages and claims are enough to rebuild the training set with `tools/build_grounded_sft.py compose`.
6. Serve: `AGBE_GGUF=model.gguf AGBE_PASSAGES=data/fulltext_v1/passages.jsonl uvicorn serve.app:app --port 8080` (FastAPI, OpenAI-compatible, optional API keys). A `Dockerfile` and `docker-compose.yml` exist and have never been built.

## 5. What went wrong, so it does not happen again

1. v16 ignored its sources, so v2 had to be trained to answer from them.
2. A quarter of the first training answers did not address the question (found by reading examples, not counting).
3. Survey findings and human-health sentences passed as advice.
4. Kaggle's newer transformers changed the chat template output.
5. The save and restore script uploaded its own scratch folder and restored stale copies (one run trained on old data).
6. `SFTConfig` no longer accepts `warmup_ratio`.
7. The free GPU has no native bfloat16: load float16, keep adapter weights in float32.
8. The adapter was saved as the base model.
9. An old `torchao` blocked the merge.
10. GGUF export needs `virtualenv` on Kaggle.
11. **The merge overwrote the trained weights with the base weights** (a copy step copied every file from the base cache folder). Every tuned model equalled the base until this was found.
12. The first memory reading (10 to 13 GB) was the inference server's own prompt cache.
13. **29 of the 70 "no source" test cases were answerable** (other documents repeat the advice); the same flaw sat in the training abstain examples, and a retraining with same-document hard negatives made abstention worse (the model likely learned that same-document sources mean abstain). The test now flags answerable cases and the training data filters them.

Lessons: read the data; make each check able to fail at the exact place of the claim; when two results are identical suspect that they are the same thing; keep a save and restore test against the real remote; a surprising number is a prompt to look for a fault before explaining it.

## 6. Next steps

1. Run `AGBE_v2_3c_train_short_prompt` (GPU) and `AGBE_v2_7_eval_run2` (CPU) in parallel, then `AGBE_v2_8_eval_run3`: does training on the short prompt recover the accuracy the shorter settings lost?
2. Push abstention (clean abstain share 0.35 to 0.40), now that wrong abstentions are 1.4%.
3. Score the retrained model's 4-bit file; test the guard wired to the real model end to end; build a larger test set split by document; have a person review answers.
4. Model card; verify the Qwen3 model cards' licence before any commercial use; container build; phone build.
5. Find an agronomist to review the fact base; follow up the permission requests to the publishers whose material is not yet usable (earliest 1 November 2026).
