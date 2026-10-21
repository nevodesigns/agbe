# AGBE v2 engineering log

What was built after the Africa Deep Tech Challenge final, in the order it happened, with the numbers, the faults found, and how to reproduce each step. Written so that someone picking the work up cold, including the author after a long break, can continue without the chat history. Last updated 21 October 2026.

## Where things stand (21 Oct 2026)

- The competition build, v16, is frozen at the tag `adtc-2026-final`. Result: top 10 finalist among 291 submissions, not top three.
- v2 is a different design: answers come from numbered, openly licensed source passages, every sentence is checked against them, and the model says so when no source covers the question.
- The whole pipeline runs end to end on a free Kaggle GPU: data, training, merge, GGUF export, held-out evaluation. The merge fault (fault 11) is fixed and confirmed: the finished folder is the tuned model and the GGUF converter now reads float16.
- **First scored v2 model (smoke test, 21 Oct 2026):** Gemma 3 1B, one epoch, 1,216 training examples, scored on 140 held-out cases (70 grounded, 70 abstain). Grounded pass 7.1% to 64.3%, valid citation 17.1% to 71.4%, right source 11.4% to 47.1%, correct abstention when no source covers the question 0% to 35.7%. The cost: it wrongly abstains on 28.6% of grounded cases (2.9% before). This is a smoke test, not the final model, and it was scored in float16 before quantisation. The cases are held out by passage, not by document, so other passages of the same documents were in training.

## The design

1. **Hard rules before the model.** Doses, medicine for people, harm, illegal crops, self harm, jailbreaks and off-domain requests get a fixed reply (`serve/guard.py`). The model never sees them.
2. **Retrieval.** BM25 over the passage store (`serve/retrieval.py`). No relevant passage means the fixed "I do not have a source" reply.
3. **Claim check.** Each answer sentence must be supported by a retrieved passage, and any chemical rate is removed whatever its support (`eval/claim_check.py`, `tools/extract_claims.py:dose_like`).
4. **Stop policy.** A refusal ends after its deferral. Answers are capped.
5. **Serving.** FastAPI endpoint, OpenAI compatible, with API key auth (`serve/app.py`).

v16 ignores retrieved sources, so the guard abstained on everything when wrapped around it. A general 3B model worked through the same guard. Hence v2 must be trained to answer from sources.

## Data pipeline

| Stage | Tool | Result |
|---|---|---|
| Licensed sources | `tools/cgspace_harvest.py`, `tools/licence_manifest.py` | 7,890 Tier A items (CC-BY-4.0, CC-BY-3.0, CC0, public domain) |
| Full text | `tools/fetch_fulltext.py` | 8,062 passages from 279 documents (`data/fulltext_v1/`) |
| Claims | `tools/extract_claims.py` with Qwen2.5-7B-Instruct, 4-bit | 1,765 proposed, 1,205 accepted, 12 for review, 548 rejected |
| Questions | `tools/build_grounded_sft.py questions` | 429 passages with an accepted claim, about 3 questions each |
| Training set | `tools/build_grounded_sft.py compose` | 1,280 examples: 964 grounded, 118 abstain, 198 refusal; 140 held-out test cases |

Claim verifier rules: the quote must appear verbatim in the passage, numbers must be in the quote, content words must be covered, study language (plots, treatments, "we") is rejected, and anything dose-like goes to review and is never accepted. Rejection reasons on the full run: claim adds words 338, quote not in passage 209, study language 36, length 29, dose-like 12, number not in quote 6.

Composer filters added after reading real examples (see faults 2 and 3): the lead claim must share at least 15% of the question's words, extra sentences at least 10%, survey findings (sentences starting "Most farmers", "Participants") and human-health sentences are dropped, and questions that say "the passage" or "the study" are dropped.

## Faults found, in order

Each was found by a test or by reading the real data. Keeping this list is the point of the log.

1. **The first model ignored its sources.** v16 answers from memory. Fix: train v2 on source-grounded examples.
2. **A quarter of grounded training answers did not address the question.** Found by reading examples. Fixed by the relevance rule above (68 of 375, later 170 of 1,450, dropped).
3. **Survey findings and human-health sentences passed as advice.** 27 survey claims and 4 health claims. Filtered.
4. **Kaggle's newer transformers returns a dict from the chat template.** Fixed with `encode_chat` and per-model failure isolation in `train/bakeoff.py`.
5. **The persistence script restored stale data.** `train/hf_persist.py` uploaded its own `.pull` scratch folder, nested deeper on every run, and restore matched nested copies, so one run trained on the old 495-example set. Fixed: push ignores `.pull`, `.cache` and `checkpoint-*`; restore goes to a temp directory. Verified by a real restore against the repository.
6. **`SFTConfig` no longer accepts `warmup_ratio`.** `train/train_grounded.py` now builds its config from the fields the installed version knows.
7. **The free T4 has no native bfloat16.** Training crashed on its first step (fp16 gradient scaler cannot unscale bf16). Fixed by loading in float16 and casting trainable weights to float32.
8. **The adapter was saved as the base model.** Training printed "adapter saved" but wrote the base. Fixed with `trainer.model.save_pretrained`. The notebook now skips training only if `adapter_config.json` exists.
9. **Old `torchao` blocks peft.** `train/merge_adapter.py` removes it before importing peft.
10. **GGUF export needs `virtualenv`.** Kaggle's Python has no `ensurepip`. `train/export_gguf.sh` falls back to virtualenv.
11. **The merge overwrote the trained weights.** After merging, a copy step copied every file from the base model's cache folder into the merged folder. That folder also held the base weights and config, so the merged model was replaced by the untrained one. Every tuned evaluation matched the base exactly on all 140 cases, and the GGUF converter reported bfloat16, the base file's format. The in-script self-check passed because it ran before the copy. Real held-out prompts (the `PROBE` lines) proved the trained model does answer in short cited sentences, for example "Rotate kraal to maintain or increase soil fertility if manure is available. [1]" against the base's "According to [1], rotating kraal systems require more effort...". Fixed by copying only tokenizer files and re-checking the finished folder (`FINAL FOLDER IS THE TUNED MODEL`). All GGUF files produced before this fix are base models and must not be used.

## Measurements so far

- **Bake-off (60 prompts, retrieval mode):** Qwen3-4B-Instruct 33/60 with 0 leaks; Gemma 3 1B 25/60, 7 leaks; Qwen3-1.7B 23/60, 4 leaks; Gemma 3 4B 23/60; SmolLM3-3B 25/60, 9 leaks. The small models are within noise. See `docs/BAKEOFF_RESULTS.md`.
- **Held-out baseline, untrained Gemma 3 1B, 140 cases:** grounded pass 7.1%, valid citation 17.1%, right source 11.4%, abstain pass 0%. (11.4% pass in earlier runs used bfloat16 loading; float16 is the correct setting on a T4.) These are the numbers training must beat.
- **Training, 1 epoch, 1,216 examples:** loss 2.8 to 0.44, token accuracy 0.89, 14 minutes on a T4.
- **First tuned model on the same 140 held-out cases (merged, float16, not yet quantised):**

| Measure | Untrained | Tuned, 1 epoch |
|---|---|---|
| Grounded pass | 0.071 | 0.643 |
| Supported sentences | 0.676 | 0.657 |
| Valid citation | 0.171 | 0.714 |
| Right source | 0.114 | 0.471 |
| Wrongly abstained (grounded cases) | 0.029 | 0.286 |
| Abstain pass (no source) | 0.0 | 0.357 |
| Answered although no source | 1.0 | 0.643 |

Reading: citing and answering from sources improved enormously. Knowing when to abstain improved but is still weak, and the model now abstains too often on answerable questions. Supported-sentence share did not improve, so the claim check in the guard stays necessary. Two epochs, larger bases and more abstain examples are the next levers.
- **Guard hard rules:** development set 53 of 62 attacks stopped and none of 30 legitimate questions blocked; holdout set (written before tuning) 28 of 30 and none of 22.
- **GGUF size:** Q4_K_M is 806 MB for Gemma 3 1B. 130 of 340 tensors fell back to other quantisation types, so the quantised model must be judged on its own, on a CPU, before any claim.

## Reproducing

```bash
cd tools && python3 -m unittest test_extract_claims test_build_grounded_sft
cd ../train && python3 -m unittest test_rag_eval
cd .. && AGBE_AUTOSTART=0 python3 -m unittest serve.test_guard serve.test_app
python3 train/make_v2_notebook.py      # writes the five Kaggle notebooks
python3 train/diagnose_data.py --claims claims.jsonl --sft sft.jsonl
```

Kaggle notebooks (generated, then imported by hand): `AGBE_v2_2b_more_data` (more claims, resumes), `AGBE_v2_3s_smoketest` (one base, one epoch, saves to its own folder), `AGBE_v2_3_train` (three bases, two epochs, GGUF, held-out test). Every tunable number is in `train/session_config.json`. Work is kept between sessions in a private Hugging Face dataset repository.

## Next steps

1. Done: the smoke test passed (`FINAL FOLDER IS THE TUNED MODEL: True`, scores above).
2. Run `AGBE_v2_3_train.ipynb` (about 4 to 6 hours): Gemma 3 1B, Qwen3 1.7B and Qwen3 4B on the same data.
3. Measure the GGUF files on a 4 core CPU (speed and memory) and score the quantised model, not only the merged one.
4. Choose an edge model and a server model from the held-out numbers, wire the guard to them, measure the whole system.
5. Grow the fact base, find an agronomist, follow up the permission emails (1 Nov 2026), then package: container, model card, phone build.

## Lessons

- Read the data, do not only count it. The relevance fault was invisible in every statistic.
- Make a check that can fail at the exact place of the claim. The merge self-check passed because it ran one step too early.
- When a result is exactly equal to another, suspect that it is the same thing, not that the method failed.
- Keep a file for every restore and save step, and test it against the real remote.
