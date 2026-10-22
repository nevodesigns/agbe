# AGBE v2 engineering log

What was built after the Africa Deep Tech Challenge final, in the order it happened, with the numbers, the faults found, and how to reproduce each step. Written so that someone picking the work up cold, including the author after a long break, can continue without the chat history. Last updated 21 October 2026.

## Where things stand (21 Oct 2026)

- The competition build, v16, is frozen at the tag `adtc-2026-final`. Result: top 10 finalist among 291 submissions, not top three.
- v2 is a different design: answers come from numbered, openly licensed source passages, every sentence is checked against them, and the model says so when no source covers the question.
- The whole pipeline runs end to end on a free Kaggle GPU: data, training, merge, GGUF export, held-out evaluation. The merge fault (fault 11) is fixed and confirmed: the finished folder is the tuned model and the GGUF converter now reads float16.
- **Full run, 21 Oct 2026:** Qwen3 1.7B reaches 87.1% grounded pass (from 24.3%) and Qwen3 4B 88.6% (from 31.4%); see the results section. Earlier smoke test:  Gemma 3 1B, one epoch, 1,216 training examples, scored on 140 held-out cases (70 grounded, 70 abstain). Grounded pass 7.1% to 64.3%, valid citation 17.1% to 71.4%, right source 11.4% to 47.1%, correct abstention when no source covers the question 0% to 35.7%. The cost: it wrongly abstains on 28.6% of grounded cases (2.9% before). This is a smoke test, not the final model, and it was scored in float16 before quantisation. The cases are held out by passage, not by document, so other passages of the same documents were in training.

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

12. **The first CPU memory reading was wrong.** llama.cpp's server keeps a cache of earlier prompts, so a 1.1 GB model showed a 10 GB peak. Caught because the numbers could not be true; the scorer now disables the caches and the repeat gave 1.0, 2.0 and 4.5 GB.

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

## Full training run results (21 Oct 2026)

Three bases trained on the same 1,216 examples for two epochs each (about 5.8 hours on a free Kaggle T4, notebook `AGBE_v2_3_train`), merged, exported to GGUF Q4_K_M, and scored on the same 140 held-out cases (70 grounded, 70 abstain; held out by passage, not by document; scored on the merged float16 model, before quantisation).

| Model | Grounded pass | Supported sentences | Valid citation | Right source | Wrongly abstains | Correct abstention |
|---|---|---|---|---|---|---|
| Gemma 3 1B, untrained | 0.071 | 0.676 | 0.171 | 0.114 | 0.029 | 0.000 |
| Gemma 3 1B, tuned | 0.657 | 0.674 | 0.743 | 0.629 | 0.257 | 0.357 |
| Qwen3 1.7B, untrained | 0.243 | 0.582 | 0.700 | 0.557 | 0.000 | 0.014 |
| **Qwen3 1.7B, tuned** | **0.871** | **0.895** | **0.957** | **0.929** | **0.043** | **0.414** |
| Qwen3 4B Instruct, untrained | 0.314 | 0.582 | 1.000 | 0.900 | 0.000 | 0.186 |
| **Qwen3 4B Instruct, tuned** | **0.886** | **0.910** | **0.971** | **0.914** | **0.029** | **0.714** |

Training time: Gemma 30 minutes, Qwen3 1.7B 59 minutes, Qwen3 4B about 2.4 hours. Mean training loss 0.51 (Gemma) and 0.45 (Qwen3 1.7B); validation token accuracy 0.92 and 0.93 respectively.

Reading:
- Qwen3 1.7B is the edge candidate: 87% grounded pass with a third of the cost of the 4B, and it wrongly abstains on only 4% of answerable cases. Its weakness is abstention: it answers in 59% of the cases where no source covers the question.
- Qwen3 4B is the server candidate: the best on every measure and the only one that abstains correctly most of the time (71%).
- Gemma 3 1B, the competition base, is clearly the weakest after training (66% pass, 26% wrong abstentions). A different base is the better choice for v2. Both Qwen3 models are released under the Apache 2.0 licence, which should be re-checked on the model cards before any commercial use; the Gemma terms are more restrictive.
- The next gap is abstention: it needs more and harder abstain examples (questions near the right topic but not covered), and the guard's claim check stays in place as the safety net.

Files (private Hugging Face repository `NEVODESIGN/agbe-v2-work`, folder `run1`): adapters, `agbe-v2-{gemma1b,qwen17,qwen4b}-q4_k_m.gguf` (806 MB, 1.1 GB, 2.5 GB) with checksums, and the per-case score files `rageval_{base,tuned}_*.json`.

Not yet done: speed on the real competition machine (only ratios to v16 on a cloud CPU); the held-out test is still small (140 cases) and shares documents with training; no human review of answers yet.

Known quirk: the `FINAL FOLDER IS THE TUNED MODEL` line printed `False` for the first merge (Gemma) although its scores prove it is tuned. The check looked for a citation marker; it now compares against the untrained answer on the same prompt.

## The shipped 4-bit files on a CPU (21 Oct 2026, notebook `AGBE_v2_4_gguf_cpu`)

The three Q4_K_M GGUF files were scored on the same 140 held-out cases with llama.cpp's own `llama-server` (built from git main, the same source that made the files), greedy decoding, the chat template without thinking, 4 threads, on a Kaggle CPU (Intel Xeon 2.20 GHz, 4 cores, 31 GB RAM). Scorer: `train/gguf_eval.py`.

| Model | Pass (fp16 to Q4) | Supported | Valid cite | Right source | Wrongly abstains | Correct abstention |
|---|---|---|---|---|---|---|
| Gemma 3 1B | 0.657 to 0.729 | 0.674 to 0.736 | 0.743 to 0.757 | 0.629 to 0.543 | 0.257 to 0.243 | 0.357 to 0.386 |
| Qwen3 1.7B | 0.871 to 0.871 | 0.895 to 0.883 | 0.957 to 0.914 | 0.929 to 0.886 | 0.043 to 0.071 | 0.414 to 0.414 |
| Qwen3 4B | 0.886 to 0.900 | 0.910 to 0.907 | 0.971 to 0.943 | 0.914 to 0.900 | 0.029 to 0.057 | 0.714 to 0.800 |

Reading: shrinking to 4-bit costs essentially nothing in quality (differences of a few points in either direction on 70 cases are noise). Qwen3 4B reaches 90.0% pass and 80.0% correct abstention as a 2.5 GB file.

Speed on that CPU (prompts average about 850 tokens: three retrieved passages plus the instructions):

| Model | File | Generation | Prompt processing | Per answer |
|---|---|---|---|---|
| Gemma 3 1B | 806 MB | 13.5 tokens/s | 46 tokens/s | 21 s |
| Qwen3 1.7B | 1.1 GB | 8.8 tokens/s | 35 tokens/s | 31 s |
| Qwen3 4B | 2.5 GB | 4.8 tokens/s | 16 tokens/s | 64 s |

Findings and caveats:
- **Prompt processing dominates the answer time** (856 tokens at 35 tokens/s is about 25 of the 31 seconds for Qwen3 1.7B). The levers are fewer and shorter retrieved passages, and caching the fixed instruction prefix. Both are product changes, not model changes.
- **This CPU is a shared cloud vCPU and slower than the competition machine** (v16 measured 24.08 tokens/s there; Gemma 3 1B here gives 13.5), so these absolute speeds are pessimistic. The comparison that means something is against v16 on the same machine (next notebook).
- **The first memory figures were wrong** (4.7, 10.5 and 13.0 GB): `llama-server` keeps a RAM cache of earlier prompts (up to 8 GB by default) that inflated the peak. See the repeat below.

### Speed and memory, repeated with the caches off (notebook `AGBE_v2_5_speed_memory`, 22 Oct 2026)

Same Kaggle CPU (Intel Xeon 2.20 GHz, 4 cores), context 2,048 as in the competition, `--cache-ram 0 --ctx-checkpoints 0`, 30 held-out cases, with the **v16 competition file timed on the same machine as a control**.

| File | Size | Peak memory | Generation | Relative to v16 | Prompt processing | Per answer |
|---|---|---|---|---|---|---|
| v16 (competition) | 814 MB | 1,049 MB | 13.87 tokens/s | 1.00x | 42.8 tokens/s | 22.8 s |
| v2 Gemma 3 1B | 806 MB | 995 MB | 13.84 tokens/s | 1.00x | 43.0 tokens/s | 20.1 s |
| v2 Qwen3 1.7B | 1,107 MB | 2,040 MB | 10.07 tokens/s | 0.73x | 37.6 tokens/s | 25.2 s |
| v2 Qwen3 4B | 2,497 MB | 4,484 MB | 4.85 tokens/s | 0.35x | 14.5 tokens/s | 61.7 s |

Reading:
- **Memory is fine.** Every file fits the 8 GB laptop budget; the edge candidate uses 2.0 GB and the server candidate 4.5 GB.
- **Speed relative to v16 is the fair comparison.** v16 measured 24.08 tokens/s on the competition machine, so scaling by the ratios gives roughly 17 tokens/s for Qwen3 1.7B and 8 tokens/s for Qwen3 4B there. That is an estimate from a ratio, not a measurement on that machine.
- **Prompt length is the main cost.** Prompts average about 750 tokens; at 37 tokens/s Qwen3 1.7B spends roughly 20 of its 25 seconds reading them.
- **Control result:** v16 on the source-grounded test scores 0% grounded pass, 0% valid citation and 17.7% supported sentences. It ignores the sources it is given, which is the reason v2 had to be trained.
- Quality figures from these 30 cases are not quoted (15 grounded and 15 abstain cases are too few); the 140-case table above is the reference.

Decision candidates: **Qwen3 1.7B for the edge** (2.0 GB, 0.73x v16 speed, 87.1% pass) and **Qwen3 4B for a server or a strong laptop** (4.5 GB, 0.35x v16 speed, 80.0% correct abstention).

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
2. Done: `AGBE_v2_3_train.ipynb` ran (results above).
3. Done: quality after shrinking, speed against v16 on the same machine, and memory (CPU sections above). Still to do: cut the prompt (fewer, shorter passages, cached prefix) and re-time.
4. Wire the guard to Qwen3 1.7B (edge) and Qwen3 4B (server) and measure the whole system, including the guard's effect on abstention.
5. Add harder abstain examples and retrain to lift correct abstention; enlarge the held-out set and split it by document.
6. Grow the fact base, find an agronomist, follow up the permission emails (1 Nov 2026), then package: container, model card, phone build.

## Lessons

- Read the data, do not only count it. The relevance fault was invisible in every statistic.
- Make a check that can fail at the exact place of the claim. The merge self-check passed because it ran one step too early.
- When a result is exactly equal to another, suspect that it is the same thing, not that the method failed.
- Keep a file for every restore and save step, and test it against the real remote.
