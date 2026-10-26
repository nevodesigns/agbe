# AGBE v2 handoff

Read [RECOVERY.md](RECOVERY.md) first for the whole picture in one page. Read this if you are picking the work up cold, whether you are me in a new session or a person. It records what exists, why, what is verified, what is not, and what to do next. Update it when something meaningful changes. Last updated 18 October 2026 (evening).

## 0. Status board (21 October 2026)

| Item | State |
|---|---|
| v16 competition record | Frozen at tag `adtc-2026-final` (commit 1d32e67). ADTC finished 17 Oct: top 10 finalist among 291 submissions (not top 3), GPU credits awarded |
| Tier A data | 7,890 items listed; 8,062 passages from 279 documents in `data/fulltext_v1/` |
| Claims and training data | Done. 1,205 accepted claims from 429 passages; 1,280 training examples and 140 held-out cases after filters. Raw copies in `data/run1_kaggle/` |
| Guard, endpoint, evaluation, CI | Built and tested on CPU, CI green |
| Kaggle pipeline | Runs end to end (train, merge, GGUF, held-out evaluation). Five generated notebooks, see `docs/V2_ENGINEERING_LOG.md` |
| Trained v2 models | Three bases trained and scored (21 Oct), shrunk to Q4_K_M with no measurable loss. Grounded pass: Qwen3 4B 90.0%, Qwen3 1.7B 87.1%, Gemma 3 1B 72.9%. Correct abstention on the 41 truly unanswerable cases: 85.4%, 51.2%, 43.9% (the original 70-case figure was contaminated, see the engineering log). Memory 1.0, 2.0 and 4.5 GB; speed 1.00x, 0.73x, 0.35x of v16 on the same CPU. Shorter prompt (3 passages x 100 words) adopted as the default: same pass rate, 39% faster. Edge candidate Qwen3 1.7B, server candidate Qwen3 4B |
| Full training (three bases) | Done. Next: CPU speed, memory and quantised quality; abstain data; wire the guard |
| Permission emails | Sent 18 Oct. IITA Cc bounced. Follow up on or after 1 Nov 2026 |
| Agronomist | Not found yet |
| Edge model (23 Oct) | Qwen3 1.7B retrained with clean abstain data, as a 4-bit file with the short prompt: 90.0% pass, 97.1% valid citation, 2.9% wrong abstentions, 58.5% correct abstention on the 41 clean cases, 2.0 GB, 19 s per answer on a slow cloud CPU (91.4% pass and 51.2% on whole passages, 26.6 s). Training on the short prompt (run3) did not help and is not adopted |
| Prepared, not yet run | none. Run4 (40% abstain share) was run: clean abstain 46.3% vs 48.8% for run2, no gain, not adopted; `AGBE_v2_9_eval_run4` and `AGBE_v2_8_eval_run3` are not needed |
| Not yet verified | speed on the real competition machine, answers reviewed by a person, a shorter prompt and its effect on speed, container build, phone build |

The complete private record (people, accounts, business arrangements, all file locations) is kept outside this repository. The full chronology of faults, fixes and numbers is in `docs/V2_ENGINEERING_LOG.md`.

## 1. Where things stand

AGBE v16 (1B Gemma, GGUF Q4_K_M) was submitted to the Africa Deep Tech Challenge 2026 and reached the final (10 of 291 submissions). The competition is over. The goal now is a production-grade, deployable, grounded successor that stands up against global entries: a model, an endpoint, and a system that can run as a local LLM or as a feature inside another product.

Author: Nwokolo Victor Oluebubechukwu, solo, Lagos. Public contact: nwokoloebube21@gmail.com.

## 2. Decisions (17 Oct 2026)

- **Data: Tier A only for now.** Tier A means the item states CC-BY-4.0, CC-BY-3.0, CC0-1.0 or public domain. Anything else is not trained on.
- **Compute: free tiers only.** Kaggle (T4) and Colab free.
- **Agronomist: none yet.** Not yet found; the author is looking. Profile in `docs/AGRONOMIST_ROLE.md`. The expert gold set cannot start without one.
- **Branch:** `agbe2` is the GitHub default branch. `main` is deliberately untouched at commit `1d32e67`, tagged `adtc-2026-final`. That is the competition record, and the commit pinned in the submission.
- **Style rules (apply to everything in this repo):** never add an AI tool as co-author or in any attribution, never use em dashes, do not fabricate numbers, label synthetic data as synthetic.

## 3. Licence findings (verified 17 Oct 2026)

| Source | Finding | Tier |
|---|---|---|
| CGSpace items with a stated CC-BY licence | 30,587 items state CC-BY-4.0. The licence is per item, in the `dcterms.license` field. | A |
| IITA | Items dated roughly 2019 onward mostly state CC-BY-4.0 and are in the Tier A list. Older items have no licence field. IITA's policy page says free reuse with acknowledgment, an older notice says non-commercial and no modification. A policy statement is not a licence. | A for licensed items, C for the rest |
| FAO publications | CC BY-NC-SA 3.0 IGO. No commercial use without a separate request. | B (evaluation only) |
| CABI PlantwisePlus | Licence per item. The CABI Hugging Face dataset is CC BY-NC 4.0. Commercial use means contacting CABI. | C |
| NAERLS | Unknown. | C |
| AGROVOC, Wikidata, FAOSTAT | Believed CC BY 4.0, CC0 and CC BY 4.0. Not yet confirmed. | to verify |

Registry with statuses: `data/sources/registry.json`. This is not legal advice.

## 4. What exists on the `agbe2` branch

The full map of folders and files, with the state of each, is in [`docs/README.md`](README.md).

## 5. Findings so far

- **v16 ignores retrieved sources** (it answers from its fine-tuned memory), so the guard abstains on everything when v16 is the generator. An instruction-following model (Qwen2.5-3B) gives cited, supported answers through the same guard. The v2 model must be trained to answer from sources.

- A claim is accepted only if code proves it: the quote is verbatim in the passage, numbers match, content words match, it is not study language, and it is not a chemical rate (those go to `review`).
- A rule-based extractor produces mostly research sentences and is useless for relevance. Its first dose detector let "10 ml herbicide diluted to 1 litre" through. The detector now flags any chemical word with a quantity, any mass or volume per area or volume, and dose wording.
- Qwen2.5-3B Q4 on CPU, 24 passages: 29 claims, 18 rejected by the verifier (about 40% of the model's claims went beyond its source), 11 accepted. The 11 read as farm facts but none is expert-reviewed and some are from outside West Africa. A pipeline test, not training data.
- The earlier ADTC finding stands: the shipped v16 is accurate in its first sentence and drifts after it, and it varies at higher temperatures. The v2 fix is retrieval with citations plus claim-level checking plus an "answer, then stop" policy.

## 5b. Added 18 October 2026 (evening)

- **Guard rules, measured** (`python3 eval/measure_guard.py`): the deterministic rules stop 53 of 62 attacks in the dev battery with 0 of 30 legitimate questions blocked, and 28 of 30 attacks in a holdout set written before the rules were extended (0 of 22 legitimate blocked). Before the extension the same rules stopped 16 of 62 and 6 of 30. The rest of the attacks go to the model and the claim checker. Two dev prompts allowed to be refused (`refusal_ok`) are not counted.
- **`tools/build_grounded_sft.py`**: builds source-grounded training conversations (grounded, abstain, refusal) from verified claims. Questions come from an LLM (stage `questions`), answers are composed from accepted claims with citations (stage `compose`). 4 tests pass. A sample file from real pilot data is `data/claims/sample_sft.jsonl`. Known limit: the answer composer picks claims by word overlap, so an LLM answer writer with sentence-level verification would be better.
- **`train/train_grounded.py`**: QLoRA fine-tune script (completion-only loss). UNTESTED, it needs a GPU session. Its data check could not be run locally because the tokenizer install timed out.
- **CI**: `.github/workflows/tests.yml` runs all unit tests and the guard measurement on every push to `agbe2`. Last run passed.
- **`Dockerfile` and `docker-compose.yml`**: written, never built (docker is not installed on the dev machine).
- **`data/LICENCE_MANIFEST.md`**: every source behind the pilot passages with its licence, generated by `tools/licence_manifest.py`, which fails if anything is outside Tier A.
- **Outreach**: the four permission emails were sent on 18 October. One IITA Cc address bounced and needs replacing. Follow up on or after 1 November 2026.
- **Full-text fetch**: a larger run (350 items) lost its network partway (DNS errors), was resumed, and was still running slowly when this was written. `data/fulltext/` is git-ignored: rerun `python3 tools/fetch_fulltext.py --pilot 400 --workers 3` to rebuild it.

## 5b2. Bake-off result (19 October 2026)
See `docs/BAKEOFF_RESULTS.md`. Qwen3-4B-Instruct is the only clear winner (0 leaks, 33/60). The small models tie within noise, and the old batteries are the wrong shape for retrieval, so notebook 3 now holds out 10% of passages, scores every base on a source-following test (`train/rag_eval.py`), trains gemma-3-1b-it, Qwen3-1.7B and Qwen3-4B-Instruct, and compares each with its base. Notebook 2 was already running when this changed; notebook 3 recomposes the training data with the held-out split itself.

## 5c. The one-notebook route (19 October 2026)

**Run it as three notebooks, one per session**, generated with the stages preset by `train/make_v2_notebook.py`: `AGBE_v2_1_bakeoff.ipynb` (stage A), `AGBE_v2_2_data.ipynb` (B), `AGBE_v2_3_train.ipynb` (C, D, E). Import each into Kaggle, Save Version, Save & Run All (background run, the browser can be closed). Nothing needs editing. Each session restores the earlier ones from the Hugging Face repo. The notebook uses `HF_WRITE_TOKEN` if present, else `HF_TOKEN`. Passages: 8,062 from 279 documents in `data/fulltext_v1/` (the larger fetch finished: 280 downloaded, 61 had no PDF, 9 failed on incomplete reads).

The v2 notebooks (generated by `train/make_v2_notebook.py`) run stages A to E on a free Kaggle GPU: bake-off in retrieval mode, base pick (`tools/pick_base.py`, rule fixed in advance: fewest leaks, then most passed, then fewest unsupported claims, then smaller), claim extraction and question generation with a 7B model, training-data composition, QLoRA training, merge, GGUF export (`train/export_gguf.sh`), and a re-test of the new model against its base. State is saved to a private Hugging Face dataset repo (`train/hf_persist.py`, default `NEVODESIGN/agbe-v2-work`) so work survives disconnects. The passages it uses are committed in `data/fulltext_v1/`. UNTESTED on a GPU: the batching code in `tools/extract_claims.py`, `train/train_grounded.py`, `train/merge_adapter.py` and `train/export_gguf.sh` were written without one. If a stage fails, the error text is the thing to send back.

## 6. Resume here (in order)

1. **Deeper harvest: DONE** (merged 18 Oct, 7,890 items). Originally started 18 Oct with `python3 tools/cgspace_harvest.py --pages 5 --out data/sources/cgspace_tier_a_deep.jsonl --summary data/sources/cgspace_summary_deep.json`. About 85% done when this was written (log at `/tmp/claude-1000/harvest3.log`, may be gone). If the files exist, merge them into the registry, gzip the jsonl, update `data/sources/README.md` and `registry.json`, commit. If not, rerun the command (about 2 hours).
2. **Bake-off on Kaggle.** (Plain words: give each candidate base model the same questions, with and without retrieved sources, and score the answers. It shows which model follows the sources best, before any training money is spent.) The owner runs `train/KAGGLE_BAKEOFF.md` and sends `summary.md`. Choose the top two models from that, not from any assumption.
3. **Claim extraction at scale on Kaggle.** (Plain words: a stronger 7-billion-parameter model reads the passages and proposes facts, which code then verifies against the text. The small model I tried on a CPU made facts up in 40% of cases, so a better model makes more usable facts and the verifier still checks every one.) `python3 tools/extract_claims.py --backend transformers --model Qwen/Qwen2.5-7B-Instruct --passages passages.jsonl --out claims.jsonl`. Fetch more full text first (`tools/fetch_fulltext.py --all --limit 2000`, or `--pilot N` for a balanced sample). Prefer practical documents (manuals, briefs, extension material).
4. **Agronomist audit** of a random sample of accepted claims, and the 500-question gold set.
5. **Retrieval and inference guard:** index the verified claims and passages, add the claim-level checker at inference, and an answer-then-stop policy.
6. **Staged training** on the best base: grounded SFT, preference training on supported versus unsupported answers, quantisation-aware export.
7. **Serving and packaging (guard, endpoint and backends are built; still to do: source-grounded training, container image, CI gate, model card, phone build):** OpenAI-compatible endpoint plus `/advise` with citations and confidence, container image, auth and quotas, CI gate on the batteries, model card, data-licence manifest, a phone build.
8. **Pilot** with one extension service.

## 7. Things only a person can do

- Permission requests are tracked outside the repository. Follow up with IITA, CABI and NAERLS; FAO is no longer needed for the Flinck work (its web content is non-commercial by default).
- Find an agronomist.
- Run the Kaggle notebooks.

## 8. Related, outside this repo

- AGBE product site: https://nevo-hackathon.pxxlspace.cv/agbe/ (source repo `nevodesigns/agbe-site`, published inside the NEVO hackathon hub with `tools/sync_hub.py`). The standalone Vercel and pxxl sites were deleted on 26 October 2026.
- Shared memory notes for the assistant: `~/.claude/projects/-home-nwokolo-projects-agbe/memory/` (see `agbe-v2-plan.md`).

## 9. Gotchas learned the hard way

- The CGSpace API is slow (40 to 60 seconds for a 100-item page). Use 4 parallel workers, not more, and run long jobs with `nohup` in the background.
- `pkill -f "<command text>"` inside a shell command kills that shell, because the text matches itself. Use a bracketed pattern such as `pgrep -f "[c]gspace_harvest"`.
- Scratch directories can be cleared between sessions. Keep real work in the repo.
- Never commit full-text PDFs or large generated files. `data/fulltext/` and the plain `.jsonl` copies are git-ignored. The committed pilot passages carry an attribution file.
- Gemma and Llama weights are gated on Hugging Face. Accept the terms before a Kaggle run.
- Run tests with `cd eval && python3 -m unittest test_claim_check` and `cd tools && python3 -m unittest test_extract_claims`.

## Update 19 Oct 2026 (after Kaggle notebook 2)

Notebook 2 produced 375 training examples from 590 passages (570 accepted claims, 191 passages with a claim). Too small, so:
- `tools/extract_claims.py`: relaxed passage selection (`--max-study-hits 2 --min-signal 2`), 1,739 of 8,062 passages instead of 590.
- `tools/build_grounded_sft.py`: three questions per passage, two grounded variants per question, 25 v16 refusals reused.
- `train/session_config.json`: every tunable number for the notebooks (extract passages, bases, epochs, holdout, variants).
- `train/diagnose_data.py`: prints claim and example statistics and random samples, so a log shows data quality.
- Notebooks (`python3 train/make_v2_notebook.py`): `2b_more_data` (resumes, adds data to run1), `3s_smoketest` (one base, one epoch, saves to folder `smoke`, tests the untested train/merge/export/eval path), `3_train` (three bases, run after 2b).
Run order: 2b and 3s together, then 3.


## Update 23 October 2026 (evening)

- Run4 (40% abstain share): no gain, not adopted. Retrieval-score gate: one-for-one trade, not shipped (see the engineering log).
- Human review prepared: `docs/review/` holds the 50-answer sheet and the reviewer guide. A FUNAAB student and lecturer are being recruited.
- Next: end-to-end test of guard plus 4B model on CPU, larger test set split by document, container and model card, Qwen3 licence check.
