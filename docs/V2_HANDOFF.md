# AGBE v2 handoff

Read this first if you are picking the work up cold, whether you are me in a new session or a person. It records what exists, why, what is verified, what is not, and what to do next. Update it when something meaningful changes. Last updated 18 October 2026 (evening).

## 0. Status board (19 October 2026, end of day)

| Item | State |
|---|---|
| v16 competition record | Frozen at tag `adtc-2026-final` (commit 1d32e67). ADTC finished 17 Oct. The live-event result is not recorded here. |
| Tier A data | 7,890 items listed; 8,062 passages from 279 documents in `data/fulltext_v1/` |
| Guard, endpoint, evaluation, CI | Built and tested on CPU (all unit tests pass, CI green) |
| Kaggle notebook 1 (bake-off) | Done. Results in `docs/BAKEOFF_RESULTS.md` |
| Kaggle notebook 2 (claims, questions, training data) | Was running (about 168 of 590 passages when last seen). Outputs save to the private Hugging Face repo `NEVODESIGN/agbe-v2-work` |
| Kaggle notebook 3 (train three bases, export GGUF, held-out test) | Ready, not run. Re-import the newest file after notebook 2 finishes |
| Trained v2 model | None yet |
| Permission emails | Sent 18 Oct. IITA Cc bounced. Follow up on or after 1 Nov 2026 |
| Agronomist | Not found yet |
| Untested on a GPU | batched extraction, question generation, `train_grounded.py`, `merge_adapter.py`, `export_gguf.sh` |

The complete private record (people, accounts, business arrangements, all file locations) is kept outside this repository.

## 1. Where things stand

AGBE v16 (1B Gemma, GGUF Q4_K_M) was submitted to the Africa Deep Tech Challenge 2026 and reached the final (10 of 291 submissions). The competition is over. The goal now is a production-grade, deployable, grounded successor that stands up against global entries: a model, an endpoint, and a system that can run as a local LLM or as a feature inside another product.

Owner: Nwokolo Victor Oluebubechukwu, solo, Lagos. Public contact: nwokoloebube21@gmail.com.

## 2. Decisions the owner made (17 Oct 2026)

- **Data: Tier A only for now.** Tier A means the item states CC-BY-4.0, CC-BY-3.0, CC0-1.0 or public domain. Anything else is not trained on.
- **Compute: free tiers only.** Kaggle (T4) and Colab free.
- **Agronomist: none yet.** The owner will look. Profile in `docs/AGRONOMIST_ROLE.md`. The expert gold set cannot start without one.
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

| Path | What it is | State |
|---|---|---|
| `docs/V2_PLAN.md` | Targets, architecture, order of work, status | current |
| `docs/AGRONOMIST_ROLE.md` | Who to recruit and what they do | current |
| `docs/outreach/permission-emails.md` | Permission emails: IITA main, CABI, FAO and NAERLS variants. Copy in `~/Downloads/AGBE-permission-emails.md`. They say AGBE was one of 10 finalists from 291 submissions. | drafted, NOT sent |
| `data/sources/` | `registry.json`, `README.md`, `cgspace_tier_a.jsonl.gz` (4,785 items, metadata only), `cgspace_summary.json` | done |
| `tools/cgspace_harvest.py` | Lists Tier A items through the CGSpace API, in parallel. Sample of 100 x pages per topic. | done, deeper run in progress |
| `tools/fetch_fulltext.py` | Re-checks each item's licence live, downloads the PDF, converts it (reading order), cuts it into cited passages | works |
| `tools/extract_claims.py` | Turns passages into verified claims. Backends: extractive, transformers, openai, llamacli. | works, 9 tests pass |
| `data/fulltext_pilot/` | 70 documents, 1,974 passages, `ATTRIBUTION.md` | pilot |
| `data/claims/` | Pilot claims and findings (`README.md`) | pilot |
| `eval/claim_check.py`, `eval/test_claim_check.py` | Claim-level answer scorer (baseline, lexical), 6 tests pass | baseline |
| `eval/run_eval.py` | The competition scorer. `score(row, answer)` is now a function, verified equal to the stored results on 994 rows | current |
| `eval/gold/` | Expert gold-set schema, stub template, README. No stub is reviewed. | needs an agronomist |
| `serve/` | The inference guard (hard rules, retrieval, claim check, stop policy), BM25 retriever, llama.cpp and OpenAI-style backends, FastAPI app with `/advise` and `/v1/chat/completions`, 14 tests. See `serve/README.md`. | works, tested |
| `train/bakeoff.py`, `train/KAGGLE_BAKEOFF.md` | Base-model bake-off, plain and with retrieved sources. CPU dry run passes. | ready, needs a GPU session |

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
- **Outreach**: the four permission emails were sent by the owner on 18 October. The IITA Cc e.ezomo@cgiar.org bounced. Follow up on or after 1 November 2026.
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

## 7. For the owner (things only a person can do)

- Send the permission emails (now complete, with verified addresses, in `docs/outreach/permission-emails.md`). Confirm the Cc addresses flagged there first. Known: FAO copyright@fao.org (and the licence request form), CABI enquiries@cabi.org. Find IITA's and NAERLS's contact pages.
- Find an agronomist.
- Run the Kaggle notebooks.

## 8. Related, outside this repo

- AGBE product site: https://agbe-omega.vercel.app (repo `nevodesigns/agbe-site`, deployed with `vercel deploy --prod --yes`). Still stale: `agbe-farm.vercel.app`, `agbe-site.pxxlspace.cv`, and the `/agbe/` mirror in the `nevo-hackathon` hub.
- Shared memory notes for the assistant: `~/.claude/projects/-home-nwokolo-projects-agbe/memory/` (see `agbe-v2-plan.md`).

## 9. Gotchas learned the hard way

- The CGSpace API is slow (40 to 60 seconds for a 100-item page). Use 4 parallel workers, not more, and run long jobs with `nohup` in the background.
- `pkill -f "<command text>"` inside a shell command kills that shell, because the text matches itself. Use a bracketed pattern such as `pgrep -f "[c]gspace_harvest"`.
- Scratch directories can be cleared between sessions. Keep real work in the repo.
- Never commit full-text PDFs or large generated files. `data/fulltext/` and the plain `.jsonl` copies are git-ignored. The committed pilot passages carry an attribution file.
- Gemma and Llama weights are gated on Hugging Face. Accept the terms before a Kaggle run.
- Run tests with `cd eval && python3 -m unittest test_claim_check` and `cd tools && python3 -m unittest test_extract_claims`.
