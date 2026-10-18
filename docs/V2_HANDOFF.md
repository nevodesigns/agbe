# AGBE v2 handoff

Read this first if you are picking the work up cold, whether you are me in a new session or a person. It records what exists, why, what is verified, what is not, and what to do next. Update it when something meaningful changes. Last updated 18 October 2026 (evening).

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

## 6. Resume here (in order)

1. **Deeper harvest: DONE** (merged 18 Oct, 7,890 items). Originally started 18 Oct with `python3 tools/cgspace_harvest.py --pages 5 --out data/sources/cgspace_tier_a_deep.jsonl --summary data/sources/cgspace_summary_deep.json`. About 85% done when this was written (log at `/tmp/claude-1000/harvest3.log`, may be gone). If the files exist, merge them into the registry, gzip the jsonl, update `data/sources/README.md` and `registry.json`, commit. If not, rerun the command (about 2 hours).
2. **Bake-off on Kaggle.** The owner runs `train/KAGGLE_BAKEOFF.md` and sends `summary.md`. Choose the top two models from that, not from any assumption.
3. **Claim extraction at scale on Kaggle.** `python3 tools/extract_claims.py --backend transformers --model Qwen/Qwen2.5-7B-Instruct --passages passages.jsonl --out claims.jsonl`. Fetch more full text first (`tools/fetch_fulltext.py --all --limit 2000`, or `--pilot N` for a balanced sample). Prefer practical documents (manuals, briefs, extension material).
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
