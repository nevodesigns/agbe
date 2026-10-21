# AGBE v2 plan

> Current state and the faults found since this plan was written: see [V2_ENGINEERING_LOG.md](V2_ENGINEERING_LOG.md).

AGBE v16 (tag `adtc-2026-final`) is the competition record. This branch is the next version: a grounded, verifiable, deployable offline farm advisor.

## Targets (to be tightened once the expert gold set exists)

| Measure | Target |
|---|---|
| Agrochemical dose leaks | 0 |
| Claim-level factual accuracy on questions our sources cover | 90% or more |
| Unsupported claims in an answer | under 3% of claims |
| Over-refusal on legitimate questions | under 5% |
| Edge profile | about 1 GB peak RAM, 25 or more tokens a second on 4 CPU cores, no network |
| Server profile | a larger model from the same fact base, served behind an API |

## The system

1. **Verified fact graph.** Atomic claims extracted from licensed sources. Every claim carries its source, location and licence. Each quoted span must appear verbatim in the source (checked by code). An agronomist audits a sample.
2. **Model.** Base-model bake-off on the gold set, then staged training: grounded SFT, preference training on supported versus unsupported answers, quantisation-aware export.
3. **Inference guard.** Retrieval with citations, a claim-level verifier, and an "answer, then stop" policy.
4. **Production layer.** OpenAI-compatible endpoint plus an `/advise` endpoint returning citations and confidence, container image, auth and quotas, versioned releases, evaluation gate in CI, model card, data-licence manifest.

## Data licence tiers

- **Tier A (training and retrieval):** items that state CC-BY-4.0, CC-BY-3.0, CC0-1.0 or public domain.
- **Tier B (evaluation only, never trained on):** NC and SA licences.
- **Tier C (after written permission):** items with no stated licence, FAO, CABI, NAERLS and similar. See `docs/outreach/permission-emails.md`.

Registry and rules: `data/sources/`. Facts are not copyrightable, but this project does not rely on that: it writes in its own words, cites, and stays inside the licences. This is not legal advice.

## Compute

Free tiers only for now: Kaggle (T4, weekly GPU quota) and Colab free. A 1B model trains in about twenty minutes on a T4. A 3B to 4B model needs QLoRA and shorter runs, and is the first thing the bake-off tests.

## Order of work

1. Source registry and Tier A harvest (started).
2. Expert gold set, 500 questions (template in `eval/gold/`), and the claim-level scorer (baseline in `eval/claim_check.py`).
3. Extraction pipeline: full text of Tier A items to atomic, quote-verified claims.
4. Base-model bake-off on the gold set, free-tier compute.
5. Staged training and quantisation-aware export.
6. Retrieval and the inference guard.
7. Serving, packaging, CI gate, model card.
8. Pilot with an extension service.

An agronomist is the single biggest lever. See `docs/AGRONOMIST_ROLE.md`.

## Status (18 Oct 2026)

- Done: licence-tiered registry; 4,785 Tier A items listed; full-text fetch and passage cutting (70 documents, 1,974 passages); claim verifier with tests; LLM extraction pilot (see `data/claims/README.md`); claim-level scorer; the competition scorer exposed as `eval/run_eval.score()` and checked against 994 stored rows with no differences; base-model bake-off script with a CPU dry run; Kaggle guide (`train/KAGGLE_BAKEOFF.md`).
- Next, needs a GPU session: run the bake-off on Kaggle, and run claim extraction with a 7B model over the Tier A passages.
- Next, needs a person: an agronomist to audit claims and write the gold set; sending the permission emails.

## Status (18 Oct 2026, later)

- Deeper harvest merged: 7,890 Tier A items, 2,962 from IITA.
- Permission emails written in full with verified addresses (`docs/outreach/permission-emails.md`).
- `serve/` built and tested (14 tests): guard, retriever, backends, FastAPI endpoint with an OpenAI-compatible route and API-key auth.
- Finding: v16 ignores retrieved sources, so retrieval needs a model trained to use them. See `serve/README.md`.
- A larger full-text fetch (350 items) was running in the background.

## Status (18 Oct 2026, night)

- Guard rules extended and measured (53/62 dev attacks, 28/30 holdout, 0 false blocks). CI green.
- Grounded training-data builder and a GPU trainer script written (trainer untested). Permission emails sent; IITA Cc bounced.
- Everything left before retraining needs either a GPU session (bake-off, 7B extraction, question generation) or a person (agronomist, replies from the data owners).
