# AGBE documentation

AGBE (Àgbẹ̀, Yoruba for farmer) is an offline agricultural advisor for smallholder farmers and extension officers in West and Central Africa. This folder documents the whole project: what exists, why it was built that way, how to rebuild it, and what was measured, including what failed.

| Read this | For |
|---|---|
| [RECOVERY.md](RECOVERY.md) | One page to restart the project from nothing: where things live, key numbers, rebuild order, faults, next steps |
| [BUILD.md](BUILD.md) | The engineering build, stage by stage, with commands: data, claims, training examples, training, export, evaluation, serving |
| [DECISIONS.md](DECISIONS.md) | Every design and policy decision with its date and reason, including reversals |
| [MEASURES.md](MEASURES.md) | Every measured result, the test set it came from, and the negative results |
| [V2_ENGINEERING_LOG.md](V2_ENGINEERING_LOG.md) | The dated engineering log: each experiment, fault and correction as it happened |
| [V2_PLAN.md](V2_PLAN.md), [V2_HANDOFF.md](V2_HANDOFF.md) | The plan and the status board |
| [BAKEOFF_RESULTS.md](BAKEOFF_RESULTS.md) | The base-model bake-off |
| [AGRONOMIST_ROLE.md](AGRONOMIST_ROLE.md), [review/](review/) | Human review of answers: reviewer profile, guide and the 50-answer sheet |
| [../LICENCES.md](../LICENCES.md) | Which licence applies to which part of the repository |
| [devpost-story.md](devpost-story.md) | The competition write-up |

## Two lines of work

- **v16, the competition build.** Gemma 3 1B fine-tuned (LoRA), GGUF Q4_K_M, answers from its own memory. One of the top 10 finalists among 291 submissions in the Africa Deep Tech Challenge 2026 (it did not place in the top three). Frozen at git tag `adtc-2026-final` (commit `1d32e67` on `main`). The competition files at the repository root and in `eval/`, `corpus/`, `provenance/`, `figures/` belong to it.
- **v2, the current line (branch `agbe2`, the GitHub default).** Answers come from numbered, openly licensed source passages, each sentence is checked against them, and the model says so when no source covers the question. Trained on free Kaggle GPUs. Not yet shipped.

## Repository map

State: **frozen** = competition record, do not edit; **active** = v2 work; **data** = inputs for rebuilding; **reference** = explains the project.

| Path | What it is | State |
|---|---|---|
| `README.md`, `REPORT.md`, `BUILDS.md` | Competition README, report and build ledger | frozen (README carries the v2 status) |
| `FINAL.json`, `submission.json`, `metadata.json`, `download_model.sh`, `.template/` | Competition submission files and the checker's pristine template copy | frozen |
| `LICENSE`, `LICENCES.md` | The GPL-3.0 file and the map of every licence in the repository | reference |
| `corpus/` | v16 training corpus: hand-written fact base (`facts.json`), generator, gold sets, held-out file | frozen |
| `provenance/` | v16 training record: adapter, log, checksums, environment | frozen |
| `eval/` | Competition scorer, test batteries, per-version results (`results-v8` to `v16`), claim checker, guard measurements | frozen results, active `claim_check.py` and `measure_guard.py` |
| `figures/` | Score-curve figure and its data | frozen |
| `model/` | Empty folder; weights are never committed (`download_model.sh` fetches v16) | frozen |
| `data/sources/` | Registry of licence tiers and the list of 7,890 openly licensed items (metadata only) | data |
| `data/fulltext_v1/` | 8,062 passages from 279 CC-BY-4.0 documents, with `ATTRIBUTION.md` | data |
| `data/LICENCE_MANIFEST.md` | Licence of every source document | data |
| `data/claims/` | Early extraction pilot: samples and findings | data (pilot) |
| `data/run1_kaggle/` | Claims (1,765 proposed, 1,205 accepted), questions, training examples and logs from the first full run | data |
| `tools/` | Data pipeline: harvest, fetch and cut passages, extract and verify claims, build training examples, licence manifest, consistency checks | active |
| `train/` | Training, merge, GGUF export, scoring, and the generators for the Kaggle notebooks (`AGBE_v2_*.ipynb`) | active |
| `serve/` | The inference guard, retrieval, prompt shortening, backends and the FastAPI endpoint, with tests | active |
| `Dockerfile`, `docker-compose.yml` | Container recipe for `serve/` (never built yet) | active |
| `docs/` | This documentation | reference |
| `.github/workflows/tests.yml` | CI: all test suites and the guard measurement | active |

Kept out of the repository on purpose (see `.gitignore`): model weights, the 600 MB raw full-text download, correspondence with publishers, the superseded 70-document pilot, internal notes, video tooling. Tokens and passwords are never stored in any file.

## Rules for this repository

- Numbers are only quoted from a saved run. Negative results are recorded with the same detail as positive ones.
- Synthetic data is labelled synthetic.
- Public documents state the competition result as top 10 finalist among 291 submissions, not as a win.
- No em dashes; no AI tool named as an author.
