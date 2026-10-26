# Measures

All v2 results are on the same 140 held-out cases (70 grounded, 70 abstain), held out by passage hash. Of the 70 abstain cases, 29 are answerable from other documents; abstention is reported on the other 41 ("clean"). The numbers are single runs: with 70 and 41 cases, one case moves a score by 1.4 and 2.4 points, so differences of one or two cases are noise.

## How a case is scored

Grounded pass: the model answers (does not abstain), every citation is valid, at least 80% of its sentences are supported by the shown passages. Abstain pass: the model says it has no source and writes little else. Wrong abstention: the model abstains on a grounded case. The support check is the project's claim checker (`eval/claim_check.py`), a lexical checker; the guard uses the same one, so the guard's support figure is not independent evidence.

## Models

| Model and file | Grounded pass | Supported | Right source | Wrong abstain | Clean abstain | Memory | Speed vs v16 |
|---|---|---|---|---|---|---|---|
| Qwen3 1.7B, run1, fp16 | 87.1% | 89.5% | 92.9% | 4.3% | 51.2% | | |
| Qwen3 1.7B, run2, fp16 | 94.3% | 96.7% | 95.7% | 1.4% | 48.8% | | |
| Qwen3 1.7B, run2, Q4 GGUF, whole passages | 91.4% | 96.4% | 95.7% | 0.0% | 51.2% | 2,042 MB | 0.73x |
| Qwen3 1.7B, run2, Q4 GGUF, short prompt (3 x 100 words) | 90.0% | 91.7% | 90.0% | 2.9% | 58.5% | 2,041 MB | faster than whole passages (19.0 s against 26.6 s) |
| Qwen3 1.7B, run3 (trained on short prompt), fp16 | 91.4% | 91.4% | 84.3% | 7.1% | 41.5% | | |
| Qwen3 1.7B, run4 (abstain share 0.40), fp16 | 97.1% | 97.1% | 95.7% | 0.0% | 46.3% | | |
| Qwen3 4B, Q4 | 90.0% | | | | 85.4% | 4.5 GB | 0.35x |
| Gemma 3 1B, Q4 | 72.9% | | | | 43.9% | 1.0 GB | 1.00x |

Untrained bases scored 7.1% (Gemma 3 1B), 24.3% (Qwen3 1.7B) and 31.4% (Qwen3 4B). v16 on the same source-grounded test scores 0%, because it ignores sources. v16 runs at 13.87 tokens per second on a Kaggle Xeon 2.2 GHz and 24.08 on the competition machine, with 1,039 MB memory.

Current edge candidate: run2, Q4, short prompt. Its weak point is abstention: 58.5% on clean cases; the 4B model reaches 85.4%.

## Experiments that did not help

| Date | Experiment | Result |
|---|---|---|
| 20 Oct | Abstain examples with same-document hard negatives | Abstention worse; abandoned |
| 23 Oct | Training on the short prompt (run3) | Lower pass, right-source and abstention |
| 23 Oct | Abstain share 0.40 (run4) | Clean abstain 46.3% against 48.8%; grounded 97.1% against 94.3% (within noise) |
| 23 Oct | Retrieval-score gate (best-sentence word coverage) | Threshold 0.30: +2 abstentions, -2 grounded answers; 0.40: +4, -10 |

## Guard

Hard rules stop 53 of 62 development attacks and 28 of 30 holdout attacks, with none of the 30 and 22 legitimate questions blocked (`eval/measure_guard.py`; the holdout was written before the rules were extended). The guard's sentence check raises grounded pass by 1 to 3 points, but its support figure is 100% by construction.

## Data

7,890 Tier A items listed; 8,062 passages from 279 documents; 1,765 claims proposed, 1,205 accepted; training set 868 grounded, 200 abstain, 196 refusal examples (run1 and run2); run4 set 662, 327, 185.

## Not yet measured

Human agreement with the answers (a 50-answer review sheet is in `docs/review/`), a test set split by document, the guard wired to the real model end to end, the 4-bit file of runs 3 and 4, and any measurement on real farmer questions.
