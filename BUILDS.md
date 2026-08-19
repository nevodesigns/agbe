# Build ledger

Every trained candidate, identified by the artifact rather than by a name. This
file exists because two numbers got close to being conflated: v11 trained on a
**924**-conversation corpus, and the corpus was then grown to **948** for the
following run. A report that credits v11's results to a 948-example corpus would
be wrong, and it is exactly the kind of small inconsistency that makes a report
look careless.

Rule: a row is only filled in from a training log that actually cloned that
corpus. Evaluation numbers are only recorded against a GGUF whose sha256 matched
the published file at download time.

## Loss columns, because these are two different things

`train_loss` is the run-level aggregate the trainer reports at the end, averaged
over every step including the very high early ones. The **last logged interval
loss** is the final windowed value, which is what indicates where the model
actually converged. They are not comparable to each other, only across runs.
Earlier drafts of the report quoted the interval loss and called it "final loss",
which was imprecise.

| Build | Corpus | Epochs | Steps | Last interval loss | train_loss | Outcome |
|---|---|---|---|---|---|---|
| v8  | 1,020 | 3 | 96  | 0.29   | 1.51  | Answered "when should I plant maize" with oil palm spacing |
| v9  | 746   | 3 | 72  | 1.107  | 2.119 | Safest to date, lost facts (blossom end rot → "bacterial wilt") |
| v10 | 812   | 4 | 104 | 0.38   | 1.556 | Recovered some facts, invented "mortjacket" and "Scarets" |
| v11 | 924   | 3 | 87  | 0.7656 | 1.909 | Diagnosis 10/12, zero leaks, 94% attack resistance |
| v12 | 948   | 3 | —   | —      | —     | v11 plus six contrast exemplars for confusable livestock pairs |

## Evaluation, identical scorer across all rows

Both batteries rescored from stored answers whenever the scorer changed, so these
compare like with like.

| Build | 66-prompt | diagnose | leaks | Hostile | Attacks withstood | tok/s |
|---|---|---|---|---|---|---|
| v8  | 49/66 | 7/12  | 0 | 76/92 | 54/62 | 19.7 |
| v9  | 43/66 | 6/12  | 3 | 81/92 | 58/62 | 23.7 |
| v10 | 46/66 | 5/12  | 3 | 77/92 | 57/62 | 21.7 |
| v11 | 47/66 | **10/12** | **0** | 79/92 | **58/62** | 23.3 |

v8 leads the 66-prompt total by two. That is not the deciding number: throughput
is capped at 15 tok/s for scoring so v11's speed advantage is worth nothing, and
the trade is +3 diagnosis and +4 attacks against 2 livestock prompts. Symptom-first
diagnosis is the likeliest hidden prompt there is, so the trade is worth taking.

## Artifact identity

The shipped GGUF must match this hash. `download_model.sh` verifies it, and it is
what the profiler was run against.

| Build | sha256 (first 32) | Bytes |
|---|---|---|
| v9  | `77a760fba0b01d0335dceba07775a060` | 814,261,088 |
| v10 | `802bbabf844da09ed34c6a56e39557ff` | 814,261,088 |
| v11 | `f18c01f2410958c2a894281b38088722` | 814,261,088 |

## The export environment is load-bearing

GGUF conversion depends on **transformers 4.57.x**, and that dependency is
invisible until it breaks.

llama.cpp's `requirements-convert_hf_to_gguf.txt` pins transformers *down* from
the 5.0.0 that Kaggle ships. Under 5.0.0, `GemmaTokenizer` injects
`<image_soft_token>` at id 262144 while the text-only 1B declares `vocab_size`
262144, so conversion dies on `assert max(tokenizer.vocab.values()) < vocab_size`
after writing all 340 tensors, leaving no file behind. Under 4.57.6 it does not.

The v12 build lost that pin by accident. A change intended to stop llama.cpp
replacing the GPU torch wrote the filtered requirements file to `/tmp`, which
broke a **relative include** inside it (`-r ./requirements-convert_legacy_llama.txt`).
pip aborted the whole install, `%%capture` hid the error, transformers stayed at
5.0.0, and conversion failed.

**Five fixes were then written for a tokenizer that was never broken.** Its base
vocab was correct at max id 262143 in every diagnostic. The token cannot be
removed from the config side at all: deleting the key lets a hardcoded class
default take over, and remapping it leaves the string in `all_special_tokens`.
A sixth change pinned llama.cpp to `5112b97` on the false premise that it predated
the converter refactor. It did not, and that was asserted without checking.

Guards now in place:

- the filtered requirements file stays inside `requirements/` so the include resolves
- the notebook asserts `transformers.__version__` starts with `4.` right after the install
- the trainer aborts rather than warns when CUDA is missing
- llama.cpp is pinned, which is still correct practice even though it was not the fix

The general lesson, which the eval work got right and this did not: **measure
before fixing.** The diagnostic that named the cause took two minutes to write and
was run only after five failed attempts.

## GGUF export warnings, checked not assumed

Conversion emits `Unknown RoPE type: default` and several
`Duplicated key name 'gemma3.*'` warnings. Both are benign here, and that is a
measurement rather than an assumption: **158 prompts were generated through
`llama-cli` against the exact published v11 GGUF** across the two batteries, at
23 tok/s, with coherent output and no loader errors. A single smoke test would
have been weaker evidence than that.
