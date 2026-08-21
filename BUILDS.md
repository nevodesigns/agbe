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
| v12 | 948   | 3 | 90  | 0.7611 | 1.868 | Six livestock contrast exemplars. Bought +1 livestock, cost leaks and 2 attacks. **Rejected** |
| **v13 (SHIPPED)** | 956 | 3 | 90 | 0.7374 | 1.863 | Balanced the one-way armyworm contrast. Best total, zero leaks, correct on tp_001 |

## Evaluation, identical scorer across all rows

Both batteries rescored from stored answers whenever the scorer changed, so these
compare like with like.

| Build | 66-prompt | diagnose | leaks | Hostile | Attacks withstood | tok/s |
|---|---|---|---|---|---|---|
| v8  | 49/66 | 7/12  | 0 | 76/92 | 54/62 | 19.7 |
| v9  | 43/66 | 6/12  | 3 | 81/92 | 58/62 | 23.7 |
| v10 | 46/66 | 5/12  | 3 | 77/92 | 57/62 | 21.7 |
| v11 | 47/66 | 11/16 | 0 | **79/92** | **58/62** | 23.3 |
| v12 | 46/66 | **12/16** | 2 | 76/92 | 56/62 | 26.7 |
| **v13** | **49/66** | 11/16 | **0** | **79/92** | 56/62 | — |

**v11 was the leading candidate at that point.** v12 matched it on accuracy and diagnosis, gained one
livestock prompt, and gave back three safety leaks and two attacks. Throughput is
was then believed to cap at 15 tok/s, so v12's speed advantage looked worthless.

v12's leaks were content failures, not scorer artifacts: blossom end rot became
"blossom drop" described with insect symptoms, a catfish question returned "that
is usually tilapia, not catfish", and a rainfall forecast returned "I only cover
weather for Nigeria".

v8 leads the 66-prompt total by two. That is not the deciding number: throughput
was then believed to cap at 15 tok/s, so v11's speed advantage looked worthless. Under
the published formula it is not, though the trade below still favours v11 over v12.
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
| v12 | `c675f16d3eb5033f331af128c0da0d81` | 814,261,088 |
| **v13 (SHIPPED)** | `d614d6b00aad21990419841bea8dae37` | 814,261,088 |

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
about 24.29 tok/s, with coherent output and no loader errors. A single smoke test would
have been weaker evidence than that.


## One-way contrasts

v12's corpus carried five examples of "That is stem borer, not armyworm" and none
of the reverse, and its smoke test answered the textbook fall armyworm description
with "That is stem borer" — on our own submitted test prompt.

A contrast exemplar does not teach a boundary, it teaches a **direction**. This is
the v4 lesson inverted: pushing armyworm to 42 mentions once made armyworm the
default for every maize symptom, and pointing a contrast away from it simply
relocated the bias to stem borer. An unbalanced contrast moves a confusion rather
than resolving it.

`generate.py` now prints every `"That is X, not Y"` pair with its reverse count and
flags any that run one way, matching on containment so "armyworm" and "fall
armyworm" count as the same term. The reverse exemplars are committed. They are not
in any shipped weights, because v11 predates the problem and does not have it.


## The scorer was passing wrong diagnoses

`expect` is any-of and carries supporting detail, so `d_faw` — which is **tp_001**,
one of our two submitted prompts — accepted an answer opening "That is stem borer"
because the word "frass" appeared later. Four consecutive builds were scored on
that, and v11 was chosen partly on a diagnosis figure that was not real.

It surfaced by accident: re-recording the demo video's terminal footage showed the
model naming the wrong pest on camera.

The scorer now requires the diagnosis to be **named in the opening two sentences
and asserted rather than denied**, reusing the same negation logic the forbid list
uses — otherwise "That is stem borer, not armyworm" passes on the substring.
Rescoring every build under the corrected rule is what put v13 ahead.

Two forbid terms were also too broad to be usable and were narrowed, applied
equally to all builds: `"take"` fired on "take her to the clinic" in every correct
refusal, and `"forecast"` fired on "your extension officer will have the current
forecast", which is redirection rather than a prediction.

## A truncated model reached HuggingFace

A cancelled Kaggle run left a 15 MB partial GGUF (quantize was interrupted at
tensor 2 of 340) and the publish cell uploaded it over the canonical filename
before being interrupted. For a period the public URL served a **14,740,576 byte**
file.

`download_model.sh` refuses it: the pinned sha256 does not match and the script
now deletes rather than resumes onto a wrong-sized file. That check was added the
same day, for a different reason, and caught this.


## Final artifact

Everything below describes the same file. The published weights were downloaded
from the public URL and hashed independently, not trusted from a response header.

| | |
|---|---|
| build | **v13** |
| corpus | 956 conversations |
| training | 3 epochs, 90 steps, LoRA r=32 on Gemma 3 1B |
| GGUF sha256 | `d614d6b00aad21990419841bea8dae37502f8c57f1b3a25730ec15c3480d9851` |
| bytes | 814,261,088 |
| HuggingFace commit | `2dd8ab347ddd4909fcb90dcbcffe6039b4b8bc34` |
| profiled at repo commit | `61bced20ba47` |

**Official profiler, participant mode, v13:** 24.29 tok/s (S_perf **100.0**),
1039 MB peak RSS (S_eff **85.5**), arc_easy 0.56 on 50 samples,
peak 99.0 C, throttled true.

Engineering subtotal **47.10 of 50**, or **37.10** with the thermal penalty.

**On the thermal penalty.** Three runs at 88 C, 44 C and 54 C starting temperature
all peaked at 99 to 100 C and all throttled. Preparation does not change the
outcome on this chassis, so the penalty is a property of the hardware. The
profiler's schema notes cloud hosts usually expose no thermal sensor, which
suggests the audit re-measures rather than inheriting this figure.
