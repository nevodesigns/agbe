## Inspiration

Agricultural advice can be unavailable at the moment a farmer needs it. Rural
coverage is unreliable, mobile data costs money, and an extension officer
cannot be physically present for every decision.

The obvious answer is a farming chatbot. But a cloud chatbot is useless at the
moment the network disappears.

So I built AGBE around the opposite constraint: download the model once, then
pull the cable and keep working.

## What it does

**AGBE** (*àgbẹ̀*, Yoruba for farmer) is an offline agricultural extension
assistant that runs on an ordinary 8 GB laptop with the internet switched off.

It covers crop diagnosis and management, pests and diseases, nutrient
deficiencies, livestock, soils, planting windows, drying and storage.

Its safety boundary matters as much as its answers. AGBE does not give
agrochemical doses, live market prices, or human medical advice. When a
question crosses that line, it refuses and points to the product label or a
human expert instead of inventing an answer.

## How I built it

I measured five candidate models against the target laptop profile instead of
choosing the largest one that would fit. The scoring formula charges memory
linearly, so a 3B model concedes about 14 engineering points before it answers
a single question.

I then iterated through **16 model builds**, changing LoRA rank, corpus
composition, sentence diversity, symptom-first diagnosis examples, contrast
examples for confusable pests and diseases, refusal behaviour and training
length. Every build was judged by reading its answers, not its loss curve,
because several builds with healthy loss curves still gave obviously wrong
advice.

The shipped build is **v16**: Gemma 3 1B, LoRA rank 32 and alpha 64, 3 epochs
and 117 optimiser steps on a free Kaggle T4, merged and quantised to GGUF
Q4_K_M for `llama.cpp`.

Every training example is composed by a script from a hand-curated fact base of
established extension practice. Nothing was scraped and nothing was generated
by a larger model, so every claim in the training data can be traced.

## What changed after Round 1

The Round 1 judges said the safety held but the agronomy was wrong. I traced
every wrong answer to a topic with **zero entries** in my fact base: cassava
brown streak, tomato blight, nitrogen deficiency and split fertiliser
application. The model was not confused; it had nothing to draw on and filled
the gap with fluent invention.

I added those topics, a differential-diagnosis slice that teaches confusable
pairs in both directions, and multi-turn examples, because judging is a live
chat. I also found the Round 1 model answered my own submitted test prompt with
"stem borer". v16 now names fall armyworm correctly.

Two rebuilds were rejected along the way. One supplied a fertiliser schedule for
a controlled crop; the next invented a pesticide dose. My own test batteries
caught both before anything was published.

## The final model

- **Base model:** Gemma 3 1B (`google/gemma-3-1b-it`)
- **Fine-tuning:** LoRA r32 / alpha 64, 3 epochs, 117 steps
- **Quantisation:** GGUF Q4_K_M, 814 MB
- **Runtime:** llama.cpp, CPU only
- **Language scope:** English
- **Corpus:** 1,237 conversations built from 1,499 unique sentences

## Measured result

Official `adtc-profiler`, participant mode:

- **24.08 tok/s**
- **1,039 MB peak RSS**, 992 MB steady
- **0.56 ARC-Easy**
- **48/66** on the behaviour battery
- **84/92** on the hostile battery
- **59/62** attacks withstood
- **0 safety leaks**

The provisional engineering calculation is **47.1/50 before the participant
thermal penalty**, and **37.1** once my laptop's thermal result is applied. The throughput term is scored relative to the fastest submission, so
15 tok/s is a provisional reference, not a ceiling.

## A result I threw away

I tested importance-matrix quantisation, calibrated on my own corpus. It scored
better on agronomy and quietly broke the safety: it offered a foreign suicide
helpline to a Nigerian farmer and called a common children's medicine toxic.
Better on the metric I was optimising, worse on the thing that matters. I built
both versions from the same source, measured them, and shipped the plain one.

## Safety

The hostile battery has **62 attacks**: prompt injection, authority claims,
roleplay, self-harm, illegal cultivation, poisoning, veterinary-to-human
crossover, obfuscation and emotional pressure. It also has **30 legitimate
questions** that exist only to measure over-refusal, because a model that ducks
anything containing the word "pesticide" would score well on safety and be
useless to a farmer.

v16 withstands **59 of 62 attacks** with **zero safety leaks**, the best result
of any build.

## What I learned

A small domain model is not improved by making the dataset bigger. Behaviour,
factual recall, coherence and sentence diversity fail independently. Adding
content to a 1B model eventually costs more than it buys: removing two
exemplars that did nothing fixed ten unrelated answers. The only reliable way to
see any of this was to read the outputs, classify the failures and change the
corpus around them.

## Model provenance

The repository includes a `provenance/` folder with the LoRA adapter, per-step
training logs, a run manifest, sha256 checksums of the base model, adapter and
final GGUF, the merge and quantisation script, and a before/after comparison
against the unmodified base model. The weights are pinned to an exact Hugging
Face commit so the submitted file cannot change.

## What's next

Field validation. AGBE should be used by extension officers and farming
communities over a real season, to learn which questions people actually ask
and where the model still fails.

Local languages are the biggest gap. I built Nigerian Pidgin, tested it, and
withdrew it from the declared scope rather than ship a capability that worked
one time in two.

## Reproducibility

Code, report, evaluation batteries, training notebook and download script are
public, and every number above comes from a tool in the repository.

**GitHub:** https://github.com/nevodesigns/agbe

**Model:** https://huggingface.co/NEVODESIGN/agbe-1b

**Training run:** https://www.kaggle.com/code/nevodm/notebookba584b3371

**Project site:** https://nevo-hackathon.pxxlspace.cv/agbe/

**Demo:** https://youtu.be/sevKpLqCt-E
