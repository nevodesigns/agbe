# AGBE, an offline agricultural extension model for 8 GB laptops

**Àgbẹ̀** is Yoruba for *farmer*. AGBE answers farming questions on an ordinary
laptop with the internet switched off: crops, pests, livestock, soils, storage
and getting produce to market.

- **Live site and build notes:** https://agbe-farm.vercel.app
- **Weights:** https://huggingface.co/NEVODESIGN/agbe-1b
- **Technical report:** [REPORT.md](REPORT.md)
- **Domain:** Agriculture · **Base:** Gemma 3 1B · **Runtime:** llama.cpp · **Quant:** GGUF Q4_K_M

---


**Demo video:** https://youtu.be/iQCgCXdy6Ww (105 seconds, real terminal sessions against the published weights)

## Try it in two commands

```bash
bash download_model.sh          # 814 MB, resumes if the connection drops
llama-cli -m model/agbe-1b-q4_k_m.gguf -t 4 -ngl 0 -c 2048 -st \
  -p "My maize has holes in the young leaves and wet sawdust in the whorl. What is this?"
```

Then disconnect your network and run it again. Nothing changes, which is the point.

---

## The problem

Nigeria has roughly one agricultural extension officer for every few thousand
farming households. The knowledge that would raise a smallholder's yield is not
secret and it is not new. It sits in extension manuals, and it does not travel
the last mile, because the last mile has no officer and often no signal.

The obvious answer is a farming chatbot, and it breaks the moment you look at
where farmers actually are. Rural coverage is patchy, mobile data is a real cost
paid from a thin margin, and a tool that needs the network is absent on the
morning the armyworm arrives.

**Offline is not a feature we added. It is the shape of the whole thing.** If it
cannot answer with the cable pulled, on hardware a farmer or co-operative already
owns, it does not count.

---

## Measured results

Official `adtc-profiler`, participant mode, on the target profile
(4 threads, `-ngl 0`, no GPU):

| Metric | Value |
|---|---|
| Throughput | **24.29 tok/s** (15.0 is the provisional reference; the real denominator is the fastest submission) |
| Peak RSS | **1,039 MB** |
| Steady RSS | 988 MB |
| Model file | 814 MB |
| `arc_easy` (50 samples) | **0.56** `acc_norm` |
| S_perf | **100.00** |
| S_eff | **85.50** |
| Engineering subtotal | **47.10 / 50** before thermal, **37.10** with the penalty |

> **Correction, checked against the official rules.** Earlier drafts of this
> document treated `S_perf` as capped: `100 × (TPS_act ÷ TPS_max)`. The
> challenge page states **`S_perf = 100 × (TPS_act ÷ TPS_max)`** with
> `TPS_REFERENCE = 15.0 provisional`, and the rules page says throughput is
> "evaluated relative to the maximum observed tokens per second". So 15.0 is a
> placeholder for the fastest submission, not a ceiling. Our 24.29 tok/s is
> therefore **not** a guaranteed 100: it is 100 only if nothing faster is
> submitted, and falls proportionally otherwise. Every engineering subtotal in
> this document assumes the provisional reference and is stated as such.

Raw output is committed as [`submission.json`](submission.json).

**Thermal, stated honestly.** Our development laptop (i7-10850H, thin chassis)
reaches 100°C and throttles during the profiler's 512-token prompt-processing
pass, so the participant run carries the flag. In generation alone at four
threads it peaks at 83°C. The profiler measures thermals again during audit
(`measured_on = "audit_cloud_vm"`), so the final penalty is determined by the
evaluation environment, not by this number. We are not claiming it will be zero.

---

## Why a 1B model, not the biggest that fits

The scoring function decides this, if you read it before writing code:

```
S_total = 0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal
S_perf  = 100 × (TPS_act ÷ TPS_max)      [15.0 provisional]
S_eff   = max(0, (7.0 − peak RAM GB) ÷ 7.0) × 100
```

Memory is charged linearly, so every gigabyte is paid for, and running the largest
model that fits in 8 GB is backwards on that term alone. We measured five
candidates rather than reasoning about them:

These are **selection-time measurements from our own harness**, taken before the
official profiler existed. They are what chose the model. The shipped figures
elsewhere in this README come from the official profiler on the final build and
differ: 24.29 tok/s, 1,039 MB, 47.10 points. Different instruments, different
numbers, and the profiler's are the ones that count.

> **Historical calculation, kept to document the decision.** The `S_perf` column
> below uses our original misreading, in which anything past the provisional
> 15 tok/s reference scored a flat 100. It is **not** the current ADTC score.
> Under the published formula, throughput is relative to the fastest submission,
> and the right-hand column changes a great deal. Both readings are set side by
> side in [REPORT.md](REPORT.md).

| Model | tok/s | Peak RAM | S_perf, as misread | S_eff | Points /50, as misread |
|---|---|---|---|---|---|
| Qwen2.5 0.5B | 46.6 | 0.50 GB | 100 | 92.9 | 48.57 |
| **Gemma 3 1B** | **26.9** | **0.88 GB** | **100** | **87.4** | **47.49** |
| Llama 3.2 1B | 24.4 | 1.26 GB | 100 | 82.0 | 46.40 |
| Qwen2.5 1.5B | 17.9 | 1.31 GB | 100 | 81.3 | 46.26 |
| Qwen2.5 3B | 11.5 | 3.26 GB | 76.7 | 53.4 | 33.69 |

A 3B concedes **13.8 points** under that reading, and **16.7** under the published
formula, before answering a single question.

---

## What it does, and what it refuses

**It will** identify a pest from field symptoms and tell you how to confirm it
before spending money; give spacing, timing and rotation advice for crops
actually grown here; cover poultry, goats, catfish, soils, drying and storage;
hold a conversation, so "what if I cannot afford that" gets a real answer.

**It will not** give an agrochemical dose (rates differ by product; a confident
wrong number is dangerous), quote a market price, advise on human health, or
pretend a virus has a cure.

Refusal is trained into the weights, not bolted on in a prompt, so it survives
`llama-cli` with no system prompt. **29%** of the corpus teaches safety
boundaries, honest uncertainty, or telling two confusable problems apart. Those
are three different behaviours and the number covers all three, not refusal
alone.

---

## The corpus

No dataset ships with this challenge, so the corpus is the substantive work.

**Provenance rule:** every training pair is composed from a curated fact base of
established extension practice ([`corpus/facts.json`](corpus/facts.json)). If a
fact is not in that file, it cannot appear in the corpus. Scraping the web or
having a large model write the answers would both have been faster, and both put
claims into training data nobody can trace. Agronomists notice invented
chemistry.

**956 conversations**, 7% multi-turn, 29% refusals, honest limits and discriminating pairs.

```
corpus/facts.json        curated fact base, the single source of truth
corpus/generate.py       composes conversations from facts
corpus/gold*.py          hand-written exemplars (refusals, Pidgin, corrections)
corpus/build/train.jsonl generated training set
eval/                    66-prompt behaviour + 92-prompt hostile batteries
train/train_lora.py      LoRA trainer, no trl
train/AGBE_train_kaggle.ipynb  full pipeline on a free Kaggle T4
```

---

## What the builds taught us

Every build was judged by **reading its answers**, not by its loss curve. The
loss curve looked healthy for every failure below.

| Build | Change | Result |
|---|---|---|
| v1 | r16, 3ep, 375 examples | Learned our answer scaffolding, not the agronomy. Told a parent to take a feverish child to an extension officer. |
| v2 | Removed repeated leads, 568 examples | Stapled unrelated facts together; the generator padded answers with other topics' facts. |
| v3 | Stopped cross-topic mixing | Refusal and timing correct. Pest wrong ("pod borers"). |
| v4 | 42 fall-armyworm mentions | Invented "fall army weevil". More examples do not fix a confusion. |
| v5 | **r32**, 5 epochs | Facts finally correct. Coherence broke: word salad, invented a pesticide called "dorabacite". |
| v6/v7 | **r32, 3 epochs** | English facts and refusal correct and stable. |
| v8 | Corrective exemplars from measured failures | Spliced oil palm spacing into a maize answer |
| v9-v12 | Sentence cap, planting calendars, adversarial exemplars, symptom-first diagnosis | See [BUILDS.md](BUILDS.md) |
| **v13** | Bidirectional contrast for confusable pests | **Shipped.** Highest behaviour-battery score of the final candidates (49/66) with zero safety leaks, and it names fall armyworm on `tp_001` where v11 and v12 said stem borer |

**Three separate axes, which took five builds to separate:**

- **Behaviours** (refusing, hedging) generalise from few examples. Fixed at rank 16.
- **Facts** (which pest, which symptom) need model *capacity*. Only fixed at rank 32.
- **Coherence** degrades with over-training, fixed by fewer epochs.

Four corpus iterations preceded any change to the training configuration. That
was the mistake: rank was the missing variable the whole time.

---

## Known limitations

- **Tail drift.** Answers are reliable for the first sentences and can add
  plausible, unsupported detail afterwards. Measured, not assumed: see
  [`eval/`](eval/).
- **Open-domain agronomy is weaker than the drilled topics.** It is strong on
  fall armyworm, refusals and post-harvest, weaker on topics thinly covered by
  the fact base.
- **English only.** Nigerian Pidgin was built, tested and **withdrawn**: it
  worked in one build and in the next named *amala*, a food, as a maize pest. We
  removed `pcm` from `language_scope` rather than ship a claim that fails half
  the time.
- It is a knowledgeable extension pamphlet that holds a conversation, not an
  agronomist, and not a vet or a doctor.

---

## Reproducing this

```bash
python corpus/generate.py                       # rebuild the corpus from facts
python train/train_lora.py --train corpus/build/train.jsonl --out out --merge
python eval/run_eval.py                         # 66-prompt behaviour battery
AGBE_PROMPTS=adversarial.jsonl AGBE_OUT=adversarial.json \
  python eval/run_eval.py                       # 92-prompt hostile battery
python tools/ledger.py --check                  # ledger vs the stored answers
adtc-profiler run --submission . --mode participant --output submission.json
```

`train/AGBE_train_kaggle.ipynb` runs the whole pipeline end to end on a free
Kaggle T4, including GGUF conversion and quantisation.

---

## Submission

`metadata.json` declares the domain, the two test prompts and the model. The
`.gguf` is not committed; `download_model.sh` fetches it from a public URL with
no credentials and verifies the length before accepting it.

Base model Gemma 3 1B, used under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). Inference on
[llama.cpp](https://github.com/ggml-org/llama.cpp).

**Nwokolo Victor Oluebubechukwu**, Lagos · Africa Deep Tech Challenge 2026
