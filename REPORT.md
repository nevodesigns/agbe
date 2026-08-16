# Technical Report — AGBE, an offline farm advisor

**Team ID:** REPLACE_WITH_DEVPOST_TEAM_ID
**Domain:** agriculture
**Model:** AGBE-1B-Q4_K_M (Gemma 3 1B, LoRA r32, GGUF Q4_K_M)
**Weights:** https://huggingface.co/NEVODESIGN/agbe-1b
**Site and build notes:** https://agbe-farm.vercel.app

---

## Problem

Nigeria has roughly one agricultural extension officer for every few thousand
farming households. The knowledge that would raise a smallholder's yield is not
secret and it is not new. It sits in extension manuals, and it does not travel the
last mile, because the last mile has no officer and often no signal.

The obvious answer is a farming chatbot, and it breaks the moment you look at where
farmers actually are. Rural coverage is patchy, mobile data is a real cost paid out
of a thin margin, and a tool that needs the network is absent on the morning the
armyworm arrives.

AGBE answers questions about crops, pests, livestock, soils, storage and getting
produce to market, on an 8 GB laptop with the network off. Offline is not a feature
added at the end. If it cannot answer with the cable pulled, on hardware a farmer or
a co-operative already owns, it does not count.

The target user is a smallholder farmer or an extension officer in West and Central
Africa. The grounding is deliberately local: cassava mosaic, striga, fall armyworm,
aflatoxin, Newcastle disease, harmattan planting windows, and the crops actually
grown here.

---

## Design Decisions

### Base model: chosen by measurement, not instinct

The scoring function decides this, if you read it before writing code:

```
S_total = 0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal
S_perf  = min(TPS ÷ 15.0, 1.0) × 100
S_eff   = max(0, (7.0 − peak RAM GB) ÷ 7.0) × 100
```

Throughput above 15 tok/s earns **nothing**, and memory is charged linearly. The
instinct to run the largest model that fits inside 8 GB is therefore backwards. We
measured five candidates on the target profile:

| Model | tok/s | Peak RAM | S_perf | S_eff | Points /50 |
|---|---|---|---|---|---|
| Qwen2.5 0.5B | 46.6 | 0.50 GB | 100 | 92.9 | 48.57 |
| **Gemma 3 1B** | **26.9** | **0.88 GB** | **100** | **87.4** | **47.48** |
| Llama 3.2 1B | 22.8 | 1.26 GB | 100 | 82.0 | 46.39 |
| Qwen2.5 1.5B | 24.9 | 1.69 GB | 100 | 75.9 | 45.19 |
| Qwen2.5 3B | 11.5 | 3.26 GB | 76.4 | 53.4 | 32.94 |

The 3B concedes **14.5 points** before answering a single question, and would have
to be twenty-nine accuracy points better to break even. Gemma 3 1B was taken over
the 0.5B for about one point, on the judgement that a 0.5B would not hold enough
agronomy, and over Llama 3.2 1B at identical file size purely on memory.

### Quantization: Q4_K_M

Q4_K_M gives 814 MB on disk and 0.88 GB peak RAM. Gemma 3 1B has an embedding
dimension of 1152, which is not divisible by 256, so `llama.cpp` falls back to
`q5_0` and `q8_0` on several tensors. That is expected, applies equally to the
reference build, and is why the file is larger than a naive Q4 estimate.

### Thermal: a result that runs backwards

Our first clean run lost the full ten-point penalty at 91°C. Sweeping thread counts:

| Threads | tok/s | Peak °C | Penalty | Points |
|---|---|---|---|---|
| 2 | 20.8 | 99.0 | −10 | 36.39 |
| 3 | 25.8 | 99.0 | −10 | 36.39 |
| **4** | **20.6** | **83.0** | **0** | **46.39** |
| 6 | 25.9 | 84.0 | 0 | 46.39 |

**Fewer threads ran hotter.** With two or three cores loaded the CPU boosts toward
its single-core turbo ceiling and per-core temperature spikes; at four or more the
all-core power limit caps clocks and spreads the same work cooler. Because `S_perf`
caps at 15 tok/s, surplus throughput can be traded for thermal headroom at no cost
to the score. Ten points recovered from a quirk of the formula.

### Alternatives considered and rejected

- **Qwen2.5 1.5B**, tried late to buy accuracy with capacity. Abandoned: training
  produced `grad_norm: nan` on step one and a null adapter, because we load fp16
  and Qwen2.5 is bf16-trained and overflows on Turing; and GGUF conversion is
  separately blocked by a `transformers` 5.0 bug in Qwen tokenizer handling. Two
  debug cycles to chase 2.29 points we would then hand back, on a 35% larger
  download.
- **`trl`'s SFTTrainer**, dropped after it broke inside its own chunked
  cross-entropy path on a PEFT-wrapped causal LM. Replaced with plain
  `transformers.Trainer`, which removed a dependency and made label masking
  explicit and verifiable.
- **Scraping or distilling the corpus from a larger model**, rejected on
  provenance grounds.

---

## Constraints

- **Target:** 8 GB RAM, integrated graphics, Ubuntu 22.04, four cores
- **CPU only.** No GPU offload (`-ngl 0`), pure `llama.cpp`
- **Offline.** Zero network calls during inference. No API key, no account
- **Connectivity is the design constraint, not a feature.** The users with the
  least extension coverage also have the least signal
- **Data cost matters.** 814 MB is downloaded once; nothing recurs

---

## The corpus

No dataset ships with this challenge, so the corpus is the substantive work.

**Provenance rule:** every training pair is composed from a curated fact base of
established extension practice. If a fact is not in that file, it cannot appear in
the corpus. Scraping the web or having a large model write the answers would both
have been faster, and both put claims into the training data that nobody can trace.
This domain is graded by agronomists who notice invented chemistry.

**Safety rule:** no fabricated yields, no prices, and above all **no agrochemical
doses**. Where a real answer needs a rate, the model is trained to point at the
product label and the local extension officer. A confident wrong dose is more
dangerous than an admission of ignorance.

Final corpus: **642 conversations**, 9% multi-turn (judges chat live, so follow-ups
are trained explicitly), 6.7% refusals and honest limits.

---

## What six builds taught us

Every build was judged by **reading its answers**, not by its loss curve. The loss
curve looked healthy for every failure below.

| Build | Change | Result |
|---|---|---|
| v1 | r16, 3ep, 375 examples | Learned our answer scaffolding, not the agronomy. 47% of answers opened with one of six sentences. Told a parent to take a feverish child to an extension officer. |
| v2 | Removed repeated leads, 568 examples | Stapled unrelated facts together, because the generator padded answers with other topics' facts. |
| v3 | Stopped cross-topic mixing | Refusal and timing correct. Pest wrong ("pod borers"), Pidgin answered in English. |
| v4 | 42 fall-armyworm mentions | Invented "fall army weevil". More examples do not fix a confusion. |
| v5 | **r32**, 5 epochs | Facts finally correct. Coherence broke: word salad, and it invented a pesticide called "dorabacite". |
| **v6/v7** | **r32, 3 epochs** | English facts and refusal correct and stable. Shipped. |

**Three separate axes, which took five builds to separate:**

- **Behaviours** (refusing, hedging, admitting no cure exists) generalise from few
  examples. Fixed at rank 16 and stable ever since.
- **Facts** (which pest, which symptom) need model *capacity*. Only fixed when LoRA
  rank doubled to 32. Adding examples made it worse, not better.
- **Coherence** degrades with over-training, and is fixed by fewer epochs.

Four corpus iterations preceded any change to the training configuration. That was
the mistake: rank was the missing variable the whole time.

---

## Benchmarks

Measured on an i7-10850H held to the Standard Laptop profile: four threads,
`-ngl 0`, memory capped to 7 GB, from a cooled machine.

| Metric | Value |
|---|---|
| Machine | i7-10850H, Ubuntu 22.04.5 |
| Model file | 814 MB (Q4_K_M) |
| RAM at peak | 0.88 GB |
| Generation speed | 22 to 27 tok/s |
| Peak core temperature | 83°C at four threads |
| Thermal throttling | None at four threads |
| Engineering points | **47.48 / 50** |

These are self-reported development benchmarks. Official scores are measured by the
ADTC profiler on the standard evaluation machine.

---

## Honest limits

- **Tail drift.** Answers are reliably correct for the first sentences and can then
  add confident invention, for example attaching "there is no spray that cures an
  infected plant" (true of cassava mosaic) to fall armyworm. This is the main
  remaining defect and it is not solved.
- **Nigerian Pidgin was withdrawn.** It worked in v5 and in v6 named *amala*, a
  food, as a maize pest. `language_scope` is `["en"]` rather than shipping a
  capability that works in one build and not the next. The African use-case claim
  is unaffected, because it rests on the domain, not the language.
- **A 1B model is a knowledgeable extension pamphlet that can hold a
  conversation.** It is not an agronomist, not a vet, and emphatically not a
  doctor. Where a real extension officer is available they are the better answer,
  and AGBE is trained to say so.

---

## Reproducing

```bash
git clone https://github.com/nevodesigns/agbe && cd agbe
bash download_model.sh          # verifies length, resumes on drop
llama-cli -m model/agbe-1b-q4_k_m.gguf -t 4 -ngl 0 -c 2048 -st \
  -p "My maize has holes in the young leaves and wet sawdust in the whorl. What is this?"
```

Corpus generation is `python corpus/generate.py`; training is
`python train/train_lora.py --merge`; and `train/AGBE_train_kaggle.ipynb` runs the
whole pipeline end to end on a free Kaggle T4.
