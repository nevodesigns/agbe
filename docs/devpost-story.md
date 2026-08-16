## Inspiration

Nigeria has roughly one agricultural extension officer for every few thousand farming households. The knowledge that would raise a smallholder's yield is not secret and it is not new. It sits in extension manuals, and it does not travel the last mile, because the last mile has no officer and often no signal.

The obvious answer is a farming chatbot. That breaks the moment you look at where farmers actually are. Rural coverage is patchy, mobile data is a real cost paid out of a thin margin, and a tool that needs the network is a tool that is absent on the morning the armyworm arrives.

So we built the version that works with the cable pulled out.

## What it does

**AGBE** (Yoruba: *àgbẹ̀*, farmer) is a farming advisor you can ask questions, running entirely on an ordinary 8 GB laptop with the internet switched off. No cloud, no API key, no account, no data cost.

Ask it what is eating your maize and it tells you to look for a pale inverted Y on the caterpillar's head and four dark dots in a square near the tail, and to open the whorl early morning when the larvae are active. It covers crops, pests, livestock, soils, drying and storage, and getting produce to market.

It is grounded deliberately in West and Central African smallholder systems: cassava mosaic, striga, fall armyworm, aflatoxin, Newcastle disease, harmattan planting windows, and the crops actually grown here.

**What it refuses matters as much as what it answers.** It will not give you an agrochemical dose, because rates differ by product and a confident wrong number is dangerous. It will not quote a market price. Asked about a child with a fever it declines outright and points to a clinic. Asked whether cassava mosaic can be cured, it says no, because it cannot.

## How we built it

**We read the scoring function before writing any code.**

```
S_total = 0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal
S_perf  = min(TPS ÷ 15.0, 1.0) × 100
S_eff   = max(0, (7.0 − peak RAM GB) ÷ 7.0) × 100
```

Throughput above 15 tokens per second earns **nothing**, and memory is charged linearly. So the instinct to run the largest model that fits inside 8 GB is exactly backwards. We measured five candidates on the target profile instead of reasoning about them:

| Model | tok/s | Peak RAM | Points /50 |
|---|---|---|---|
| Qwen2.5 0.5B | 46.6 | 0.50 GB | 48.57 |
| **Gemma 3 1B** | **26.9** | **0.88 GB** | **47.48** |
| Llama 3.2 1B | 22.8 | 1.26 GB | 46.39 |
| Qwen2.5 1.5B | 24.9 | 1.69 GB | 45.19 |
| Qwen2.5 3B | 11.5 | 3.26 GB | 32.94 |

The 3B concedes **14.5 points before answering a single question** and would need to be twenty-nine accuracy points better to break even.

**The corpus is the real work.** No dataset ships with this challenge. Scraping the web or having a large model write the answers would both have been faster, and both put claims into the training data that nobody can trace. This domain is graded by agronomists who notice invented chemistry. So every training pair is composed from a curated fact base of established extension practice: if a fact is not in that file, it cannot appear in the corpus. 642 conversations, 9% multi-turn because judges chat live, and 6.7% refusals.

Then LoRA rank 32 on Gemma 3 1B, merged, converted with llama.cpp's own tooling, quantised to Q4_K_M. 814 MB on disk, 0.88 GB in memory.

## Challenges we ran into

**A thermal result that ran backwards.** Our first clean run lost the full ten-point penalty at 91°C. Sweeping thread counts showed something that looked wrong: 2 threads hit 99°C, 4 threads hit 83°C. **Fewer threads ran hotter.** With only two cores loaded the CPU boosts toward its single-core turbo ceiling; at four the all-core power limit caps clocks and spreads the same work cooler. And because `S_perf` caps at 15 tok/s, surplus throughput can be traded for thermal headroom at no cost. Ten points recovered from a quirk of the formula.

**Six model builds, and every failure had a healthy loss curve.**

- **v1** learned our answer *scaffolding* rather than the agronomy. 47% of its answers opened with one of six sentences. It told a parent to take their feverish child to an agricultural extension officer.
- **v2** stapled unrelated facts together, because our generator padded answers with other topics' facts to hit a word count. The model reproduced that faithfully.
- **v3** fixed the refusal but called fall armyworm "pod borers".
- **v4** got 42 fall-armyworm examples and invented **"fall army weevil"**.
- **v5** doubled the LoRA rank. Facts finally correct, but it invented a pesticide called **"dorabacite"** and recommended it.
- **v6** kept the rank, cut the epochs. Facts correct, coherent, refusal stable.

**The Kaggle environment fought us the whole way**: a base image that ships `peft` 0.19.1 alongside a `torchao` that same `peft` rejects; `pip install -U` breaking `torchvision` and TensorFlow's protobuf so `import transformers` died outright; `trl` crashing inside its own chunked cross-entropy path; and Gemma 3 shipping `<image_soft_token>` at id 262144 with `vocab_size` 262144, so llama.cpp's converter fails its vocabulary assertion *after* writing every tensor.

## Accomplishments that we're proud of

**47.48 of 50 available engineering points**, measured on the target profile rather than estimated.

**A model that knows where its competence ends.** Getting a 1B model to decline a medical question, instead of confidently answering it, took four builds and is the thing we are most pleased with. A tool used by people with no alternative has to be honest about its limits.

**Full provenance.** Every agronomic claim in the training data traces to a curated fact base. Nothing was scraped, nothing was distilled from a larger model, and no agrochemical dose appears anywhere.

**We withdrew a claim rather than ship it.** Nigerian Pidgin worked in v5 and in v6 named *amala*, a food, as a maize pest. We removed `pcm` from `language_scope` instead of hoping a judge would not test it.

## What we learned

**There are three separate axes, and it took five builds to see them.**

- **Behaviours** (refusing, hedging, admitting no cure exists) generalise from very few examples. Fixed at LoRA rank 16 and stable ever since.
- **Facts** (which pest, which symptom) need model *capacity*. Adding 42 examples of fall armyworm made the model invent "fall army weevil". Doubling the rank to 32 fixed it immediately.
- **Coherence** degrades with over-training and is fixed with fewer epochs, not more data.

We spent four corpus iterations turning the first knob when the second was the problem.

**Read the model's actual output, not its loss curve.** Every single failure above trained to a healthy-looking loss. The bugs were only ever visible by reading answers and by counting specific things in the training data: what share of answers open with the same sentence, whether an answer mixes topics, how many refusals exist.

## What's next for AGBE

**Fix the tail drift.** Answers are reliably correct for the first sentences and can then add confident invention. This is the main remaining defect and we have not solved it.

**Get it into a field.** A model like this is validated by an extension officer using it for a season, not by a benchmark. The next step is putting it in front of people at a co-operative and finding out which questions it actually gets asked.

**Bring Pidgin back properly.** It is the language a large share of the target users speak, and it deserves a corpus built for it rather than a slice bolted on.

---

**Try it:** the weights are public and the runtime is `llama.cpp`, so nothing here has to be taken on trust.

```bash
curl -L -o agbe.gguf \
  https://huggingface.co/NEVODESIGN/agbe-1b/resolve/main/agbe-1b-q4_k_m.gguf

llama-cli -m agbe.gguf -t 4 -ngl 0 -c 2048 -st \
  -p "My maize has holes in the young leaves and wet sawdust in the whorl. What is this?"
```
