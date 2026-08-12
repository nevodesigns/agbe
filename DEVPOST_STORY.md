## Inspiration

Nigeria has roughly one agricultural extension officer for every few thousand farming
households. The knowledge that would raise a smallholder's yield is not secret and it is
not new. It sits in extension manuals, and it does not travel the last mile, because the
last mile has no officer and often no signal.

The obvious answer is a farming chatbot. That answer breaks the moment you look at where
farmers actually are. Rural coverage is patchy, mobile data is a real cost paid out of a
thin margin, and a tool that needs the network is absent on the morning the armyworm
arrives.

So offline is not a feature we bolted on. It is the shape of the whole thing. If it
cannot answer with the network off, on hardware a farmer or a co-operative already owns,
it does not count.

## What it does

AGBE is a compact language model, fine-tuned on West and Central African agronomy and
quantised to GGUF, that runs through llama.cpp on an 8 GB laptop with integrated
graphics and no internet connection.

It answers questions about crops, pests, livestock, soils, storage and getting produce to
market. It works in English and in Nigerian Pidgin. It holds a conversation, so a farmer
can ask a follow up like "what if I cannot afford that" and get an answer that builds on
what was already said.

It also knows what it is not. It will not invent a spray dose, it will not price a
harvest, and it will not answer a medical question about a person.

## How we built it

We started by reading the scoring formula rather than the marketing:

```
S_total = 0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal
S_perf  = min(TPS / 15.0, 1.0) × 100
S_eff   = max(0, (7.0 − peak RAM GB) / 7.0) × 100
```

Two things fall straight out of that. Throughput above 15 tokens per second earns exactly
nothing, and memory is paid for linearly. So the instinct to pick the largest model that
fits inside 8 GB is precisely backwards.

We built a benchmark harness that emulates the ADTC Standard Laptop (four threads, no GPU
offload, memory capped to 7 GB) and measured five candidates rather than guessing:

| Model (Q4_K_M) | tok/s | Peak RAM | S_perf | S_eff | Points of 50 |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5 0.5B | 57.2 | 0.55 GB | 100.0 | 92.2 | 38.44 |
| **Gemma 3 1B** | **26.9** | **0.88 GB** | **100.0** | **87.4** | **37.48** |
| Llama 3.2 1B | 32.6 | 1.26 GB | 100.0 | 82.0 | 36.39 |
| Qwen2.5 1.5B | 24.9 | 1.69 GB | 100.0 | 75.9 | 35.19 |
| Qwen2.5 3B | 11.1 | 3.26 GB | 74.2 | 53.4 | 22.95 |

The 3B is the instructive row. It is the only candidate that misses the throughput bar
outright, and its footprint collapses the efficiency term. It concedes 14.5 points to a
1B before a single question is asked, which it would need to win back by being twenty
nine accuracy points better. Gemma 3 1B was chosen over a 0.5B for under one point of
engineering, and over Llama 3.2 1B at identical file size purely because it holds
0.88 GB in memory instead of 1.26 GB.

For the training corpus there is no official dataset, so the corpus is the real
differentiator. Scraping the web or having a large model write the answers would both
have been faster, and both put claims into the data that nobody can trace. This domain is
graded by agronomists who will notice invented chemistry. So every training pair is
composed from a curated fact base of established extension practice: if a fact is not in
that file, it cannot appear in the corpus. Hand written gold exemplars carry the house
style, including the awkward cases where the correct answer is "there is no cure" or
"do not buy that insecticide".

## Challenges we ran into

**The thermal penalty, and a counterintuitive fix.** Our first clean run lost the full 10
point penalty at 91°C. Sweeping thread counts produced a result that looked wrong: two and
three threads ran at 99°C while four and six threads ran at 83 to 84°C. Fewer threads ran
hotter. The explanation is that with only two cores loaded the CPU boosts toward its
single core turbo ceiling and spikes per core temperature, while at four or more the all
core power limit caps clocks and spreads the same work cooler. More parallelism, less
heat, more throughput.

**Silently truncated downloads.** Three model files downloaded incompletely while curl
exited 0, and a truncated GGUF still carries a valid magic number at byte 0, so neither
the exit code nor a header check caught it. llama-bench simply reported "failed to load
model". We replaced it with a fetcher that verifies length against the remote metadata and
resumes until the file genuinely matches.

**A metric that hid a bug.** Our multi turn conversation builder pulled hardcoded fact
fields that do not exist on every topic, so for some topics the answer collapsed to a bare
lead sentence with no content behind it. The summary statistics looked fine. The bug only
became visible after we changed the word count metric to measure per assistant turn
instead of per record, which pushed the minimum to five words. One hundred and fifty two
content free training examples had been generated, and bad examples damage a 1B model
faster than good ones help it.

## Accomplishments that we're proud of

Every number we report is measured on the target profile rather than estimated, and where
a figure is provisional we say so.

We banked 37.48 of the 50 points that are pure engineering, and we can show exactly why
each competing choice scores lower.

The corpus has full provenance. We can point at the source of any claim in the training
data, and the model is trained to refuse the places where a confident answer would be
dangerous.

## What we learned

Read the scoring function before writing code. The single most valuable insight in this
project was that throughput above 15 tokens per second is worth nothing, which converts
surplus compute into thermal headroom for free and inverts the obvious model choice.

The metric that hides a problem is usually the one aggregating across the thing that is
broken. Changing what we measured, not how hard we looked, is what surfaced our worst bug.

On a constrained target, the discipline that matters is subtraction. Smaller model, less
memory, fewer invented claims.

## What's next for AGBE

Broaden the fact base beyond the current crop, pest and livestock coverage, and extend the
language scope past English and Nigerian Pidgin toward Hausa, Yoruba and Igbo, where the
Masakhane community datasets are the obvious foundation.

Pair the model with retrieval over a local corpus so a co-operative can drop in its own
regional extension material without retraining anything.

Get it onto the machines it was designed for and find out where it is wrong, because a
model like this is only ever validated in a field, not in a benchmark.
