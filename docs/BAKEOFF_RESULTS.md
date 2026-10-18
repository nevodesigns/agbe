# Base-model bake-off, 19 October 2026

Kaggle T4, retrieval mode (the numbered sources plus the AGBE rules in the system prompt), greedy decoding, 60 prompts sampled across the behaviour and hostile batteries (26 attacks, 34 legitimate), competition scorer. Small models in full precision, 3B and 4B in 4-bit. The two Llama models were skipped: the Hugging Face terms had not been accepted.

| model | pass | attacks withstood | legit answered | leaks | over-refused | drift | unsupported claims |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 33/60 | 10/26 | 23/34 | **0** | 2 | 0 | 0.969 |
| google/gemma-3-1b-it | 25/60 | 8/26 | 17/34 | 7 | 1 | 1 | 0.932 |
| HuggingFaceTB/SmolLM3-3B | 25/60 | 3/26 | 22/34 | 9 | 1 | 1 | 0.961 |
| Qwen/Qwen3-1.7B | 23/60 | 2/26 | 21/34 | 4 | 1 | 0 | 0.951 |
| google/gemma-3-4b-it | 23/60 | 5/26 | 18/34 | 5 | 1 | 0 | 0.843 |

For comparison, the shipped v16 (fine-tuned, no retrieval) withstood 59 of 62 attacks with 0 leaks.

## How to read it, honestly
- **Qwen3-4B-Instruct is the only clear result**: zero leaks and the highest pass count. It is the server-profile candidate.
- **The small models are too close to call.** 23 to 25 of 60 is within the noise of a 60-prompt sample. The pre-registered rule (fewest leaks first) picks Qwen3-1.7B for the edge profile, but Gemma 3 1B is smaller and faster. So both are trained and compared after training.
- **Bare models fail attacks, as expected.** A system prompt alone does not make a base model refuse. In the product the guard stops 53 of 62 attacks before the model runs, and fine-tuning trains refusals into the model. These attack numbers say nothing about the finished system.
- **The "unsupported claims" column (0.84 to 0.97) is not usable.** The old batteries ask general farm questions, most of which the retrieved passages do not contain, so almost every correct answer from the model's own knowledge counts as unsupported. It measures retrieval coverage, not model quality.
- **The test is the wrong shape for RAG.** It rewards answering from memory and punishes "my sources do not cover this". That is why `train/rag_eval.py` exists: held-out passages, grounded and abstain cases, scored on whether the answer is supported by the sources, cited, and honest about gaps.

## Decision
Train three bases on the same grounded data (gemma-3-1b-it, Qwen3-1.7B, Qwen3-4B-Instruct), score each and its untrained base on the held-out source-following test, and choose from those numbers.
