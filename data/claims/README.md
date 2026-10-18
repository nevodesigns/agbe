# Claim extraction, pilot results (18 Oct 2026)

Pipeline: `tools/fetch_fulltext.py` (Tier A PDFs to cited passages) then `tools/extract_claims.py` (passages to verified claims).

A claim is accepted only if code proves it: its quote is in the passage, its numbers are in the quote, its content words are in the quote, it is not study language (plots, treatments, years, "we"), and it is not a chemical rate. Rates go to `review` and are never accepted automatically.

## Pilot
- 70 Tier A documents (CC-BY-4.0), 1,974 passages. 20 selected items had no PDF on CGSpace and were skipped.
- **Rule-based extractive baseline:** 4,194 candidate claims, mostly research-paper sentences, not farmer advice. Verified by construction, wrong in relevance. Not usable for training. Its dose filter first let "10 ml herbicide diluted to 1 litre" through, which is why the detector now flags any chemical word with a quantity.
- **LLM extraction, Qwen2.5-3B Q4 on CPU, 24 passages:** 29 candidate claims. The verifier rejected 18 (8 quotes not in the passage, 4 claims adding words, the rest study-specific or too short) and accepted 11. The model invented or paraphrased beyond the source in about 40% of its claims, and the verifier caught all of those. That is the reason claims are never trusted on the model's word.
- The 11 accepted claims read as usable farm facts, but some come from outside West Africa and none has been reviewed by an agronomist. They are a pipeline test, not training data.

## What this means
Verification works. Relevance needs a stronger model (a 7B on Kaggle) and an agronomist audit of a sample. `claims_qwen3b.jsonl` is the output; the rule-based file is git-ignored.
