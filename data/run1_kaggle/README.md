# Kaggle run 1 (19 Oct 2026), copied out of the private Hugging Face repo as a backup

Fetched with `kaggle kernels output`. All passages come from Tier A (CC-BY) sources, see data/fulltext_v1/ATTRIBUTION.md.

- claims.jsonl: 800 claims from 590 passages by Qwen2.5-7B-Instruct (570 accepted, 7 review, 223 rejected by the verifier)
- questions.jsonl: questions written for the 191 passages that have an accepted claim
- sft.jsonl: the 375 training conversations from the first composer (before the off-topic filter)
- notebook1_bakeoff_log.txt, notebook2_log.txt: the Kaggle logs

Quality check (19 Oct 2026, train/diagnose_data.py): accepted claims are mostly sound advice sentences quoted verbatim.
Two defects found and fixed in tools/build_grounded_sft.py: about a quarter of grounded answers did not address the question
(lead claim shared less than 15% of the question words), and 27 accepted claims were survey findings (Most farmers..., Farmers do not...).
Re-composing the same data with the filter gives 332 grounded, 60 abstain, 103 refusal examples and skips 68 off-topic ones.
