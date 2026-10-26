# Decisions

Each entry: what was decided, when, and why. Reversals are kept, not erased.

| Date | Decision | Reason |
|---|---|---|
| 15 Jun 2026 | The repository began from the competition's submission template, which brought a GPL-3.0 licence file | The author did not choose it; it is recorded in `LICENCES.md` and the git history |
| 12 Aug 2026 | Fine-tune a small open model (Gemma 3 1B) for the competition laptop limit (8 GB RAM, CPU only) | The competition scored on a standard laptop; a 1B model in 4-bit fits |
| 17 Oct 2026 | Branch `agbe2` is the default; `main` stays at the competition commit, tagged `adtc-2026-final` | The competition record must not change |
| 17 Oct 2026 | Train only on Tier A data: CC-BY-4.0, CC-BY-3.0, CC0, public domain | Anything else needs permission, or forbids commercial use |
| 17 Oct 2026 | Free compute only (Kaggle T4, Colab) | No budget; keeps the build reproducible by anyone |
| 17 Oct 2026 | Hold FAO, CABI, NAERLS and items with no licence until written permission | FAO publications are CC BY-NC-SA 3.0 IGO (no commercial use); CABI is per item; permission requests were sent on 18 Oct |
| 17 Oct 2026 | Answers must come from numbered sources, be checked per sentence, and say "I do not have a source" when none covers the question | v16 answered from memory and ignored retrieved sources (it scores 0% on the source-grounded test) |
| 19 Oct 2026 | Move the v2 base from Gemma 3 1B to Qwen3 (1.7B edge, 4B server) | Bake-off: Qwen3 4B led; Qwen3 1.7B reached 87% to 94% pass; Qwen3 models are Apache-2.0 (verified on their model cards, 26 Oct 2026), while Gemma's terms treat hosting as distribution and carry use restrictions |
| 20 Oct 2026 | Hold out the test set by passage, report abstention on clean cases only | 29 of 70 abstain cases were answerable from other documents; counting them wrongly penalised the model |
| 20 Oct 2026 | Abandon training abstain examples built from same-document hard negatives | Abstention got worse; the model likely learned that same-document sources mean abstain |
| 22 Oct 2026 | Clean abstain data (retrieved passages that support no accepted claim), share 0.25 | Raised grounded pass 87.1% to 94.3% with 1.4% wrong abstentions (run2) |
| 22 Oct 2026 | Default prompt: best 3 passages, trimmed to about 100 words | 29% to 39% faster answers; pass rate stayed close (91.4% to 90.0% on the shipping file) |
| 23 Oct 2026 | Do not train on the short prompt (run3) | Pass 91.4% against 94.3%; clean abstention fell |
| 23 Oct 2026 | Do not raise the abstain share to 0.40 (run4) | Clean abstention 46.3% against 48.8%; no gain |
| 23 Oct 2026 | Do not add a retrieval-score gate | Each extra correct abstention cost about one correct grounded answer |
| 23 Oct 2026 | Treat the guard's support figure as circular; ask a person to review answers | Guard and scorer share one claim checker |
| 26 Oct 2026 | Add `LICENCES.md`; rewrite the template shell script; keep private correspondence and the superseded pilot out of the repository | A reader must be able to tell which licence covers which material |

## Open decisions

- Which base model ships: Qwen3 1.7B (2.0 GB, about 0.73x v16 speed on the same CPU) or Qwen3 4B (4.5 GB, about 0.35x, 85.4% correct abstention).
- Whether to release v2 weights publicly and under which notices.
