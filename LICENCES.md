# Licences in this repository

The repository holds several kinds of material with different licences. This file says which is which.

| Material | Where | Licence | Note |
|---|---|---|---|
| Code, scripts and documentation written for AGBE | most of the repository | GPL-3.0 (see `LICENSE`) | The licence file arrived with the competition template at the first commit |
| v16 training corpus (original work, no third-party text) | `corpus/`, `provenance/` | Same as the code | Composed by `corpus/generate.py` from a hand-written fact base |
| v2 source documents and the passages cut from them | `data/fulltext_v1/`, `data/sources/` | CC-BY-4.0, each document by its own authors | The text belongs to its authors. Attribution is in `data/fulltext_v1/ATTRIBUTION.md` and `data/LICENCE_MANIFEST.md`. Keep it with any copy |
| v2 claim file, questions and training examples | `data/run1_kaggle/` | Offered under the same licence as the code | Claims were extracted from the CC-BY documents by Qwen2.5-7B-Instruct (Apache-2.0) and are quotations or close restatements of them, so the source attribution applies. The questions and examples are synthetic |
| v16 model weights | Hugging Face `NEVODESIGN/agbe-1b` | Gemma Terms of Use and Prohibited Use Policy: https://ai.google.dev/gemma/terms | A fine-tune of Google's Gemma 3 1B. Gemma's terms count hosting as distribution |
| v2 model weights | not yet released | Apache-2.0 base models (Qwen3), plus the notices that licence requires | Check each base model's licence file again at release |
| llama.cpp, PyTorch, Transformers, PEFT and other libraries | not included | Their own licences | Not part of this repository |

Anyone who already holds a copy of the public code keeps the rights the GPL-3.0 gave them. Nothing here withdraws them.

## For anyone deploying a model from this project

- **Gemma-based files (v16).** Gemma's terms define "Distribution" to include "making Gemma or its functionality available as a hosted service via API, web access, or any other electronic or remote means" (section 1.1(b)). A host must therefore carry Google's use restrictions into its own user terms and give recipients a copy of the terms (section 3.1). Read https://ai.google.dev/gemma/terms before deploying.
- **Qwen3-based files (v2).** The base model cards state `license: apache-2.0`. Redistribution needs a copy of that licence, notices on modified files, and any NOTICE text. Check the base model's licence file again at release.
- **Source passages.** Any product that shows or reuses them must keep the attribution in `data/fulltext_v1/ATTRIBUTION.md`. CC-BY-4.0 has no non-commercial limit and no share-alike condition.
- **Material outside this repository.** FAO web content is, by default, licensed for non-commercial use only. Do not add it to a commercial training set without permission.
