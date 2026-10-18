# Start here: the one-notebook route

Upload **`train/AGBE_v2_Kaggle.ipynb`** (copy at `~/Downloads/agbe-kaggle/AGBE_v2_Kaggle.ipynb`) to Kaggle: Create, New Notebook, File, Import Notebook. It clones this repository, which already contains the passages (`data/fulltext_v1/`), so there is nothing else to upload. Set GPU T4 x2, Internet On, add the `HF_TOKEN` secret (write access), accept the Gemma and Llama terms on Hugging Face, then run the cells top to bottom. It does the bake-off, picks the base, extracts claims, builds the training set, trains, exports a GGUF and re-tests it, saving every stage to a private Hugging Face dataset repo so a disconnect loses nothing.

Everything below is the manual version of stage A only.

# Running the bake-off on free Kaggle compute

1. New Kaggle notebook, accelerator **GPU T4 x2**, internet **on**.
2. Add your Hugging Face token as a Kaggle secret named `HF_TOKEN`. Accept the terms of the gated models you want to test (Gemma, Llama) on their Hugging Face pages first.
3. Cells:

```python
!git clone -b agbe2 https://github.com/nevodesigns/agbe.git && cd agbe && pip -q install -U transformers accelerate bitsandbytes
import os; from kaggle_secrets import UserSecretsClient; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!cd agbe && gunzip -c data/fulltext_pilot/passages.jsonl.gz > /tmp/passages.jsonl
# small models first, full precision
!cd agbe && python3 train/bakeoff.py --models google/gemma-3-1b-it Qwen/Qwen3-1.7B meta-llama/Llama-3.2-1B-Instruct --passages /tmp/passages.jsonl
# 3B and 4B need 4-bit on a T4
!cd agbe && python3 train/bakeoff.py --load-4bit --models google/gemma-3-4b-it Qwen/Qwen3-4B-Instruct-2507 HuggingFaceTB/SmolLM3-3B --passages /tmp/passages.jsonl --out train/bakeoff_results_4bit
```

4. Download `train/bakeoff_results*/summary.md` and the per-model JSON files, and put them in the repo.

Notes:
- A full run of the 158 prompts in two modes takes roughly 10 to 25 minutes per small model on a T4.
- Kaggle gives about 30 GPU hours a week. Colab free works too, with shorter sessions. Run models one command at a time so a disconnect does not lose everything.
- Speed on the target laptop is measured later, on CPU, for the top two models only.
- The same pattern runs the claim extraction: `python3 tools/extract_claims.py --backend transformers --model Qwen/Qwen2.5-7B-Instruct --passages /tmp/passages.jsonl --out claims.jsonl` (use `--model Qwen/Qwen2.5-3B-Instruct` if the 7B does not fit).
