"""Emit the Kaggle notebook that trains AGBE and exports GGUF.

Generated rather than hand-written so the cells stay in sync with train_lora.py
and so the JSON is always valid.
"""

from __future__ import annotations

import json
import pathlib

MD = "markdown"
CODE = "code"

CELLS = [
(MD, """# AGBE — LoRA fine-tune of Gemma 3 1B

Trains the agriculture advisor for the **Africa Deep Tech Challenge 2026** and exports
a `Q4_K_M` GGUF ready for `llama.cpp`.

## Before you run this

1. **Accept the Gemma licence.** Open <https://huggingface.co/google/gemma-3-1b-it> while
   signed in and accept the terms. Gemma is a gated repo; without this the download 401s.
2. **Add your HuggingFace token to Kaggle Secrets** as `HF_TOKEN`
   (Add-ons → Secrets). Get one at <https://huggingface.co/settings/tokens>, read scope
   is enough.
3. **Turn on the GPU**: Settings → Accelerator → **GPU T4 x2** (one is used).
4. Session needs internet on: Settings → Internet → On.

Expect roughly 15 to 30 minutes end to end on a T4.

**Why the run is short.** The corpus is a few hundred conversations by design, and
over-training a 1B on a narrow domain destroys the general instruction following that a
judge will exercise with a hidden prompt. Three epochs on a small LoRA is the point,
not a limitation."""),

(CODE, """# T4 is Turing: fp16 only, no bf16. Verify what we actually got.
import torch, subprocess
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "n/a")"""),

(CODE, """# Use Kaggle's preinstalled transformers/peft. Upgrading them was a mistake:
# `pip install -U` pulled versions that no longer matched the preinstalled
# torchvision ("operator torchvision::nms does not exist") and disturbed
# TensorFlow's protobuf. Both break `import transformers` outright.
#
# One preinstall still has to go. Kaggle ships peft 0.19.1 alongside torchao
# 0.10.0, and that peft RAISES on any torchao below 0.16 from inside its LoRA
# dispatcher, so the base image contradicts itself here. We do no quantised
# training, so remove torchao rather than chase a torchao/torch version match.
!pip uninstall -q -y torchao

# transformers probes for TensorFlow and JAX at import time; Kaggle's TF is
# fragile and merely looking for it can take the import down. PyTorch only.
import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

import subprocess
r = subprocess.run(
    ["python", "-c",
     "import transformers, peft; print(transformers.__version__, peft.__version__)"],
    capture_output=True, text=True, env=dict(os.environ))
print("READY -", r.stdout.strip()) if r.returncode == 0 else print("BROKEN:", r.stderr[-800:])"""),

(CODE, """import os
from kaggle_secrets import UserSecretsClient
try:
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as e:
    print("Could not load HF_TOKEN from Secrets:", e)
    print("Add it under Add-ons -> Secrets, or Gemma will refuse to download.")"""),

(CODE, """# Corpus and trainer live in the submission repo, so the notebook stays thin
# and the training data is the same version that ships with the submission.
# Step out of the clone target first: on a RE-run the shell is already inside it,
# and rm -rf on your own working directory breaks getcwd for every later command.
%cd /kaggle/working
!rm -rf /kaggle/working/agbe
!git clone -q https://github.com/nevodesigns/agbe.git /kaggle/working/agbe
%cd /kaggle/working/agbe
!wc -l corpus/build/train.jsonl corpus/build/holdout.jsonl"""),

(CODE, """import json
rows = [json.loads(l) for l in open("corpus/build/train.jsonl")]
multi = sum(1 for r in rows if r["_meta"].get("turns", 1) > 1)
print(f"conversations: {len(rows)}   multi-turn: {multi} ({multi/len(rows)*100:.0f}%)")
print("\\nsample:")
for m in rows[0]["messages"][1:]:
    print(f"[{m['role']}] {m['content'][:220]}\\n")"""),

(CODE, """!python train/train_lora.py --train corpus/build/train.jsonl --out /kaggle/working/out --merge"""),

(MD, """## Export to GGUF

The challenge scores through `llama.cpp`, so we convert with `llama.cpp`'s own tooling
rather than a third-party exporter. What we measure locally is then exactly what the
judges run."""),

(CODE, """%%capture
!git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp
!pip install -q -r /kaggle/working/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
!cmake -S /kaggle/working/llama.cpp -B /kaggle/working/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
!cmake --build /kaggle/working/llama.cpp/build --target llama-quantize llama-cli -j4"""),

(CODE, """!python /kaggle/working/llama.cpp/convert_hf_to_gguf.py \\
    /kaggle/working/out/merged --outfile /kaggle/working/agbe-f16.gguf --outtype f16
!ls -lh /kaggle/working/agbe-f16.gguf"""),

(CODE, """# Q4_K_M is the quantisation the score curve was measured on.
!/kaggle/working/llama.cpp/build/bin/llama-quantize \\
    /kaggle/working/agbe-f16.gguf /kaggle/working/agbe-1b-q4_k_m.gguf Q4_K_M
!ls -lh /kaggle/working/agbe-1b-q4_k_m.gguf"""),

(MD, """## Smoke test

Four prompts. The first two are in-domain, the third is Pidgin, and the fourth is
deliberately **out of scope** — a small model that answers a medical question confidently
is a model that will lose accuracy marks in front of a judge."""),

(CODE, """# Flags matter here:
#   -st/--single-turn   generate one turn then EXIT. Without it llama-cli waits on
#                       a terminal that never comes (this burned a 900s timeout).
#   --simple-io         documented as "better compatibility in subprocesses".
#   NO -no-cnv          conversation mode must stay ON so the Gemma chat template
#                       is applied. Raw -p prompts bypass the format the model was
#                       trained in, and you get base Gemma talking about frost.
#   NO -sys             judges chat through their own interface without our system
#                       prompt, so this is the honest test.
PROMPTS = [
  ("diagnosis", "My maize has holes in the young leaves and there is something like wet sawdust in the centre of the plant. What is this?"),
  ("timing", "When should I plant maize?"),
  ("pidgin", "My maize get hole for leaf and I dey see like sawdust for inside the middle. Wetin be dis?"),
  ("MUST REFUSE", "My child has a fever and is vomiting. What medicine should I give?"),
]
import subprocess
for label, p in PROMPTS:
    print("=" * 78); print(f"[{label}]  {p}"); print("-" * 78)
    try:
        out = subprocess.run([
            "/kaggle/working/llama.cpp/build/bin/llama-cli",
            "-m", "/kaggle/working/agbe-1b-q4_k_m.gguf",
            "-t", "4", "-ngl", "0", "-c", "2048", "-n", "200",
            "--temp", "0.3", "-st", "--simple-io", "--no-warmup", "-p", p,
        ], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
        body = out.stdout
        i = body.find(p)
        print((body[i + len(p):] if i >= 0 else body).strip()[:1400])
    except subprocess.TimeoutExpired:
        print("TIMED OUT")
    print()"""),

(CODE, """# Publish the weights. download_model.sh in the submission repo fetches
# from exactly this public URL, so this is the step that wires the submission.
from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi, create_repo

wtok = UserSecretsClient().get_secret("HF_WRITE_TOKEN")
repo = "NEVODESIGN/agbe-1b"
create_repo(repo, token=wtok, exist_ok=True, repo_type="model", private=False)
HfApi().upload_file(
    path_or_fileobj="/kaggle/working/agbe-1b-q4_k_m.gguf",
    path_in_repo="agbe-1b-q4_k_m.gguf",
    repo_id=repo, token=wtok)
print("published ->", f"https://huggingface.co/{repo}/resolve/main/agbe-1b-q4_k_m.gguf")"""),

(MD, """## Download

Right-click `agbe-1b-q4_k_m.gguf` in the Kaggle output panel and download it, or publish
it to a HuggingFace repo so `download_model.sh` can fetch it. The submission template
requires `download_model.sh` to pull the weights, and forbids committing the `.gguf`
itself.

Then locally:

```bash
python work/bench.py agbe-1b-q4_k_m.gguf        # our fast harness
adtc-profiler run --submission . --mode participant --output submission.json
```

Run the profiler **from a cold machine**: the thermal penalty is 10 points and we
measured 98 to 99°C under sustained back-to-back load."""),
]


def main() -> None:
    nb = {
        "cells": [
            {"cell_type": kind,
             "metadata": {},
             "source": src.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if kind == CODE else {})}
            for kind, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = pathlib.Path(__file__).resolve().parent / "AGBE_train_kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out}  ({len(CELLS)} cells, {out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
