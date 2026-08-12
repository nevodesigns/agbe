"""LoRA fine-tune of Gemma 3 1B on the AGBE agriculture corpus, then GGUF export.

Written to run on a Kaggle free T4 (16GB, Turing). Notes that matter on that box:

  - Turing has no bf16, so fp16 throughout. bf16 silently falls back and wastes
    the session.
  - The corpus is small (hundreds of conversations), so this is a short run.
    Over-training a 1B on a narrow domain is the fastest way to wreck its general
    instruction following, which a judge WILL exercise with a hidden prompt.
  - Gemma is a gated repo on HuggingFace. Accept the licence on the model page and
    supply HF_TOKEN, or the download 401s.

The export path is deliberately the same llama.cpp the challenge scores against,
so what we measure locally is what the judges run.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

BASE_MODEL = "google/gemma-3-1b-it"

# Held small on purpose. r=16 is enough to move style and domain vocabulary on a
# 1B without overwriting what the base model already knows.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

MAX_SEQ_LEN = 1024          # our longest multi-turn conversations sit well inside this
EPOCHS = 3
LR = 2e-4
BATCH = 2
GRAD_ACCUM = 8              # effective batch 16


def load_corpus(path: pathlib.Path):
    from datasets import Dataset

    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # Drop our bookkeeping before it reaches the tokenizer.
        rows.append({"messages": rec["messages"]})
    print(f"loaded {len(rows)} conversations from {path.name}")
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="corpus/build/train.jsonl")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=float, default=EPOCHS)
    ap.add_argument("--merge", action="store_true",
                    help="merge the adapter into the base weights after training")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set. Gemma is gated and the download will 401.")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,     # Turing: fp16, not bf16
        device_map="auto",
        token=token,
        attn_implementation="eager",   # Gemma 3 recommends eager attention
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.1f}M of {total/1e6:.1f}M ({trainable/total*100:.2f}%)")

    ds = load_corpus(pathlib.Path(args.train))

    cfg = SFTConfig(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        optim="adamw_torch",
        max_length=MAX_SEQ_LEN,
        gradient_checkpointing=True,
        report_to=[],
        seed=20260812,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()

    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"adapter saved to {adapter_dir}")

    if args.merge:
        # llama.cpp converts a plain HF model, not an adapter, so merge first.
        from peft import PeftModel

        del model, trainer
        torch.cuda.empty_cache()

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="cpu",
            token=token, attn_implementation="eager",
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged_dir = out / "merged"
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tok.save_pretrained(str(merged_dir))
        print(f"merged model saved to {merged_dir}")
        print("\nnext, convert and quantise:")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {merged_dir} "
              f"--outfile agbe-f16.gguf --outtype f16")
        print(f"  ./llama.cpp/build/bin/llama-quantize agbe-f16.gguf "
              f"agbe-1b-q4_k_m.gguf Q4_K_M")


if __name__ == "__main__":
    main()
