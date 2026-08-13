"""LoRA fine-tune of Gemma 3 1B on the AGBE agriculture corpus, then GGUF export.

Written to run on a Kaggle free T4 (16GB, Turing).

**No trl.** We used trl's SFTTrainer first and it broke inside its own chunked
cross-entropy path (`_chunked_ce_forward` reads `outputs.last_hidden_state`, which a
PEFT-wrapped causal LM does not return). Nothing we could pass would fix that, and
trl's API had already churned twice in this stack. Plain `transformers.Trainer` plus
about sixty lines of explicit data handling removes the dependency and, more usefully,
makes the label masking visible instead of hidden behind a helper.

Design notes that matter here:

  - **Loss is computed on assistant turns only.** User questions are masked to -100.
    Training on the questions too teaches the model to generate farmer questions, which
    is not the job, and wastes a small model's limited capacity.
  - Turing has no native bf16, so fp16 throughout.
  - The corpus is a few hundred conversations by design. Over-training a 1B on a narrow
    domain destroys the general instruction following a judge WILL probe with a hidden
    prompt. The short run is the point.
  - Gemma's chat template has no system role, so the system prompt is folded into the
    first user turn rather than dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

# Set BEFORE transformers is imported anywhere below.
#
# transformers probes for TensorFlow and JAX at import time. On Kaggle, TF is
# preinstalled and its protobuf dependency gets disturbed by the pip upgrade the
# notebook performs, so merely LOOKING for TF raises
# "cannot import name 'runtime_version' from 'google.protobuf'" and takes
# `import transformers` down with it. We train in PyTorch only.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

BASE_MODEL = "google/gemma-3-1b-it"

# r=16 is enough to move style and domain vocabulary on a 1B without overwriting
# what the base model already knows.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

MAX_SEQ_LEN = 1024
EPOCHS = 3
LR = 2e-4
BATCH = 2
GRAD_ACCUM = 8              # effective batch 16
IGNORE = -100


def fold_system(messages: list[dict]) -> list[dict]:
    """Gemma's chat template rejects a system role, so merge it into the first user turn."""
    system, out = None, []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
            continue
        if system and m["role"] == "user" and not out:
            out.append({"role": "user", "content": f"{system}\n\n{m['content']}"})
            system = None
        else:
            out.append(dict(m))
    return out


def _ids(rendered) -> list[int]:
    """Flatten whatever apply_chat_template returned into a list of token ids.

    transformers 5.x returns a dict-like BatchEncoding here where 4.x returned a
    plain list. Taking len() of the dict counts KEYS, which silently made every
    conversation look empty and dropped the whole corpus ("encoded 0 of 375").
    Some versions also return a batch of one. Normalise all three shapes.
    """
    if hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    elif isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], (list, tuple)):
        rendered = rendered[0]
    return list(rendered)


def encode(messages: list[dict], tok, max_len: int) -> dict | None:
    """Tokenise a conversation, masking everything that is not an assistant turn.

    Built incrementally: render the first i+1 messages, and whatever tokens that adds
    over the previous render belong to message i. Chat templates are prefix-stable, so
    the diff is exactly that message's span, which lets us mask precisely without
    string-matching for delimiters.
    """
    msgs = fold_system(messages)
    input_ids: list[int] = []
    labels: list[int] = []

    for i, msg in enumerate(msgs):
        rendered = _ids(tok.apply_chat_template(
            msgs[: i + 1], tokenize=True, add_generation_prompt=False))
        if len(rendered) <= len(input_ids):
            continue                      # template produced nothing new; skip
        new = rendered[len(input_ids):]
        labels.extend(new if msg["role"] == "assistant" else [IGNORE] * len(new))
        input_ids = list(rendered)

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    if not any(l != IGNORE for l in labels):
        return None                       # nothing to learn from, drop it
    return {"input_ids": input_ids, "labels": labels}


class Collator:
    """Pad a batch to its longest sequence. Labels pad with -100 so padding is ignored."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]):
        import torch

        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            gap = width - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * gap)
            labels.append(f["labels"] + [IGNORE] * gap)
            attn.append([1] * len(f["input_ids"]) + [0] * gap)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="corpus/build/train.jsonl")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=float, default=EPOCHS)
    ap.add_argument("--merge", action="store_true",
                    help="merge the adapter into the base weights after training")
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set. Gemma is gated and the download will 401.")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,           # Turing: fp16, not bf16
        token=token,
        attn_implementation="eager",   # Gemma 3 recommends eager attention
    )
    model.config.use_cache = False
    if torch.cuda.is_available():
        model = model.cuda()

    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.1f}M of {total/1e6:.1f}M ({trainable/total*100:.2f}%)")

    # ---- data -------------------------------------------------------------
    raw = [json.loads(l) for l in pathlib.Path(args.train).read_text().splitlines() if l.strip()]
    dataset = []
    for rec in raw:
        enc = encode(rec["messages"], tok, MAX_SEQ_LEN)
        if enc:
            dataset.append(enc)
    print(f"encoded {len(dataset)} of {len(raw)} conversations")
    if not dataset:
        raise SystemExit(
            "encoded 0 conversations: the chat template produced no supervised "
            "tokens. Check apply_chat_template's return shape for this "
            "transformers version before training.")

    lens = sorted(len(d["input_ids"]) for d in dataset)
    sup = sorted(sum(1 for l in d["labels"] if l != IGNORE) for d in dataset)
    print(f"tokens per example: p50={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} max={lens[-1]}")
    print(f"supervised tokens:  p50={sup[len(sup)//2]} (loss is on assistant turns only)")

    # Prove the masking is right before spending a GPU hour on it.
    sample = dataset[0]
    kept = [t for t, l in zip(sample["input_ids"], sample["labels"]) if l != IGNORE]
    print("\n--- masking check, text the loss is computed on ---")
    print(tok.decode(kept)[:400].replace("\n", " "))
    print("--- end check ---\n")

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        logging_steps=5,
        save_strategy="no",
        fp16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to=[],
        seed=20260812,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=dataset,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"adapter saved to {adapter_dir}")

    if args.merge:
        # llama.cpp converts a plain HF model, not an adapter, so merge first.
        from peft import PeftModel

        del trainer, model
        torch.cuda.empty_cache()

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.float16, token=token,
            attn_implementation="eager",
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged_dir = out / "merged"
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tok.save_pretrained(str(merged_dir))
        print(f"merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
