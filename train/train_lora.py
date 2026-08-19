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

# Qwen2.5-1.5B was tried and abandoned. Recorded so nobody repeats it:
#
#   1. Training silently did nothing. grad_norm went nan on step 1 and stayed
#      there, loss logged 0.0 for every step, and the resulting adapter was
#      identical to the base model. Cause: we load weights directly in float16
#      and train fp16 with no master copy. Gemma tolerates that; Qwen2.5 is a
#      bf16-trained model and overflows on Turing. Fixable by loading fp32 and
#      letting AMP handle it.
#   2. GGUF conversion is blocked anyway. llama.cpp's converter looks for a
#      sentencepiece tokenizer.model that Qwen does not ship, falls back to the
#      gpt2 path, and hits an AttributeError inside transformers 5.0's Qwen
#      tokenizer ("'list' object has no attribute 'keys'"). Not our bug to fix.
#
# Two debug cycles to chase 2.29 engineering points we would then hand back, on
# a base model that is also 35% larger to download. Gemma 3 1B works, banks
# 47.48 of 50, and names fall armyworm correctly.
BASE_MODEL = "google/gemma-3-1b-it"

# Raised from r=16 after four corpus iterations failed to teach FACTS.
#
# The evidence: style transferred completely (the model reproduces our answer
# shape, our closers, our phrasing) while facts did not (it answered "pod borers",
# then invented "fall army weevil"). That is underfitting the content while
# overfitting the form, and no amount of corpus editing fixes it, because the
# corpus was never the limiting factor. r=16 with 3 epochs over ~640 short
# examples is roughly 120 optimiser steps, which is very little signal for
# specific facts.
#
# Refusal behaviour DID transfer at r=16, which is consistent: behaviours
# generalise from few examples, facts do not.
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

MAX_SEQ_LEN = 1024
# Back to 3 after r=32/5ep overshot. Rank 32 is what made the FACTS stick
# (v5 finally said "fall armyworm" instead of inventing "fall army weevil"), so
# that stays. The extra epochs are what cost coherence: v5 produced word salad
# ("Maize in frequent banality is planted under apathy") and, far worse, invented
# a pesticide called "dorabacite" and recommended it. Keep the capacity, drop the
# over-training.
# Back to 3 for v11, reverting the v10 experiment exactly as this comment said to.
#
# The v9/v10 pair separated two things that had been confounded all along:
#
#   v9  (3 epochs, 746 ex)  lost facts. Read as underfitting at the time.
#   v10 (4 epochs, 812 ex)  recovered some facts AND started inventing vocabulary:
#                           it called coccidiosis "mortjacket" and opened an answer
#                           with "Scarets on a plant". That is the v5 "dorabacite"
#                           signature. It also contradicted itself inside one
#                           answer ("that is stem borer, not armyworm ... that is
#                           wilt and is armyworm"). On the hostile battery it went
#                           BACKWARDS against v9, 77/92 to 81/92, and the legitimate
#                           edge cases collapsed from 9/10 to 5/10.
#
# So epochs were never the constraint. The real cause of v9's lost facts was in the
# corpus and this build fixes it there:
#
#   1. EVERY generated diagnosis question named the disease ("I think I have
#      coccidiosis, how do I confirm it?"). The model was only ever trained to
#      elaborate on a diagnosis it had been handed, never to map symptoms onto a
#      name, which is the question a judge actually types. 58 symptom-first
#      examples now exist and are exempt from the cap.
#   2. The sentence cap had starved the rarest and most valuable slice: pests and
#      diseases fell from 95 examples to 47, leaving striga on ONE. Diagnosis
#      slices now get a higher cap and no disease sits below six examples.
#
# Fix the data, keep the setting that does not confabulate.
EPOCHS = 3
LR = 1.5e-4  # slightly lower, since rank and epochs both went up
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


def strip_oversized_tokens(merged_dir: pathlib.Path) -> None:
    """Drop tokenizer entries whose id is >= config.vocab_size.

    Gemma 3 ships <image_soft_token> at id 262144 while the text-only 1B declares
    vocab_size 262144, so the highest token id equals the vocabulary size. Google's
    own tokenizer carries it, so this is not an artefact of re-saving. llama.cpp's
    converter asserts `max(tokenizer.vocab.values()) < vocab_size` and dies at the
    very END of conversion, after writing every tensor, leaving no .gguf behind.

    The token is a placeholder for image input. A text-only model has no embedding
    row for it and can neither consume nor emit it, so removing it costs nothing.
    """
    cfg = json.loads((merged_dir / "config.json").read_text())
    vs = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size")
    if not vs:
        return

    tj_path = merged_dir / "tokenizer.json"
    if tj_path.exists():
        tj = json.loads(tj_path.read_text())
        dropped = []

        added = tj.get("added_tokens", [])
        keep = [a for a in added if a["id"] < vs]
        if len(keep) != len(added):
            dropped += [a["content"] for a in added if a["id"] >= vs]
            tj["added_tokens"] = keep

        # The BASE vocab too, not only added_tokens. This was the gap: stripping
        # added_tokens alone satisfied the old converter, but llama.cpp was
        # refactored (conversion/base.py) and now asserts on tokenizer.vocab,
        # which merges the base vocab in. <image_soft_token> sits in BOTH, so the
        # assertion still fired after a "successful" strip, at the very end of
        # conversion, leaving no .gguf behind.
        vocab = tj.get("model", {}).get("vocab")
        if isinstance(vocab, dict):
            over = [t for t, i in vocab.items() if isinstance(i, int) and i >= vs]
            for t in over:
                del vocab[t]
            dropped += over
        elif isinstance(vocab, list):
            # unigram style: [[token, score], ...] where index IS the id
            if len(vocab) > vs:
                dropped += [v[0] if isinstance(v, (list, tuple)) else v
                            for v in vocab[vs:]]
                tj["model"]["vocab"] = vocab[:vs]

        if dropped:
            tj_path.write_text(json.dumps(tj, ensure_ascii=False))
            print(f"  stripped out-of-range tokens: {sorted(set(dropped))}")

    at_path = merged_dir / "added_tokens.json"
    if at_path.exists():
        at = json.loads(at_path.read_text())
        at_path.write_text(json.dumps(
            {k: v for k, v in at.items() if v < vs}, ensure_ascii=False))

    tc_path = merged_dir / "tokenizer_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        dec = tc.get("added_tokens_decoder", {})
        keep = {k: v for k, v in dec.items() if int(k) < vs}
        if len(keep) != len(dec):
            tc["added_tokens_decoder"] = keep
            tc_path.write_text(json.dumps(tc, ensure_ascii=False))


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

    # Required for gradient checkpointing + LoRA. The base weights are frozen, so
    # the embedding output carries requires_grad=False, and a checkpointed block
    # then has nothing to backpropagate through:
    #   "element 0 of tensors does not require grad and does not have a grad_fn"
    # This makes the embedding output require grad so the graph connects.
    model.enable_input_require_grads()

    if torch.cuda.is_available():
        model = model.cuda()
    elif os.environ.get("AGBE_ALLOW_CPU") == "1":
        print("CUDA not available. Proceeding on CPU because AGBE_ALLOW_CPU=1.")
    else:
        # Hard stop, not a warning.
        #
        # This used to print a warning and carry on, and a run duly started
        # training 948 examples on CPU. It does not crash and it does not obviously
        # hang: it just crawls, and from the outside a progress bar sitting at 0%
        # looks identical to a deadlock. Roughly 40 minutes were lost deciding
        # whether it was stuck before anyone read two cells up.
        #
        # A silent 30x slowdown is worse than a crash, so this now refuses.
        raise SystemExit(
            "\nSTOP: CUDA is not available, so this would train on CPU and take\n"
            "hours rather than 15 minutes.\n\n"
            f"  torch  : {torch.__version__}\n"
            f"  cuda   : {torch.cuda.is_available()}\n\n"
            "Cause: llama.cpp's requirements-convert_hf_to_gguf.txt installs a\n"
            "torch+cpu wheel and pins transformers down with it. Once that has run\n"
            "in a session, the GPU build is gone for the rest of that container.\n\n"
            "Fix: Session -> Restart session (not just clear outputs), then Run All\n"
            "top to bottom. Training is cell 6, the llama.cpp install is cell 8, so\n"
            "the order is already correct as long as nothing is run out of turn.\n"
            "Check that cell 1 prints 'cuda True' before continuing.\n\n"
            "To train on CPU deliberately anyway, set AGBE_ALLOW_CPU=1.\n")

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
        strip_oversized_tokens(merged_dir)
        print(f"merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
