#!/usr/bin/env python3
"""Source-grounded LoRA fine-tune for AGBE v2 (QLoRA, built for a free Kaggle T4).

UNTESTED UNTIL IT RUNS ON A GPU: it was written without one. Run the data check first (--dry-data), then a few steps
(--max-steps 20) before a full run.

Input is the JSONL from tools/build_grounded_sft.py: chat messages with the sources in the system prompt. Loss is counted on
the assistant's reply only (prompt/completion format), because training on the sources would teach the model to recite them.

    python3 train/train_grounded.py --data sft.jsonl --base google/gemma-3-1b-it --out adapter-1b --dry-data
    python3 train/train_grounded.py --data sft.jsonl --base google/gemma-3-1b-it --out adapter-1b
    python3 train/train_grounded.py --data sft.jsonl --base Qwen/Qwen3-4B-Instruct-2507 --out adapter-4b --rank 16 --lr 1e-4

Afterwards: merge the adapter, convert with llama.cpp's convert_hf_to_gguf.py, quantise (Q4_K_M, or the QAT route), then run
eval/run_eval.py on the GGUF and train/bakeoff.py-style checks before anything ships. Keep the training log, the adapter and the
checksums (see provenance/ for the v16 layout).
"""
from __future__ import annotations

import argparse, json, random


def to_prompt_completion(tok, rec):
    msgs = rec["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    if tok.bos_token and prompt.startswith(tok.bos_token):  # the trainer adds BOS itself, so a second one would be a double BOS
        prompt = prompt[len(tok.bos_token):]
    return {"prompt": prompt, "completion": msgs[-1]["content"] + (tok.eos_token or "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="google/gemma-3-1b-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=0, help="default 2 x rank")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--val-share", type=float, default=0.05)
    ap.add_argument("--dry-data", action="store_true", help="tokenise and report lengths only")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    random.Random(3).shuffle(rows)
    data = [to_prompt_completion(tok, r) for r in rows]
    lens = sorted(len(tok(d["prompt"] + d["completion"]).input_ids) for d in data)
    print(f"{len(data)} examples; tokens p50={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} max={lens[-1]}; over {a.max_len}: {sum(l > a.max_len for l in lens)}")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print("mix:", kinds)
    if a.dry_data:
        print(data[0]["prompt"][-300:], "\n---\n", data[0]["completion"])
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    nval = max(8, int(len(data) * a.val_share))
    train_ds, val_ds = Dataset.from_list(data[nval:]), Dataset.from_list(data[:nval])
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb, device_map="auto", attn_implementation="eager")
    lora = LoraConfig(r=a.rank, lora_alpha=a.alpha or 2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
                    per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum, max_length=a.max_len, fp16=True,
                    logging_steps=5, eval_strategy="steps", eval_steps=50, save_strategy="steps", save_steps=100, save_total_limit=2,
                    completion_only_loss=True, report_to="none", seed=3)
    SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds, peft_config=lora, processing_class=tok).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("adapter saved to", a.out)


if __name__ == "__main__":
    main()
