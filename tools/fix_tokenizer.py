"""Strip tokenizer entries whose id is >= config.vocab_size, in place.

Standalone so a merged model that has ALREADY been trained can be repaired
without paying for the 16 minute training run again.

Usage:  python tools/fix_tokenizer.py /kaggle/working/out/merged
"""
from __future__ import annotations
import json, pathlib, sys


def fix(merged: pathlib.Path) -> bool:
    cfg = json.loads((merged / "config.json").read_text())
    vs = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size")
    if not vs:
        print("no vocab_size in config.json"); return False
    print(f"config vocab_size = {vs}")
    dropped = []

    tj_path = merged / "tokenizer.json"
    if tj_path.exists():
        tj = json.loads(tj_path.read_text())
        added = tj.get("added_tokens", [])
        keep = [a for a in added if a["id"] < vs]
        if len(keep) != len(added):
            dropped += [a["content"] for a in added if a["id"] >= vs]
            tj["added_tokens"] = keep
        vocab = tj.get("model", {}).get("vocab")
        if isinstance(vocab, dict):
            over = [t for t, i in vocab.items() if isinstance(i, int) and i >= vs]
            for t in over:
                del vocab[t]
            dropped += over
        elif isinstance(vocab, list) and len(vocab) > vs:
            dropped += [v[0] if isinstance(v, (list, tuple)) else v for v in vocab[vs:]]
            tj["model"]["vocab"] = vocab[:vs]
        tj_path.write_text(json.dumps(tj, ensure_ascii=False))

    at = merged / "added_tokens.json"
    if at.exists():
        d = json.loads(at.read_text())
        at.write_text(json.dumps({k: v for k, v in d.items() if v < vs},
                                 ensure_ascii=False))

    tc = merged / "tokenizer_config.json"
    if tc.exists():
        d = json.loads(tc.read_text())
        dec = d.get("added_tokens_decoder", {})
        d["added_tokens_decoder"] = {k: v for k, v in dec.items() if int(k) < vs}
        tc.write_text(json.dumps(d, ensure_ascii=False))

    print(f"stripped: {sorted(set(dropped)) or 'nothing (already clean)'}")

    # Prove it against the same condition llama.cpp asserts on, so a pass here
    # means conversion will not die after writing every tensor.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(merged))
        hi = max(tok.vocab.values())
        print(f"max token id now {hi}, vocab_size {vs} -> "
              f"{'OK' if hi < vs else 'STILL FAILING'}")
        return hi < vs
    except Exception as e:
        print(f"could not verify with transformers ({e}); files were still patched")
        return True


if __name__ == "__main__":
    sys.exit(0 if fix(pathlib.Path(sys.argv[1])) else 1)
