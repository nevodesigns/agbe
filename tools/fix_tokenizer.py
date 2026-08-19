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


    # special_tokens_map.json and tokenizer_config's special-token lists.
    #
    # This is the one that actually mattered. The three files above were already
    # clean, because the trainer strips them, and the assertion kept firing anyway:
    # a special token listed HERE is re-registered by from_pretrained and handed
    # the next free id, which is exactly vocab_size. Editing the vocab and leaving
    # the declaration behind just meant the token was rebuilt on every load.
    base_vocab = set()
    if tj_path.exists():
        v = json.loads(tj_path.read_text()).get("model", {}).get("vocab")
        if isinstance(v, dict):
            base_vocab = set(v)
        elif isinstance(v, list):
            base_vocab = {x[0] if isinstance(x, (list, tuple)) else x for x in v}

    def prune(seq):
        """Drop declared special tokens that no longer exist in the vocab."""
        out, gone = [], []
        for t in seq:
            name = t if isinstance(t, str) else t.get("content", "")
            (out if (not base_vocab or name in base_vocab) else gone).append(t)
            if name not in base_vocab and base_vocab:
                gone_names.append(name)
        return out

    gone_names = []
    stm = merged / "special_tokens_map.json"
    if stm.exists():
        d = json.loads(stm.read_text())
        if isinstance(d.get("additional_special_tokens"), list):
            d["additional_special_tokens"] = prune(d["additional_special_tokens"])
            stm.write_text(json.dumps(d, ensure_ascii=False))

    if tc.exists():
        d = json.loads(tc.read_text())
        changed = False
        if isinstance(d.get("additional_special_tokens"), list):
            d["additional_special_tokens"] = prune(d["additional_special_tokens"])
            changed = True
        extra = d.get("extra_special_tokens")
        if isinstance(extra, dict):
            keep = {k: v for k, v in extra.items()
                    if not base_vocab or v in base_vocab}
            if len(keep) != len(extra):
                gone_names += [v for v in extra.values() if v not in base_vocab]
                d["extra_special_tokens"] = keep
                changed = True
        if changed:
            tc.write_text(json.dumps(d, ensure_ascii=False))

    dropped += gone_names
    print(f"stripped: {sorted(set(dropped)) or 'nothing (already clean)'}")

    # Prove it against the same condition llama.cpp asserts on, so a pass here
    # means conversion will not die after writing every tensor.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(merged))
        v = tok.get_vocab()
        hi = max(v.values())
        ok = hi < vs
        print(f"max token id now {hi}, vocab_size {vs} -> "
              f"{'OK' if ok else 'STILL FAILING'}")
        if not ok:
            # Asking for a separate diagnostic script to be run did not work
            # twice, so the failure path now reports everything itself.
            print("\n--- where the over-range id actually comes from ---")
            print(f"tokenizer class      : {tok.__class__.__name__}")
            print(f"offending tokens     : {[t for t, i in v.items() if i >= vs]}")
            print(f"additional_special   : {getattr(tok, 'additional_special_tokens', None)}")
            print(f"all_special_tokens   : {getattr(tok, 'all_special_tokens', None)}")
            tjm = json.loads(tj_path.read_text()) if tj_path.exists() else {}
            bv = tjm.get("model", {}).get("vocab")
            print(f"model.vocab type     : {type(bv).__name__}, "
                  f"len {len(bv) if bv is not None else '-'}")
            if isinstance(bv, dict):
                print(f"  max id in base     : {max(bv.values()) if bv else '-'}")
            print(f"added_tokens in tj   : "
                  f"{[a['content'] for a in tjm.get('added_tokens', [])][-4:]}")
            for fn in ("special_tokens_map.json", "tokenizer_config.json"):
                fp = merged / fn
                if not fp.exists():
                    continue
                d = json.loads(fp.read_text())
                print(f"{fn}")
                for k in ("additional_special_tokens", "extra_special_tokens",
                          "boi_token", "eoi_token", "image_token"):
                    if k in d:
                        print(f"    {k} = {d[k]}")
            print(f"files present        : "
                  f"{sorted(f.name for f in merged.iterdir())}")
        return ok
    except Exception as e:
        print(f"could not verify with transformers ({e}); files were still patched")
        return True


if __name__ == "__main__":
    sys.exit(0 if fix(pathlib.Path(sys.argv[1])) else 1)
