"""Report exactly WHERE an out-of-range token id comes from.

Two attempts to strip this by editing files reported success and the assertion
still fired, which means the id is not (only) in the files being edited. Rather
than guess a third time, print every source transformers merges into
`tokenizer.vocab` and say which one carries it.
"""
from __future__ import annotations
import json, pathlib, sys

m = pathlib.Path(sys.argv[1])
cfg = json.loads((m / "config.json").read_text())
vs = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size")
print(f"config.json vocab_size            : {vs}")

tj = json.loads((m / "tokenizer.json").read_text()) if (m / "tokenizer.json").exists() else {}
added = tj.get("added_tokens", [])
print(f"tokenizer.json added_tokens      : {len(added)} entries, "
      f"max id {max((a['id'] for a in added), default='-')}")
over = [a for a in added if a["id"] >= vs]
print(f"  over-range here                : {[a['content'] for a in over] or 'none'}")

vocab = tj.get("model", {}).get("vocab")
if isinstance(vocab, dict):
    mx = max(vocab.values()) if vocab else "-"
    print(f"tokenizer.json model.vocab (dict): {len(vocab)} entries, max id {mx}")
    print(f"  over-range here                : "
          f"{[t for t, i in vocab.items() if i >= vs] or 'none'}")
elif isinstance(vocab, list):
    print(f"tokenizer.json model.vocab (list): {len(vocab)} entries "
          f"(index is the id, so max id {len(vocab) - 1})")
else:
    print(f"tokenizer.json model.vocab       : absent (type {type(vocab).__name__})")

for name, key in (("added_tokens.json", None),
                  ("tokenizer_config.json", "added_tokens_decoder")):
    p = m / name
    if not p.exists():
        print(f"{name:<33}: absent"); continue
    d = json.loads(p.read_text())
    d = d.get(key, {}) if key else d
    ids = [int(v) if key is None else int(k) for k, v in d.items()]
    bad = [k for k, v in d.items() if (int(v) if key is None else int(k)) >= vs]
    print(f"{name:<33}: {len(d)} entries, max id {max(ids, default='-')}")
    print(f"  over-range here                : {bad or 'none'}")

print(f"\nextra files present              : "
      f"{sorted(f.name for f in m.iterdir() if 'token' in f.name.lower())}")

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(m))
    v = tok.get_vocab()
    hi = max(v.values())
    print(f"\ntransformers {tok.__class__.__name__}")
    print(f"  get_vocab() size               : {len(v)}, max id {hi}")
    print(f"  tokens at id >= {vs}      : {[t for t, i in v.items() if i >= vs]}")
    print(f"  additional_special_tokens      : "
          f"{getattr(tok, 'additional_special_tokens', None)}")
    print(f"\n  VERDICT: {'OK' if hi < vs else 'still over range'}")
except Exception as e:
    print(f"\ntransformers load failed: {type(e).__name__}: {e}")
