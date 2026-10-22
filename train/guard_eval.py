#!/usr/bin/env python3
"""Model plus guard: score recorded model answers after the guard has reviewed them.

The model alone was scored on the held-out cases. The product is the model behind the guard, which removes sentences the sources do not
support (and any chemical rate) and replaces an answer with nothing left by the fixed "I do not have a source" reply. This applies the
guard's own `review_answer` to the answers a model already gave, so no model run is needed, and scores the result with the same scorer.

    python3 train/guard_eval.py --eval rag_eval.jsonl --rows rageval_gguf_qwen17.json --out guard_qwen17.json
"""
from __future__ import annotations

import argparse, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "train"))
from rag_eval import mark_leaky, score, summarise  # noqa: E402
from gguf_eval import case  # noqa: E402
from serve.guard import review_answer  # noqa: E402


def apply_guard(recs, answers, threshold: float = 0.6, k: int = 0, max_words: int = 0):
    """k and max_words must be the prompt setting the answers were produced with: the answer's [n] citations index the ranked, trimmed
    passages the model was shown, and the guard checks against the full text of those same passages."""
    rows, changed = [], 0
    for rec0, ans in zip(recs, answers):
        _, rec = case(rec0, k, max_words)
        advice = review_answer(ans, rec["sources"], threshold)
        final = advice.answer
        changed += final.strip() != ans.strip()
        rows.append({"kind": rec["kind"], "question": rec["question"], "answer": final, "guard": advice.status, **score(rec, final)})
    return rows, changed


def mark_rows(rows, recs):
    """Rows saved before the leaky flag existed: attach it by position."""
    return [dict(r, leaky=rec.get("leaky")) if rec["kind"] == "abstain" else r for r, rec in zip(rows, recs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--out"); ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--sources", type=int, default=0); ap.add_argument("--max-words", type=int, default=0)
    a = ap.parse_args()
    recs = mark_leaky([json.loads(l) for l in open(a.eval) if l.strip()])
    model = json.load(open(a.rows))
    answers = [r["answer"] for r in model["rows"]]
    recs = recs[: len(answers)]
    rows, changed = apply_guard(recs, answers, a.threshold, a.sources, a.max_words)
    after = summarise(rows)
    print(model.get("model"), "answers changed by the guard:", changed, "of", len(rows))
    keys = ["grounded_pass", "supported_ratio", "valid_citation", "right_source", "wrongly_abstained", "abstain_pass", "abstain_clean_pass"]
    print(f"{'':26}" + "".join(f"{k[:14]:>16}" for k in keys))
    before = summarise(mark_rows(model["rows"], recs))
    print(f"{'model alone':26}" + "".join(f"{before[k]:>16}" for k in keys))
    print(f"{'model plus guard':26}" + "".join(f"{after[k]:>16}" for k in keys))
    if a.out:
        json.dump({"model": model.get("model"), "alone": before, "with_guard": after, "changed": changed, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
