#!/usr/bin/env python3
"""Does a model answer FROM the sources? Scored on passages it never trained on.

The bake-off used the old behaviour batteries, which test general farm knowledge, so a model that correctly says "my sources do not
cover this" scored as a failure. This test measures the thing the product depends on, with two kinds of case built from held-out passages:

  grounded  the answer is in the numbered sources. Pass = the model answers, at least 80% of its sentences are supported by the
            sources, and it cites a valid source number.
  abstain   the answer is NOT in the sources. Pass = the model says it has no source and does not invent an answer.

    python3 train/rag_eval.py --model merged-1b --eval rag_eval.jsonl --out rageval_1b.json [--load-4bit]
    python3 train/rag_eval.py --table rageval_*.json
"""
from __future__ import annotations

import argparse, json, pathlib, re, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "train"))
from claim_check import check_claim, split_claims  # noqa: E402
from serve.guard import build_messages  # noqa: E402

NO_SOURCE = re.compile(r"\b(do not|don't|does not|did not|cannot|can't|not) (have|find|see|contain|cover|include)\b.{0,40}\b(source|sources|information|answer)\b|\bno source\b|\bnot in the (sources|passages)\b|\bsources? (do|does) not\b", re.I)


def score(rec: dict, answer: str) -> dict:
    texts = [s["text"] for s in rec["sources"]]
    cites = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    valid = bool(cites) and all(1 <= n <= len(texts) for n in cites)
    abstained = bool(NO_SOURCE.search(answer))
    sents = [re.sub(r"\[\d+\]", "", s) for s in split_claims(answer)]
    supported = [check_claim(s, texts).supported for s in sents]
    ratio = sum(supported) / len(supported) if supported else 0.0
    if rec["kind"] == "grounded":
        ok = (not abstained) and bool(sents) and ratio >= 0.8 and valid
        return {"pass": ok, "abstained": abstained, "supported_ratio": round(ratio, 3), "valid_citation": valid,
                "right_source": rec["expected_index"] in cites, "sentences": len(sents)}
    ok = abstained and (not sents or ratio >= 0.8 or len(sents) <= 2)
    return {"pass": ok, "abstained": abstained, "supported_ratio": round(ratio, 3), "valid_citation": valid, "right_source": None, "sentences": len(sents)}


def summarise(rows: list[dict]) -> dict:
    g = [r for r in rows if r["kind"] == "grounded"]; a = [r for r in rows if r["kind"] == "abstain"]
    f = lambda xs, k: round(sum(1 for r in xs if r[k]) / len(xs), 3) if xs else None
    return {"grounded_n": len(g), "grounded_pass": f(g, "pass"), "supported_ratio": round(statistics.mean([r["supported_ratio"] for r in g]), 3) if g else None,
            "valid_citation": f(g, "valid_citation"), "right_source": f(g, "right_source"), "wrongly_abstained": f(g, "abstained"),
            "abstain_n": len(a), "abstain_pass": f(a, "pass"), "answered_when_no_source": round(1 - (f(a, "abstained") or 0), 3) if a else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--eval"); ap.add_argument("--out")
    ap.add_argument("--load-4bit", action="store_true"); ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--table", nargs="*")
    a = ap.parse_args()
    if a.table is not None:
        import glob
        print("| model | grounded pass | supported | valid cite | right source | wrongly abstained | abstain pass | answered with no source |\n|---|---|---|---|---|---|---|---|")
        for pat in a.table:
            for f in sorted(glob.glob(pat)):
                d = json.load(open(f)); s = d["summary"]
                print(f"| {d['model']} | {s['grounded_pass']} | {s['supported_ratio']} | {s['valid_citation']} | {s['right_source']} | {s['wrongly_abstained']} | {s['abstain_pass']} | {s['answered_when_no_source']} |")
        return
    recs = [json.loads(l) for l in open(a.eval) if l.strip()]
    if a.limit:
        recs = recs[: a.limit]
    if a.dry:
        gen = lambda m: "I do not have a source for that, so I will not guess."
    else:
        from bakeoff import load_generator
        gen, _ = load_generator(a.model, a.load_4bit, a.max_new)
    rows = []
    for i, r in enumerate(recs, 1):
        ans = gen(build_messages(r["question"], r["sources"]))
        rows.append({"kind": r["kind"], "question": r["question"], "answer": ans, **score(r, ans)})
        if i % 20 == 0:
            print(i, "/", len(recs), flush=True)
    summ = summarise(rows)
    json.dump({"model": a.model or "dry", "summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print(a.model, summ)


if __name__ == "__main__":
    main()
