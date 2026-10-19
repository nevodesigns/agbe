#!/usr/bin/env python3
"""Print what the training data actually looks like, so a log is enough to judge its quality.

    python3 train/diagnose_data.py --claims claims.jsonl --sft sft.jsonl
"""
import argparse, collections, json, pathlib, random, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from claim_check import check_claim, split_claims  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--claims"); ap.add_argument("--sft")
a = ap.parse_args()
rnd = random.Random(11)
if a.claims and pathlib.Path(a.claims).exists():
    cl = [json.loads(l) for l in open(a.claims)]
    print("CLAIMS:", dict(collections.Counter(c["status"] for c in cl)), "| rejection reasons:", dict(collections.Counter(r for c in cl for r in c["reasons"]).most_common(6)))
    print("passages with an accepted claim:", len({c["passage_id"] for c in cl if c["status"] == "accepted"}), "| documents:", len({c["handle"] for c in cl if c["status"] == "accepted"}))
    acc = [c for c in cl if c["status"] == "accepted"]
    print("\n12 random accepted claims (claim | quote start):")
    for c in rnd.sample(acc, min(12, len(acc))):
        print(" -", c["claim"][:170], "|", c["quote"][:90])
if a.sft and pathlib.Path(a.sft).exists():
    ex = [json.loads(l) for l in open(a.sft)]
    print("\nSFT EXAMPLES:", len(ex), dict(collections.Counter(e["kind"] for e in ex)))
    g = [e for e in ex if e["kind"] == "grounded"]
    ratios = []
    for e in g:
        srcs = re.findall(r"^\[\d+\] \(.*?\) (.*)$", e["messages"][0]["content"], re.M)
        sents = [re.sub(r"\[\d+\]", "", s) for s in split_claims(e["messages"][-1]["content"])]
        if sents:
            ratios.append(sum(check_claim(s, srcs).supported for s in sents) / len(sents))
    if ratios:
        print(f"grounded answers: share of sentences supported by their own sources, mean {sum(ratios)/len(ratios):.3f}, minimum {min(ratios):.2f}")
    qs = collections.Counter(e["messages"][1]["content"] for e in g)
    print("distinct grounded questions:", len(qs), "| most repeated:", qs.most_common(1)[0][1] if qs else 0)
    for kind in ("grounded", "abstain", "refusal"):
        pool = [e for e in ex if e["kind"] == kind]
        for e in rnd.sample(pool, min(3, len(pool))):
            print(f"\n[{kind}] USER: {e['messages'][1]['content'][:150]}\n[{kind}] ASSISTANT: {e['messages'][-1]['content'][:320]}")
