#!/usr/bin/env python3
"""How much of the attack batteries do the guard's deterministic rules stop before any model runs?

    python3 eval/measure_guard.py

Dev set: eval/adversarial.jsonl and eval/prompts.jsonl (the rules were written while looking at these).
Holdout: eval/guard_holdout.jsonl, written BEFORE the rules were extended and never used to tune them.
A legitimate question blocked by a rule is a false positive. Prompts whose contract allows a refusal
(refusal_ok) are counted as neither.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.guard import hard_rule  # noqa: E402


def load(f):
    return [json.loads(l) for l in open(ROOT / "eval" / f) if l.strip()]


def report(name, rows, attack_key):
    att = [r for r in rows if attack_key(r)]
    leg = [r for r in rows if not attack_key(r) and not r.get("refusal_ok")]
    caught = [r for r in att if hard_rule(r["q"])]
    fp = [r for r in leg if hard_rule(r["q"])]
    print(f"{name:34s} attacks stopped {len(caught):>3}/{len(att):<3}  legitimate blocked {len(fp):>2}/{len(leg)}")
    return [r["id"] for r in att if not hard_rule(r["q"])], [(r["id"], hard_rule(r["q"])) for r in fp]


if __name__ == "__main__":
    for f in ("adversarial.jsonl", "prompts.jsonl"):
        report(f"dev: {f}", load(f), lambda r: r.get("must_refuse"))
    missed, fp = report("holdout: guard_holdout.jsonl", load("guard_holdout.jsonl"), lambda r: r["attack"])
    print("holdout misses go to the model and the claim checker:", missed)
