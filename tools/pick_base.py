#!/usr/bin/env python3
"""Choose the base model from bake-off results, by a rule written down before seeing the numbers.

Only the retrieval ("rag") runs count, because that is the production shape. Rank by:
  1. fewest payload leaks (a model that leaks a dose or a medicine is out whatever else it does),
  2. most prompts passed,
  3. lowest share of unsupported claims,
  4. smaller model, because the edge profile has a memory budget.
Prints the ranking and writes chosen_base.json with the best model and the best SMALL one (about 2B parameters or less), so the
edge and server profiles can each be trained.

    python3 tools/pick_base.py train/bakeoff_results*/summary.json --out chosen_base.json
"""
import argparse, glob, json, re, sys


def size_b(name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name.lower().replace("-", " ").replace("_", " "))
    return float(m.group(1)) if m else 99.0


def rank(summaries: dict):
    rows = []
    for key, s in summaries.items():
        model, mode = key.rsplit(" [", 1)
        if mode.rstrip("]") != "rag":
            continue
        rows.append({"model": model, "pass": s["pass"], "n": s["n"], "leaks": s["leaks"], "over_refused": s["over_refused"], "drift": s["drift"],
                     "unsupported": s.get("mean_unsupported_claim_rate", 1.0), "size_b": size_b(model)})
    rows.sort(key=lambda r: (r["leaks"], -r["pass"], r["unsupported"], r["size_b"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--out", default="chosen_base.json")
    a = ap.parse_args()
    merged = {}
    for pat in a.summaries:
        for f in glob.glob(pat):
            merged.update(json.load(open(f)))
    rows = rank(merged)
    if not rows:
        sys.exit("no rag results found")
    print(f"{'model':45s} {'pass':>9s} {'leaks':>5s} {'over':>5s} {'drift':>5s} {'unsupp':>7s}")
    for r in rows:
        print(f"{r['model']:45s} {r['pass']:>4d}/{r['n']:<4d} {r['leaks']:>5d} {r['over_refused']:>5d} {r['drift']:>5d} {r['unsupported']:>7.3f}")
    small = [r for r in rows if r["size_b"] <= 2.1]
    choice = {"best": rows[0]["model"], "best_small": (small[0]["model"] if small else None), "ranking": rows}
    json.dump(choice, open(a.out, "w"), indent=1)
    print("\nbest:", choice["best"], "| best small:", choice["best_small"])


if __name__ == "__main__":
    main()
