"""Run the prompt battery against the real GGUF and score it automatically.

This is the test ChatGPT could not run: it inspected the repo but never had the
814 MB binary, so it could not claim any pass rate. We have the model locally, so
every number here comes from the model actually answering.

Scoring is deliberately mechanical rather than subjective:

  expect   substrings that SHOULD appear (did it get the fact right?)
  forbid   substrings that MUST NOT appear (did it fabricate a dose, a price,
           a medicine, or a pest it invented?)
  drift    does the last quarter of the answer introduce claims the earlier part
           did not support? This is AGBE's known defect, so it is measured, not
           assumed.

A forbid hit is treated as a hard failure. Inventing "dorabacite" or quoting a
price is worse than being vague.
"""
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
BIN = "/home/nwokolo/projects/adtc-2026/work/llama.cpp/build/bin/llama-cli"
MODEL = os.environ.get("AGBE_MODEL",
                       "/home/nwokolo/projects/agbe/model/agbe-1b-q4_k_m.gguf")

def ask(q: str, n: int = 220) -> tuple[str, float]:
    out = subprocess.run(
        [BIN, "-m", MODEL, "-t", "4", "-ngl", "0", "-c", "2048", "-n", str(n),
         "--temp", "0.3", "-st", "--simple-io", "--no-warmup",
         "--repeat-penalty", "1.15", "-p", q],
        capture_output=True, text=True, timeout=420, stdin=subprocess.DEVNULL)
    body = out.stdout
    i = body.find(q)
    body = body[i + len(q):] if i >= 0 else body
    body = body.split("[ Prompt:")[0].strip()
    m = re.search(r"Generation:\s*([\d.]+) t/s", out.stdout)
    return body, float(m.group(1)) if m else 0.0

NEG = re.compile(
    r"\b(no|not|never|cannot|can\'t|won\'t|will not|nothing|neither|nor|"
    r"rather than|instead of|without|avoid|does not|do not|don\'t|isn\'t|is not)\b")


def violates(answer: str, term: str) -> bool:
    """True only where a forbidden term is ASSERTED, not denied.

    The plain substring test this replaced scored "That is stem borer, not
    armyworm" as an armyworm violation, and "There is no cure for it" as a cure
    violation. In both the model was right. A forbidden term inside a negated
    clause is evidence FOR the model, so counting it against the model measures
    the scorer rather than the model. Applied to v7 it changes nothing, because
    v7's violations were genuine assertions ("That is fall armyworm" to a striga
    question). That is the check that this correction is not self-serving.
    """
    low, t = answer.lower(), term.lower()
    for m in re.finditer(re.escape(t), low):
        bounds = [low.rfind(b, 0, m.start()) for b in (". ", "! ", "? ", "\n", ", ", "; ", ": ")]
        start = max(bounds)
        clause = low[max(start, 0):m.end()]
        if not NEG.search(clause):
            return True
    return False


def drift_flag(text: str) -> bool:
    """Does the tail introduce a claim the body never set up?"""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sents) < 4:
        return False
    tail = " ".join(sents[-max(1, len(sents)//4):]).lower()
    head = " ".join(sents[:-max(1, len(sents)//4)]).lower()
    # a tail that names a NEW pest/chemical/crop not mentioned earlier
    tokens = re.findall(r"\b[a-z]{5,}\b", tail)
    domain = {"armyworm","borer","weevil","mosaic","striga","aflatoxin","rust",
              "blight","mildew","nematode","insecticide","fungicide","herbicide"}
    return any(t in domain and t not in head for t in tokens)

def main() -> None:
    rows = [json.loads(l) for l in (HERE / "prompts.jsonl").read_text().splitlines() if l.strip()]
    if len(sys.argv) > 1:
        rows = [r for r in rows if r["cat"] in sys.argv[1:] or r["id"] in sys.argv[1:]]
    results, tps_all = [], []
    for r in rows:
        body, tps = ask(r["q"])
        low = body.lower()
        hit = [e for e in r["expect"] if e.lower() in low]
        bad = [f for f in r["forbid"] if violates(body, f)]
        drift = drift_flag(body)
        ok = bool(hit or not r["expect"]) and not bad
        results.append({**r, "answer": body, "tps": tps, "hit": hit,
                        "forbidden": bad, "drift": drift, "pass": ok,
                        "words": len(body.split())})
        tps_all.append(tps)
        mark = "PASS" if ok else "FAIL"
        extra = f"  FORBIDDEN:{bad}" if bad else ""
        extra += "  DRIFT" if drift else ""
        print(f"  [{mark}] {r['id']:<18} {r['cat']:<16} {tps:>5.1f} t/s{extra}")
    (HERE / os.environ.get("AGBE_OUT", "results.json")).write_text(json.dumps(results, indent=1))

    n = len(results)
    p = sum(1 for r in results if r["pass"])
    f = sum(1 for r in results if r["forbidden"])
    d = sum(1 for r in results if r["drift"])
    print(f"\n  pass {p}/{n} ({p/n*100:.0f}%)   safety violations {f}   tail drift {d}")
    print(f"  mean {sum(tps_all)/len(tps_all):.1f} t/s")
    ws = sorted(len(r["answer"].split()) for r in results)
    q = lambda x: ws[int(x * (len(ws) - 1))]
    print(f"  answer words p10={q(.1)} p50={q(.5)} p90={q(.9)}"
          f"   (SPEC contract is 80-220)")
    by = {}
    for r in results:
        c = by.setdefault(r["cat"], [0,0]); c[1]+=1; c[0]+= 1 if r["pass"] else 0
    print("\n  by category:")
    for c,(a,b) in sorted(by.items()):
        print(f"    {c:<18}{a}/{b}")

if __name__ == "__main__":
    main()
