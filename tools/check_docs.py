"""Validate every public document against FINAL.json.

The artifact checker verified the model's identity; nothing verified the
documents'. That gap let README, REPORT, BUILDS and the site drift apart while
`check_submission.sh` still printed READY TO SUBMIT: at various points the report
claimed a 812-conversation corpus, the ledger called v11 the shipped build, and
the site quoted 23 tok/s. Each was fixed by hand and another instance survived
somewhere else.

So the final state lives in FINAL.json and this script rejects any public file
that contradicts it. Fix FINAL.json first, then the prose.
"""
from __future__ import annotations
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
F = json.loads((ROOT / "FINAL.json").read_text())

DOCS = ["README.md", "REPORT.md", "BUILDS.md", "docs/devpost-story.md",
        "corpus/SPEC.md", "metadata.json",
        "../agbe-site/index.html", "../agbe-site/notes/index.html"]

# Strings that contradict the shipped build wherever they appear as a claim.
# The value is why it is wrong, printed on a hit.
BANNED = {
    "812 conversations": "v9-era corpus; final is %d" % F["corpus_conversations"],
    "| 1,020 | 812 |": "diversity table still shows the v9 corpus",
    "v11 is the shipped": "v13 is shipped",
    "capped at 15": "throughput is relative to the fastest submission",
    "caps at 15": "throughput is relative to the fastest submission",
    "above 15 tok": "throughput is relative to the fastest submission",
    "earns nothing": "surplus throughput is never worthless under the real formula",
    "stops paying at 15": "throughput is relative to the fastest submission",
    "22-prompt battery": "the batteries are 66-prompt and 92-prompt",
    "26 tokens a second": "final throughput is %s" % F["tokens_per_second"],
    "26.23": "v8 throughput",
    "29.29": "v11 throughput",
    "23 tok/s": "stale throughput",
    "3.8x average": "final sentence reuse is %sx" % F["corpus_sentence_reuse"],
    "REPLACE_WITH": "placeholder",
    "q5_0": "internal tensor detail, not for publication",
    "q8_0": "internal tensor detail, not for publication",
    "104 optimiser": "v10-era step count; final is %d" % F["optimiser_steps"],
    "v11 is the shipped": "%s is shipped" % F["build"],
    "v11 is the ship": "%s is shipped" % F["build"],
}
# Historical mentions that are legitimate when the sentence marks them as past.
# A version number alone is NOT enough: "v11 is the shipped build" mentions v11
# and is still a false present-tense claim, which an earlier version of this
# regex waved through. Present-tense assertions are checked first and never
# exempted.
# A claim about the SHIPPED build is never exempt, however it is phrased.
PRESENT_CLAIM = re.compile(
    r"\bis the shipped\b|\bis shipped\b|\bwe ship\b|\bthe shipped (model|build)\b|"
    r"\bfinal (model|build)\b|\bv13\b", re.I)

# Historical exemptions must NAME the past explicitly. A bare "was" or "were" is
# not enough: "the shipped model was trained on 812 conversations" is a false
# present-tense claim wearing a past-tense verb, and an earlier version of this
# regex passed exactly that, along with "throughput was capped at 15 tok/s".
# Both were caught only by deliberately injecting them and watching this script
# say DOCS CONSISTENT.
HISTORICAL_OK = re.compile(
    r"originally|at the time|we then believed|we (then |initially )?read|historic|"
    r"selection-time|misread|first version|used to|previously|"
    r"\bv[1-9]\b|\bv1[0-2]\b|earlier (version|draft|build|reading)|"
    r"no longer|until|before we|turned out", re.I)

def main() -> int:
    bad = 0
    for rel in DOCS:
        p = ROOT / rel
        if not p.exists():
            print(f"  MISSING  {rel}"); bad += 1; continue
        lines = p.read_text().split("\n")
        for i, line in enumerate(lines, 1):
            for needle, why in BANNED.items():
                if needle not in line:
                    continue
                if PRESENT_CLAIM.search(line) or not HISTORICAL_OK.search(line):
                    print(f"  FAIL  {rel}:{i}  '{needle}' — {why}")
                    print(f"        {line.strip()[:100]}")
                    bad += 1
    # positive checks: the shipped facts must actually appear where they matter
    readme = (ROOT / "README.md").read_text()
    for label, val in (("throughput", str(F["tokens_per_second"])),
                       ("corpus", str(F["corpus_conversations"])),
                       ("team_id", F["team_id"])):
        if val not in readme:
            print(f"  FAIL  README.md missing the final {label} ({val})"); bad += 1
    print(f"\n  {'DOCS CONSISTENT with FINAL.json' if bad == 0 else str(bad) + ' contradiction(s)'}")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
