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
    "85.15": "S_eff is 85.50 on the binary reading, 85.16 on the decimal; 85.15 is neither",
    "982 MB": "steady RSS is 987.86 in submission.json, so 988",
    "1.01 GB": "quote memory in MB; 1,039 MB is 1.01 GiB, not 1.01 GB",
    "22.8 | 1.26": "stale Llama selection figure; the measured run is 24.4",
    "24.9 | 1.69": "stale 1.5B selection figure; a 1.5B cannot outrun a 1B",
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
                    print(f"  FAIL  {rel}:{i}  '{needle}': {why}")
                    print(f"        {line.strip()[:100]}")
                    bad += 1
    # Arithmetic, not strings. The banned-substring pass cannot catch a document
    # whose numbers are individually plausible and jointly impossible: REPORT.md
    # printed S_eff 85.15 directly above an engineering subtotal of 47.10, and
    # 85.15 implies 47.03. FINAL.json is the source, so its own sums must close.
    eng = 0.30 * F["s_perf_provisional"] + 0.20 * F["s_eff"]
    if abs(eng - F["engineering_points"]) > 0.05:
        print(f"  FAIL  FINAL.json: 0.30*{F['s_perf_provisional']} + 0.20*{F['s_eff']}"
              f" = {eng:.2f}, but engineering_points says {F['engineering_points']}")
        bad += 1
    if abs((F["engineering_points"] - 10) - F["engineering_points_with_thermal_penalty"]) > 0.05:
        print("  FAIL  FINAL.json: the thermal penalty is not exactly 10 points apart")
        bad += 1
    # S_eff must follow from the measured peak on the binary reading of 7 GB
    derived = (7168.0 - F["peak_rss_mb"]) / 7168.0 * 100
    if abs(derived - F["s_eff"]) > 0.05:
        print(f"  FAIL  FINAL.json: peak {F['peak_rss_mb']} MB gives S_eff"
              f" {derived:.2f}, but s_eff says {F['s_eff']}")
        bad += 1

    # FINAL.json is the source of truth for the docs, but nothing checked it
    # against the telemetry it summarises. Memory figures come from the profiler,
    # so they must match submission.json exactly or the manifest is fiction.
    telem = json.loads((ROOT / "submission.json").read_text())
    for key, path in (("peak_rss_mb", ("memory", "peak_rss_mb")),
                      ("steady_rss_mb", ("memory", "steady_state_rss_mb")),
                      ("tokens_per_second", ("throughput", "tokens_per_second_generation"))):
        node = telem
        for step in path:
            node = node[step]
        if abs(node - F[key]) > 0.011:
            print(f"  FAIL  FINAL.json {key}={F[key]} but submission.json says {node}")
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
