#!/usr/bin/env python3
"""Scan every public document for numbers that contradict FINAL.json.

`check_docs.py` bans strings we already know are wrong. That is reactive: it only
ever catches the mistake after someone has found it by hand, and it caught none
of the three that mattered. "104 optimiser steps" survived three audits because
the phrase wrapped across a line break. "13.0M trainable" survived all of them
because nobody had thought to ban it.

This inverts the test. For each quantity the manifest defines, it finds EVERY
number stated in that quantity's context anywhere in the public docs, and fails
on any that disagrees. It does not need to know what the wrong value is, which
is the whole point: it catches values nobody has seen yet.

Whitespace is normalised across the entire document first, so a claim that wraps
across lines is scanned exactly like one that does not.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
F = json.loads((ROOT / "FINAL.json").read_text())

DOCS = ["README.md", "REPORT.md", "BUILDS.md", "docs/devpost-story.md",
        "corpus/SPEC.md", "metadata.json",
        "../agbe-site/index.html", "../agbe-site/notes/index.html"]

# (label, regex with one capture group, expected value, tolerance)
# The regex must be tight enough that it only fires on a claim about THIS
# quantity. A loose pattern that matches unrelated prose is worse than no check,
# because the noise trains you to ignore the output.
def num(x):
    return float(str(x).replace(",", ""))


CHECKS = [
    ("optimiser steps", r"(\d+)\s+optimiser steps", num(F["optimiser_steps"]), 0),
    ("corpus conversations", r"(\d[\d,]*)\s+conversations", num(F["corpus_conversations"]), 0),
    ("unique sentences", r"(\d[\d,]*)\s+unique sentences", num(F["corpus_unique_sentences"]), 0),
    ("sentence reuse", r"reuse[^.\n]{0,20}?(\d+\.\d+)\s*(?:x|×)", num(F["corpus_sentence_reuse"]), 0.05),
    ("throughput", r"(\d+\.\d+)\s*(?:tok/s|tokens per second)", num(F["tokens_per_second"]), 0.005),
    ("peak RSS MB", r"(\d[\d,]*)\s*MB\s+peak", round(num(F["peak_rss_mb"])), 1),
    ("model bytes", r"(\d{3},\d{3},\d{3})\s*bytes", num(F["model_bytes"]), 0),
    ("arc_easy", r"arc[_ ]easy[^.\n]{0,40}?(0\.\d+)", num(F["arc_easy"]), 0.001),
    ("S_eff", r"S_eff[^.\n|]{0,30}?(\d{2}\.\d+)", num(F["s_eff"]), 0.01),
    ("LoRA rank", r"(?:rank|r)\s*=?\s*(\d+)\s*(?:across|,\s*alpha)", num(F["lora_r"]), 0),
    ("trainable params", r"(\d+\.\d+)M\s+trainable", num(F["trainable_params"]) / 1e6, 0.05),
    ("trainable params", r"trainable\s*\|\s*(\d+\.\d+)M", num(F["trainable_params"]) / 1e6, 0.05),
    ("base params", r"of\s+([\d,]+(?:\.\d+)?)M\b", num(F["base_params"]) / 1e6, 15),
    # "Gemma 3 1B banks 47.5 of 50" sat in a site caption through three
    # audits because no pattern looked at engineering subtotals at all.
    ("engineering points", r"(\d{2}\.\d)\s*(?:of|/)\s*50\b", num(F["engineering_points"]), 0.05),
]

# Fractions must match exactly wherever they appear as a battery result.
FRACTIONS = [
    ("behaviour battery", r"\b(\d+)/66\b", F["behaviour_battery"]),
    ("hostile battery", r"\b(\d+)/92\b", F["hostile_battery"]),
    ("attacks withstood", r"\b(\d+)/62\b", F["attacks_withstood"]),
]

# Lines that are explicitly about an earlier build or an earlier reading are
# allowed to carry other numbers; that is what a development history is.
# Build names are handled by owning_build(), not here. Leaving a bare "v8" in
# this list is what let "the shipped build, v13, runs ... at 104 optimiser steps
# against v8's 96" exempt itself: one incidental mention of an old build
# excused a false claim about the current one.
HISTORICAL = re.compile(
    # an explicit earlier reading, stated as such
    r"originally|earlier|previously|at the time|"
    r"we then believed|misread|historical|stale|by mistake|used to|"
    r"before we|no longer|as we first|capped reading|as misread|selection[- ]time|"
    # a before/after comparison: the old value is the point of the sentence
    r"from \d[\d,]* to \d[\d,]*|grew|growing|made the model worse|made it worse|"
    r"fell to|rose to|dropped to|down from|up from|instead of|"
    # a named side experiment whose result is deliberately not the shipped figure
    r"pinn(?:ed|ing)|thread|safest build|best hostile|per-core|four physical cores|"
    # a candidate comparison: "of 50" there is a rival model's selection-time
    # subtotal, not the shipped model's engineering points
    r"Qwen|Llama|candidate|selection table",
    re.I)


BUILD_MARK = re.compile(r"\bv(\d{1,2})\b", re.I)


def flatten(raw):
    """Collapse whitespace but keep a map back to the source line.

    Matching needs one flat string so a claim that wraps across lines reads the
    same as an inline one. Attribution needs the original line, because a
    markdown table row IS the unit of ownership. Distance cannot substitute:
    a legitimate ledger row sits 274 characters from its build name while an
    unrelated config table sat 261 from a stray v10, so the ranges overlap and
    no threshold separates them.
    """
    parts, spans, off = [], [], 0
    for idx, line in enumerate(raw.split("\n")):
        norm = re.sub(r"\s+", " ", line).strip()
        parts.append(norm)
        spans.append((off, off + len(norm), idx, norm))
        off += len(norm) + 1
    return " ".join(parts), spans


def source_line(spans, pos):
    for start, end, idx, text in spans:
        if start <= pos <= end:
            return text
    return ""


def owning_build(line):
    """Which build owns a number, judged from its own table row or line.

    A build name claims only the figures on its own row. The training-config
    table has no build name on the Trainable row, so that row is checked against
    the manifest rather than excused as some earlier build's history.
    """
    last = None
    for m in BUILD_MARK.finditer(line):
        last = m
    return int(last.group(1)) if last else None


def is_comparison(row, expected):
    """True if the manifest's own value also appears in this row."""
    for form in (f"{expected:g}", f"{expected:,.0f}", f"{expected:.1f}", f"{expected:.2f}"):
        if form in row:
            return True
    return False


def context(text, start, end, width=240):
    lo, hi = max(0, start - width), min(len(text), end + width)
    return text[lo:hi].strip()


def main() -> int:
    bad = 0
    for rel in DOCS:
        p = ROOT / rel
        if not p.exists():
            print(f"  MISSING  {rel}")
            bad += 1
            continue
        # one flat string: a wrapped claim reads the same as an inline one
        flat, spans = flatten(p.read_text())

        for label, pattern, expected, tol in CHECKS:
            for m in re.finditer(pattern, flat, re.I):
                try:
                    got = num(m.group(1))
                except ValueError:
                    continue
                if abs(got - expected) <= tol:
                    continue
                snippet = context(flat, m.start(), m.end())
                row = source_line(spans, m.start())
                owner = owning_build(row)
                if owner is not None and owner != 13:
                    continue
                # A row carrying BOTH the old value and the manifest value is a
                # before/after comparison, which is the point of the row:
                # "| average sentence reuse | 5.6x | 4.0x |".
                if is_comparison(row, expected):
                    continue
                if HISTORICAL.search(snippet):
                    continue
                print(f"  FAIL  {rel}  {label}: found {m.group(1)}, manifest says {expected}")
                print(f"        ...{snippet}...")
                bad += 1

        for label, pattern, expected in FRACTIONS:
            want = expected.split("/")[0]
            for m in re.finditer(pattern, flat):
                if m.group(1) == want:
                    continue
                snippet = context(flat, m.start(), m.end())
                row = source_line(spans, m.start())
                owner = owning_build(row)
                if owner is not None and owner != 13:
                    continue
                if want in row.replace(m.group(0), "", 1):
                    continue
                if HISTORICAL.search(snippet):
                    continue
                print(f"  FAIL  {rel}  {label}: found {m.group(0)}, manifest says {expected}")
                print(f"        ...{snippet}...")
                bad += 1

    print(f"\n  {'NO CONTRADICTIONS against FINAL.json' if bad == 0 else str(bad) + ' contradiction(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
