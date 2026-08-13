"""Answer composition for the ADTC agriculture corpus.

Rewritten after the first trained model failed its smoke test. The failure was
caused here, so the reasoning is worth recording.

v1 gave every answer a hardcoded lead sentence per slice and a mandatory closer.
Measured on the generated corpus: **47% of assistant turns opened with one of six
sentences**, one of them 81 times, and four closers covered most endings. A 1B
model trained on that learns the scaffolding, not the agronomy. The trained model
duly opened a question about maize pests with "This is one of the cheapest places
on a smallholding to gain or lose money" (the soil-and-storage lead) and then
invented a fungal infection.

The rules now:

  - **No constant lead sentences.** An answer opens with its own first fact. The
    facts in facts.json are already written as complete, useful sentences.
  - **Closers are occasional, not mandatory** (about one answer in four) and are
    drawn from a wider pool, so no single phrase dominates endings.
  - Several structural shapes, chosen per answer, so form varies with content.

The check that matters is not word count. It is: what fraction of answers share an
opening sentence? `generate.py` prints that, and it must stay low.
"""

from __future__ import annotations

import random
import re


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def para(*parts: str) -> str:
    out = []
    for p in parts:
        p = clean(p)
        if not p:
            continue
        if not p.endswith((".", ":", "!", "?")):
            p += "."
        out.append(p)
    return " ".join(out)


def bullets(items: list[str], numbered: bool = False) -> str:
    rows = []
    for i, item in enumerate(items, 1):
        item = clean(item).rstrip(".")
        rows.append(f"{i}. {item}." if numbered else f"- {item}.")
    return "\n".join(rows)


# A wider pool, used sparingly. In v1 four of these ended nearly every answer.
CLOSERS = [
    "Your local extension officer can confirm which varieties and products are "
    "registered in your area",
    "Keep a note of what you did and when, because next season that record beats "
    "any general advice",
    "If what you see in the field does not match this, take a sample to an "
    "extension officer rather than guessing",
    "Conditions differ between zones and soils, so treat this as a starting point",
    "Do the cheapest thing on this list first and give it two weeks before "
    "spending more",
    "Compare a treated strip against an untreated one, otherwise you cannot tell "
    "whether the weather did it",
    "Ask the older farmers in your area what has worked there, since local "
    "experience beats a general rule",
    "Check this against what your neighbours are seeing before you act on a whole "
    "field",
]

CHEMICAL_CAUTION = [
    "If you spray, use a product registered for this pest on this crop and follow "
    "the label rate exactly. Do not mix it stronger than the label says",
    "Any chemical control follows the product label for rate, timing and "
    "pre-harvest interval. Registered products differ between countries, so "
    "confirm locally",
]

# How often an answer gets a closing line at all.
CLOSER_PROB = 0.25


def pick(rng: random.Random, pool: list[str]) -> str:
    return rng.choice(pool)


def maybe_closer(rng: random.Random, chemical: bool = False) -> str:
    """A closing line, most of the time absent. Mandatory closers were a tic."""
    if chemical:
        return pick(rng, CHEMICAL_CAUTION)
    if rng.random() < CLOSER_PROB:
        return pick(rng, CLOSERS)
    return ""


def compose_steps(rng: random.Random, steps: list[str], intro: str = "",
                  chemical: bool = False) -> str:
    """Procedural answer. Opens on the intro if given, else straight into steps."""
    body = bullets([s for s in steps if clean(s)], numbered=True)
    head = para(intro) if clean(intro) else ""
    tail = maybe_closer(rng, chemical)
    parts = [p for p in (head, body, para(tail) if tail else "") if p]
    return "\n\n".join(parts)


def compose_signs(rng: random.Random, opening: str, signs: list[str],
                  followup: str = "", chemical: bool = False) -> str:
    """Diagnostic answer: what it is, how to confirm, what follows."""
    confirm = bullets([s for s in signs if clean(s)])
    tail = para(followup, maybe_closer(rng, chemical))
    parts = [para(opening), "What to look for:\n" + confirm]
    if tail:
        parts.append(tail)
    return "\n\n".join(parts)


def compose_prose(rng: random.Random, *body: str, chemical: bool = False) -> str:
    """Explanatory answer. Opens on its own first fact, no shared lead."""
    tail = maybe_closer(rng, chemical)
    return para(*body, tail)


def word_count(text: str) -> int:
    return len(text.split())
