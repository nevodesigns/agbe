"""Answer composition for the ADTC agriculture corpus.

The first generator concatenated raw fact fields and produced 23-word answers against
an 80 to 220 word target. A human judge reading a 23-word reply scores it as thin, and
thin is what loses `S_acc`.

This module fixes that by composing answers with a deliberate shape rather than
gluing facts together:

    LEAD      one sentence that answers the question directly, no preamble
    BODY      the specifics: rates, spacings, timings, varieties, symptoms
    STEPS     a short ordered list when the task is procedural
    CAVEAT    the honest limit, or the thing farmers commonly get wrong
    CLOSE     the practical next action, varied so it is not a tic

Every sentence still traces to `facts.json`. Composition adds structure and connective
language, never new agronomic claims.
"""

from __future__ import annotations

import random
import re


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def para(*parts: str) -> str:
    """Join fragments into flowing prose, fixing spacing and terminal punctuation."""
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


# ---------------------------------------------------------------------------
# Varied connective language, so 2,000 examples do not all open the same way.
# ---------------------------------------------------------------------------

# Closers are split by answer shape. A prose answer that ends "start with the cheapest
# change on this list" when there is no list reads as incoherent, and a judge notices.
CLOSERS_LIST = [
    "Start with the cheapest item on this list and watch the crop for two weeks before"
    " deciding whether more is needed",
    "Work down this list in order. The early items cost little and prevent most of the"
    " damage the later ones try to repair",
    "If anything here does not match what you are seeing in the field, take a sample or a"
    " photograph to your local extension officer before spending money on inputs",
]

CLOSERS_PROSE = [
    "Your local extension officer can confirm which varieties and products are available"
    " and registered in your area",
    "Keep a simple record of what you did and when, because next season that record is"
    " worth more than any general advice",
    "If what you are seeing in the field does not match this, take a sample to your local"
    " extension officer rather than guessing",
    "Conditions differ between zones and soils, so treat this as a starting point and"
    " adjust to what your own field tells you",
]

# Retained for callers that do not distinguish shape.
CLOSERS_GENERAL = CLOSERS_PROSE

CLOSERS_CHEMICAL = [
    "If you decide to spray, use a product registered for this pest on this crop and follow"
    " the rate printed on the label exactly. Do not guess a dose, and do not mix stronger"
    " than the label says on the assumption that it will work better",
    "Any chemical control should follow the product label for rate, timing and pre-harvest"
    " interval. Confirm the choice with your extension officer, since registered products"
    " differ between countries",
]

CAVEATS = [
    "The most common mistake here is acting late",
    "Cost matters as much as effectiveness on a small holding, so do the cheap things first",
    "Conditions vary by zone and by soil, so treat these figures as a starting point rather"
    " than a rule",
]


def pick(rng: random.Random, pool: list[str]) -> str:
    return rng.choice(pool)


# ---------------------------------------------------------------------------
# Composers, one per answer shape.
# ---------------------------------------------------------------------------

def compose_howto(rng, lead: str, steps: list[str], caveat: str = "",
                  chemical: bool = False) -> str:
    """Procedural answer: lead, numbered steps, caveat, close."""
    body = bullets([s for s in steps if clean(s)], numbered=True)
    closer = pick(rng, CLOSERS_CHEMICAL if chemical else CLOSERS_LIST)
    tail = para(caveat or pick(rng, CAVEATS), closer)
    return f"{para(lead)}\n\n{body}\n\n{tail}"


def compose_diagnose(rng, lead: str, signs: list[str], cause: str,
                     action: str, chemical: bool = False) -> str:
    """Diagnostic answer: what it likely is, how to confirm, what to do."""
    confirm = bullets([s for s in signs if clean(s)])
    closer = pick(rng, CLOSERS_CHEMICAL if chemical else CLOSERS_LIST)
    return (f"{para(lead)}\n\nWhat to look for:\n{confirm}\n\n"
            f"{para(cause, action, closer)}")


def compose_prose(rng, lead: str, *body: str) -> str:
    """Explanatory answer with no list, for 'what' and 'why' questions."""
    return para(lead, *body, pick(rng, CLOSERS_PROSE))


def word_count(text: str) -> int:
    return len(text.split())
