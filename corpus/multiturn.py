"""Multi-turn conversations for the ADTC corpus.

The judging FAQ says a judge "chats with it live through our in-browser interface".
A corpus of isolated question-answer pairs teaches a 1B model to treat every turn as
a fresh question, and the failure modes are obvious to anyone holding a conversation:

  - it forgets which crop is under discussion and answers about a different one
  - it repeats its closing line verbatim every turn, which reads as robotic
  - it re-explains from scratch instead of building on what it just said

These conversations train the follow-up behaviour a judge will actually exercise.
Follow-ups are the ones real farmers ask: cost, substitution, timing, and "I already
tried that".

As everywhere in this corpus, no new agronomic claim is introduced. The follow-up
answers recombine facts already stated in `facts.json`.
"""

from __future__ import annotations

import random

from compose import clean, para, word_count  # noqa: F401

# Fields that carry substantive guidance, in the order we prefer to use them.
# Pulling a fixed hardcoded subset produced bare one-line answers on any topic
# missing those exact keys (coccidiosis has `driver` not `control`; cocoa has
# `shade` and `pruning` not `critical_window`), which is how 152 content-free
# training examples got generated.
BODY_FIELDS = (
    "control", "non_chemical", "critical_window", "spacing", "fertiliser",
    "varieties", "seed_choice", "management", "training", "pruning", "shade",
    "window", "driver", "identify", "spread", "planting_material", "land_prep",
    "harvest", "quality", "curing", "storage", "rotation_value", "inoculant",
    "nursery", "material", "biosecurity", "cause_signal", "photoperiod",
    "zone_fit", "systems", "water", "calcium", "drying", "season_risk",
)

MIN_ANSWER_WORDS = 45
# A follow-up turn is legitimately shorter than an opening answer: in real
# conversation the second reply builds on the first rather than restating it.
MIN_FOLLOWUP_WORDS = 30


def body_from(f: dict, limit: int = 4, skip: int = 0) -> list[str]:
    """Substantive fact fragments for a topic, whatever keys it happens to have."""
    vals = [f[k] for k in BODY_FIELDS if isinstance(f.get(k), str) and f[k].strip()]
    return vals[skip:skip + limit]

# Follow-up intents, chosen because they are what a judge probing a farm advisor
# would naturally ask second.
FOLLOWUPS = {
    "cost": [
        "That sounds expensive. What if I cannot afford it?",
        "I have very little cash this season. What is the cheapest part of that?",
        "Which of those matters most if I can only do one?",
    ],
    "timing": [
        "How soon will I see a difference?",
        "Is it too late in the season to do that now?",
        "When exactly should I do this?",
    ],
    "already_tried": [
        "I already did that and it did not help. What else?",
        "I tried that last season and the problem came back. Why?",
    ],
    "clarify": [
        "Can you explain that more simply?",
        "Sorry, what does that mean exactly?",
    ],
}

# Several phrasings per intent. v1 used ONE fixed sentence for each, so four
# sentences opened roughly 80 follow-up answers between them and became the thing
# the model learned instead of the agronomy.
PRINCIPLE = {
    "cost": [
        "Labour costs time rather than money, and on a small plot it changes the "
        "outcome more than most purchased inputs",
        "The unpaid work usually beats the bought input: clean planting material, "
        "correct spacing, and weeding on time",
        "Spend nothing first. Sanitation, timing and spacing are free and carry most "
        "of the gain",
    ],
    "timing": [
        "Give it about two weeks and compare a treated strip against an untreated one, "
        "or you cannot separate the weather from the treatment",
        "Field changes are rarely visible in days. Leave an untreated strip so you have "
        "something to compare against",
        "Judge it against an untreated patch after a fortnight rather than by eye the "
        "next morning",
    ],
    "already_tried": [
        "When a problem returns each season the source was usually never removed, "
        "rather than the treatment having failed",
        "Recurrence points at a carrier: infected planting material, crop residue left "
        "standing, or a neighbouring plot",
        "Something carries it between seasons. Find that and the treatment starts "
        "working",
    ],
    "clarify": [
        "Put simply: fix the cause, not the symptom, and do the cheap thing first",
        "The short version is to treat what is causing it rather than what you can see",
        "In plain terms, find the source, deal with that, and start with what costs "
        "nothing",
    ],
}

BODY_FIELDS = (
    "control", "non_chemical", "critical_window", "spacing", "fertiliser",
    "varieties", "seed_choice", "management", "training", "pruning", "shade",
    "window", "driver", "identify", "spread", "planting_material", "land_prep",
    "harvest", "quality", "curing", "storage", "rotation_value", "inoculant",
    "nursery", "material", "biosecurity", "cause_signal", "photoperiod",
    "zone_fit", "systems", "water", "calcium", "drying", "season_risk",
)

MIN_ANSWER_WORDS = 45
# A follow-up turn is legitimately shorter than an opening answer: in real
# conversation the second reply builds on the first rather than restating it.
MIN_FOLLOWUP_WORDS = 30


def body_from(f: dict, limit: int = 4, skip: int = 0) -> list[str]:
    """Substantive fact fragments for a topic, whatever keys it happens to have."""
    vals = [f[k] for k in BODY_FIELDS if isinstance(f.get(k), str) and f[k].strip()]
    return vals[skip:skip + limit]

# Follow-up intents, chosen because they are what a judge probing a farm advisor
# would naturally ask second.
FOLLOWUPS = {
    "cost": [
        "That sounds expensive. What if I cannot afford it?",
        "I have very little cash this season. What is the cheapest part of that?",
        "Which of those matters most if I can only do one?",
    ],
    "timing": [
        "How soon will I see a difference?",
        "Is it too late in the season to do that now?",
        "When exactly should I do this?",
    ],
    "already_tried": [
        "I already did that and it did not help. What else?",
        "I tried that last season and the problem came back. Why?",
    ],
    "clarify": [
        "Can you explain that more simply?",
        "Sorry, what does that mean exactly?",
    ],
}

CHEAPEST_FIRST = (
    "If cash is the constraint, do the things that cost labour rather than money first."
    " Clean planting material, correct spacing, timely weeding and good sanitation cost"
    " you time and change the outcome more than most purchased inputs do"
)

TIMING_GENERIC = (
    "Field changes rarely show overnight. Give the crop about two weeks and compare a"
    " treated area against an untreated strip, otherwise you cannot tell whether the"
    " change came from what you did or from the weather"
)

RECURRENCE = (
    "A problem that returns each season usually means the source was never removed"
    " rather than that the treatment failed. Look at what carries it over: infected"
    " planting material, crop residues left in the field, a neighbouring plot, or soil"
    " fertility that never recovered"
)

SIMPLER = (
    "Put simply: fix the cause, not just the symptom, and do the cheap thing before the"
    " expensive one"
)


def build_multiturn(rng: random.Random, facts: dict, rec_fn, max_convos: int = 110) -> list[dict]:
    """Two-turn conversations grounded in the crop and pest fact base."""
    out: list[dict] = []

    subjects: list[tuple[str, str, dict, str]] = []
    for crop, f in facts["crops"].items():
        subjects.append(("crop", crop, f, crop.replace("_", " ")))
    for pest, f in facts["pests_diseases"].items():
        subjects.append(("pest", pest, f, pest.replace("_", " ")))

    for kind, key, f, name in subjects:
        for intent in ("cost", "timing", "already_tried", "clarify"):
            if len(out) >= max_convos:
                return out

            # Opening turn and its answer, drawn from the same facts as elsewhere.
            if kind == "crop":
                q1 = rng.choice([
                    f"How do I get a better crop of {name}?",
                    f"What should I focus on to improve my {name}?",
                    f"My {name} is underperforming. Where do I start?",
                ])
                a1 = para(*body_from(f, limit=4))
            else:
                host = f.get("crop") or f.get("species") or f.get("target", "the crop")
                q1 = rng.choice([
                    f"How do I deal with {name} on my {host}?",
                    f"I have {name} in my {host}. What should I do?",
                    f"What is the best approach to {name}?",
                ])
                a1 = para(*body_from(f, limit=4))

            # A lead sentence with no facts behind it is worse than no example.
            if word_count(a1) < MIN_ANSWER_WORDS:
                continue

            q2 = rng.choice(FOLLOWUPS[intent])
            # Each follow-up pairs its generic principle with facts specific to this
            # topic, drawn from whatever fields the topic actually has.
            # Topic facts FIRST, general principle second. v1 led with the
            # generic sentence, which is the position a model learns hardest.
            extra = body_from(f, limit=2, skip=4) or body_from(f, limit=2)
            a2 = para(*extra, rng.choice(PRINCIPLE[intent]))

            if word_count(a2) < MIN_FOLLOWUP_WORDS:
                continue

            out.append(rec_fn(q1, a1, "multiturn", key, f"multiturn_{intent}",
                              extra_turns=[(q2, a2)]))
    return out
