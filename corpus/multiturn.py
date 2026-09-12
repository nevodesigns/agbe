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


# Turn-two questions that CHANGE the subject. Round 1's judge transcript is the
# reason this exists: asked "What about soil conditions" after a chilli question,
# the model produced a generic drainage-and-NPK paragraph, and then produced the
# same paragraph again for the next question, which was an unrelated fall
# armyworm prompt. Two different failures, one cause.
#
#   - every follow-up in FOLLOWUPS keeps the subject fixed, so the corpus never
#     taught the model to move to a new topic mid-conversation
#   - and 93% of the shipped corpus was single-turn, so a second turn was close
#     to unseen territory
#
# The contract these examples teach is narrow and it is the whole point: answer
# the question in front of you, from the facts for THAT subject, not from what
# you just said.
SHIFTS = {
    "soil": ["What about soil conditions?", "And the soil side of it?",
             "What should I be doing about the soil?"],
    "storage": ["What about storing it afterwards?", "And after harvest?",
                "What about storage?"],
    "market": ["What about selling it?", "And getting it to market?",
               "What about the market side?"],
    "weather": ["What about the rains?", "And if the weather turns?",
                "What about weather this season?"],
}
SHIFT_BLOCK = {"soil": "soil_water", "storage": "postharvest",
               "market": "market", "weather": "weather"}


def build_topic_shift(rng: random.Random, facts: dict, rec_fn,
                      max_convos: int = 96) -> list[dict]:
    """Two-turn conversations whose second turn moves to a different subject.

    The second answer is composed from a DIFFERENT fact block than the first, so
    the only way to produce it is to read the new question rather than continue
    the previous answer.
    """
    out: list[dict] = []
    subjects: list[tuple[str, dict, str]] = []
    for crop, f in facts["crops"].items():
        subjects.append((crop, f, crop.replace("_", " ")))
    for pest, f in facts["pests_diseases"].items():
        subjects.append((pest, f, pest.replace("_", " ")))

    shift_keys = list(SHIFTS)
    shift_cursor = {k: 0 for k in shift_keys}
    rng.shuffle(subjects)
    for i, (key, f, name) in enumerate(subjects):
        if len(out) >= max_convos:
            break
        for j in range(3):
            shift = shift_keys[(i + j) % len(shift_keys)]
            if len(out) >= max_convos:
                break
            block = facts.get(SHIFT_BLOCK[shift]) or {}
            if not block:
                continue

            host = f.get("crop") or f.get("species") or f.get("target", "")
            q1 = rng.choice([
                f"I have a problem with {name}. What should I do?",
                f"What is the right way to handle {name}?",
                f"Tell me how to deal with {name}.",
            ]) if not host else rng.choice([
                f"I have {name} on my {host}. What should I do?",
                f"How do I deal with {name} on my {host}?",
            ])
            a1 = para(*body_from(f, limit=3))
            if word_count(a1) < MIN_ANSWER_WORDS:
                continue

            # The second answer comes from a different block entirely, walked
            # round-robin rather than sampled. Sampling concentrated these turns
            # on a few entries, and because they reuse the same sentences that
            # build_block already emits, they pushed those blocks over the
            # sentence cap: soil fell from 39 examples to 28 and postharvest from
            # 41 to 29 on the first build with topic shifts in it. Spreading the
            # draw keeps the cost off any single entry.
            entries = sorted(block)
            topic2 = entries[shift_cursor[shift] % len(entries)]
            shift_cursor[shift] += 1
            fact2 = block[topic2]
            a2 = para(fact2) if isinstance(fact2, str) else para(
                *[v for v in fact2.values() if isinstance(v, str)][:2])
            if word_count(a2) < MIN_FOLLOWUP_WORDS:
                continue

            out.append(rec_fn(q1, a1, "multiturn", f"{key}__{shift}",
                              f"shift_{shift}",
                              extra_turns=[(rng.choice(SHIFTS[shift]), a2)]))
    return out
