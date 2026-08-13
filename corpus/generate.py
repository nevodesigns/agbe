"""Compose the ADTC agriculture instruction corpus from the curated fact base.

Provenance: we do not scrape the web and we do not distil from a larger model.
Every pair is composed from `facts.json`, which holds only established
extension-service practice. If a fact is not in that file it cannot appear here.
This domain is graded by agronomists who notice invented chemistry.

**Rewritten after v1's trained model failed its smoke test.** v1 gave each slice a
constant lead sentence and a mandatory closer, so 47% of assistant turns opened
with one of six sentences (one appeared 81 times). The model learned that
scaffolding instead of the agronomy, and answered a maize pest question with the
soil-and-storage opener before inventing a fungal infection.

What changed:
  - no constant lead sentences; answers open on their own first fact
  - closers appear on roughly a quarter of answers, from a wider pool
  - the system prompt is on a MINORITY of examples, so the tuning is
    unconditional (v1 only behaved correctly when the prompt was supplied)
  - a diversity gate: `main()` reports the share of answers sharing an opening,
    and that is the number to watch, not word count

Anti-overfitting: a held-out slice is never trained on, standing in for the
judges' hidden prompts. Our two submitted test prompts are excluded from training.
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compose import (  # noqa: E402
    compose_prose, compose_signs, compose_steps, para, word_count,
)
from multiturn import build_multiturn  # noqa: E402

FACTS = json.loads((HERE / "facts.json").read_text())

SEED = 20260813
SYSTEM = (
    "You are an agricultural advisor for smallholder farmers and extension officers "
    "in West and Central Africa. Answer with practical, specific guidance the farmer "
    "can act on. Never invent agrochemical doses, prices or yield figures."
)

# Fraction of examples carrying the system prompt. v1 used 100% and the model
# only behaved correctly when it was supplied; judges will not supply it.
SYSTEM_PROB = 0.25

# How many times the hand-written gold set is repeated in training.
GOLD_REPEAT = 4

CROP_DISPLAY = {
    "cassava": ("cassava", False), "maize": ("maize", False),
    "yam": ("yam", False), "rice": ("rice", False),
    "sorghum_millet": ("sorghum and millet", True),
    "cowpea": ("cowpea", False), "groundnut": ("groundnut", False),
    "tomato": ("tomato", False),
    "plantain_banana": ("plantain and banana", True),
    "vegetables": ("leafy vegetables", True), "soybean": ("soybean", False),
    "cocoa": ("cocoa", False), "oil_palm": ("oil palm", False),
    "sweet_potato": ("sweet potato", False), "pepper": ("pepper", False),
    "onion": ("onion", False),
}

ASK = {
    "howto_plant": ["How do I plant {c}?",
                    "What is the right way to establish {a} {c} field?",
                    "Walk me through planting {c} on a small plot.",
                    "I want to start growing {c}. Where do I begin?",
                    "What do I need to get right when planting {c}?"],
    "when": ["When should I plant {c}?", "What is the right timing for {c}?",
             "I am not sure when to put my {c} in the ground. Any guidance?",
             "Is there a best time of year for {c}?"],
    "improve": ["My {c} yield is poor. How do I improve it?",
                "How can I get more out of my {c} this season?",
                "What am I most likely doing wrong with {c}?",
                "My {c} is not yielding like my neighbour's. What should I change?"],
    "what": ["Tell me about growing {c}.", "Can you explain {c} farming for a beginner?",
             "How is {c} grown?", "What should a new farmer know about {c}?"],
    "diagnose_crop": ["Something is wrong with my {c} and I cannot tell what. What should I check?",
                      "My {c} is not doing well. How do I work out the problem?",
                      "What usually goes wrong with {c}?"],
}


def rec(question: str, answer: str, slice_name: str, topic: str, form: str,
        rng: random.Random, extra_turns: list[tuple[str, str]] | None = None) -> dict:
    """One training record. The system prompt is applied only sometimes."""
    messages: list[dict] = []
    if rng.random() < SYSTEM_PROB:
        messages.append({"role": "system", "content": SYSTEM})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    words = word_count(answer)
    for uq, ua in (extra_turns or []):
        messages.append({"role": "user", "content": uq})
        messages.append({"role": "assistant", "content": ua})
        words += word_count(ua)
    n_user = sum(1 for m in messages if m["role"] == "user")
    return {"messages": messages,
            "_meta": {"slice": slice_name, "topic": topic, "form": form,
                      "words": words, "turns": n_user}}


def build_crops(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for crop, f in FACTS["crops"].items():
        name, is_plural = CROP_DISPLAY.get(crop, (crop.replace("_", " "), False))
        article = "" if is_plural else "a"
        is_are, staple = ("are", "staples") if is_plural else ("is", "a staple")
        loc = ", ".join(f.get("local_names", []))

        def q(t: str) -> str:
            return t.format(c=name, a=article).replace("  ", " ").strip()

        steps = [s for s in (f.get("planting_material"), f.get("land_prep"),
                             f.get("spacing"), f.get("planting_position"),
                             f.get("fertiliser"), f.get("staking"),
                             f.get("transplanting"), f.get("nursery"),
                             f.get("inoculant"), f.get("material")) if s]
        if len(steps) >= 2:
            for tpl in ASK["howto_plant"]:
                out.append(rec(q(tpl),
                               compose_steps(rng, steps,
                                             intro=f.get("critical_window", "")),
                               "crop_agronomy", crop, "howto", rng))

        if f.get("planting_time") or f.get("harvest"):
            for tpl in ASK["when"][:3]:
                out.append(rec(q(tpl),
                               compose_prose(rng, f.get("planting_time", ""),
                                             FACTS["zones"]["planting_rule"],
                                             f.get("harvest", ""), f.get("storage", "")),
                               "crop_agronomy", crop, "when", rng))

        levers = [s for s in (f.get("critical_window"), f.get("spacing"),
                              f.get("fertiliser"), f.get("varieties"),
                              f.get("seed_choice"), f.get("management"),
                              f.get("training"), f.get("pruning"), f.get("shade"),
                              f.get("rotation_value"), f.get("rotation"),
                              f.get("calcium"), f.get("water")) if s]
        if len(levers) >= 2:
            for tpl in ASK["improve"]:
                out.append(rec(q(tpl), compose_steps(rng, levers),
                               "crop_agronomy", crop, "improve", rng))

        overview = para(
            f"{name.capitalize()} {is_are} {staple} of West and Central African "
            f"smallholder systems" + (f", known locally as {loc}" if loc else ""),
            f.get("spacing", ""), f.get("harvest", ""), f.get("systems", ""),
            f.get("cash_flow", ""), f.get("processing_note", ""), f.get("quality", ""))
        if overview:
            for tpl in ASK["what"][:3]:
                out.append(rec(q(tpl), compose_prose(rng, overview),
                               "crop_agronomy", crop, "what", rng))

        probs = f.get("common_problems", [])
        if probs:
            out.append(rec(
                q(rng.choice(ASK["diagnose_crop"])),
                compose_signs(rng,
                              f"On {name} a short list of causes covers most of what "
                              f"goes wrong",
                              [f"Check for {p}" for p in probs],
                              followup="Match the symptom to the cause before spending "
                                       "money. The wrong treatment costs cash and loses "
                                       "the window for the right one."),
                "crop_agronomy", crop, "diagnose", rng))
    return out


def build_pests(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for key, f in FACTS["pests_diseases"].items():
        name = key.replace("_", " ")
        host = f.get("crop") or f.get("species") or f.get("target", "the crop")
        chemical = bool(f.get("chemical_note"))

        if f.get("identify"):
            signs = [f["identify"]]
            if f.get("scouting"):
                signs.append(f["scouting"])
            cause = f.get("spread", "") or f.get("driver", "") or f.get("cause_signal", "")
            action = f.get("control", "") or f.get("non_chemical", "")
            for tpl in (f"I think I have {name} on my {host}. How do I confirm it?",
                        f"My {host} is showing damage and I suspect {name}. How do I be sure?",
                        f"How do I tell {name} apart from other problems on {host}?"):
                out.append(rec(tpl, compose_signs(
                    rng,
                    f"{name.capitalize()} on {host} can be identified in the field "
                    f"before you treat",
                    signs, followup=para(cause, action), chemical=chemical),
                    "pests_diseases", key, "diagnose", rng))

        steps = [s for s in (f.get("non_chemical"), f.get("control"), f.get("window"),
                             f.get("biosecurity"), f.get("seed_bank"),
                             f.get("chemical_note")) if s]
        if steps:
            for tpl in (f"How do I control {name}?",
                        f"What is the right way to deal with {name} on {host}?",
                        f"What actually works against {name}?"):
                out.append(rec(tpl, compose_steps(rng, steps,
                                                  intro=f.get("cause_signal", ""),
                                                  chemical=chemical),
                               "pests_diseases", key, "howto", rng))
            out.append(rec(f"How do I stop {name} coming back next season?",
                           compose_steps(rng, steps, chemical=chemical),
                           "pests_diseases", key, "prevent", rng))

        if f.get("misconception"):
            out.append(rec(f"Can I just spray something to cure {name}?",
                           compose_prose(rng, f["misconception"], f.get("control", "")),
                           "pests_diseases", key, "what", rng))
    return out


def build_block(rng: random.Random, fact_key: str, slice_name: str,
                actions: dict[str, str]) -> list[dict]:
    """Generic builder for the flat fact blocks.

    There is deliberately NO shared lead sentence. v1 opened all 50 livestock
    answers and all 81 soil/post-harvest answers with the same phrase, which is
    exactly what the model memorised instead of the agronomy.
    """
    out: list[dict] = []
    block = FACTS[fact_key]
    for key, fact in block.items():
        action = actions.get(key, key.replace("_", " "))
        related = [v for k, v in block.items() if k != key]
        rng.shuffle(related)
        for tpl in (f"How do I {action}?", f"What is the right way to {action}?",
                    f"I want to {action}. What should I know?"):
            out.append(rec(tpl, compose_prose(rng, fact, *related[:2]),
                           slice_name, key, "howto", rng))
        out.append(rec(f"Why does it matter how I {action}?",
                       compose_prose(rng, fact, *related[2:3]),
                       slice_name, key, "what", rng))
        out.append(rec(f"What goes wrong when farmers do not {action}?",
                       compose_prose(rng, fact, *related[3:5]),
                       slice_name, key, "prevent", rng))
    return out


LIVESTOCK_ACTIONS = {
    "brooding": "brood day old chicks", "poultry_density": "stock my broiler house",
    "layer_calcium": "keep egg shell quality good",
    "small_ruminant_housing": "house goats", "deworming": "deworm my small ruminants",
    "catfish_density": "stock a catfish pond",
    "catfish_sorting": "manage catfish as they grow",
    "broiler_feeding": "feed broilers to market weight",
    "layer_lighting": "manage lighting for laying birds",
    "local_chicken": "keep backyard chickens profitably",
    "cattle_dry_season": "feed cattle through the dry season",
    "rabbits": "start keeping rabbits", "beekeeping": "start beekeeping",
    "tilapia": "raise tilapia in a pond",
}
SOIL_ACTIONS = {
    "soil_testing": "decide which fertiliser to buy",
    "nitrogen_placement": "apply fertiliser so it is not wasted",
    "poultry_manure": "use poultry manure safely",
    "acidity": "deal with soil that stays poor despite fertiliser",
    "erosion": "protect a sloping field from erosion", "rotation": "plan a rotation",
    "irrigation": "decide about dry season production", "mulching": "use mulch on my plots",
    "cover_crops": "use cover crops between seasons", "compost": "make and use compost",
    "agroforestry": "combine trees with my crops",
}
POST_ACTIONS = {
    "grain_moisture": "know when grain is dry enough to store",
    "drying_surface": "dry my harvest properly",
    "aflatoxin": "avoid aflatoxin in groundnut and maize",
    "hermetic": "store grain without chemicals",
    "cassava_processing": "handle cassava after harvest",
    "tomato_handling": "get tomatoes to market without losses",
    "market_timing": "avoid selling into the harvest glut",
    "solar_drying": "dry produce faster and cleaner",
    "packaging": "package produce for market",
    "cooperatives": "sell through a cooperative", "record_keeping": "keep farm records",
}


def build_zones(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for zone, desc in FACTS["zones"].items():
        if zone == "planting_rule":
            continue
        z = zone.replace("_", " ")
        for tpl in (f"I farm in the {z} zone. What should I be growing?",
                    f"Which crops suit the {z} zone?",
                    f"What grows well in {z} conditions?"):
            out.append(rec(tpl, compose_prose(rng, desc, FACTS["zones"]["planting_rule"]),
                           "crop_agronomy", f"zone_{zone}", "compare", rng))
    return out


def load_gold() -> list[dict]:
    p = HERE / "gold.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in rows:
        meta = r.setdefault("_meta", {})
        meta.setdefault("slice", "gold")
        meta.setdefault("form", "gold")
        meta.setdefault("topic", "gold")
        meta["words"] = sum(word_count(m["content"])
                            for m in r["messages"] if m["role"] == "assistant")
        meta["turns"] = sum(1 for m in r["messages"] if m["role"] == "user")
    return rows


def main() -> None:
    rng = random.Random(SEED)
    pairs: list[dict] = []
    pairs += build_crops(rng)
    pairs += build_pests(rng)
    pairs += build_block(rng, "livestock", "livestock_poultry_fish", LIVESTOCK_ACTIONS)
    pairs += build_block(rng, "soil_water", "soil_fertiliser_water", SOIL_ACTIONS)
    pairs += build_block(rng, "postharvest", "postharvest_storage_market", POST_ACTIONS)
    pairs += build_zones(rng)
    pairs += build_multiturn(
        rng, FACTS,
        lambda q, a, s, t, f, extra_turns=None: rec(q, a, s, t, f, rng,
                                                    extra_turns=extra_turns))

    # Gold is hand-written and carries the behaviours templating cannot produce:
    # refusals, honest limits, Pidgin, real diagnostic reasoning. v1 had ONE
    # refusal example against 374 in-domain ones and the domain prior swallowed
    # it, so the model told a parent to take their feverish child to an extension
    # officer. Oversampling weights those behaviours without inventing more data.
    gold = load_gold()
    pairs += gold * GOLD_REPEAT

    seen, deduped = set(), []
    for p in pairs:
        key = " || ".join(m["content"].strip().lower()
                          for m in p["messages"] if m["role"] == "user")
        # Gold is deliberately repeated, so it bypasses the dedup that exists to
        # catch accidental collisions in the templated builders.
        if p["_meta"]["slice"] != "gold":
            if key in seen:
                continue
            seen.add(key)
        deduped.append(p)

    rng.shuffle(deduped)
    n_hold = max(24, int(len(deduped) * 0.08))
    holdout, train = deduped[:n_hold], deduped[n_hold:]

    out_dir = HERE / "build"
    out_dir.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("holdout", holdout)):
        with (out_dir / f"{name}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(train)
    by_slice = collections.Counter(r["_meta"]["slice"] for r in train)
    print(f"train={total}  holdout={len(holdout)}  gold={len(gold)}\n")
    print("by slice:")
    for k, v in by_slice.most_common():
        print(f"   {k:<30}{v:>5}  {v/total*100:5.1f}%")

    # ---- the gate that matters -------------------------------------------
    # v1's model learned opening sentences rather than agronomy. Watch this,
    # not word count.
    answers = [m["content"] for r in train for m in r["messages"]
               if m["role"] == "assistant"]
    opens = collections.Counter(a.split(".")[0][:60] for a in answers)
    top6 = sum(v for _, v in opens.most_common(6))
    print(f"\nopening-sentence diversity ({len(answers)} answers):")
    for k, v in opens.most_common(4):
        print(f"   {v:>4}x  {k}")
    print(f"   top-6 share: {top6}/{len(answers)} ({top6/len(answers)*100:.0f}%)"
          f"   [v1 was 47%, and the model memorised it]")

    with_sys = sum(1 for r in train if r["messages"][0]["role"] == "system")
    print(f"\nsystem prompt on: {with_sys}/{total} ({with_sys/total*100:.0f}%)"
          f"   [v1 was 100%, so tuning only fired when supplied]")

    lens = sorted(word_count(a) for a in answers)
    print(f"answer words: p10={lens[len(lens)//10]} p50={lens[len(lens)//2]} "
          f"p90={lens[int(len(lens)*0.9)]}")
    multi = sum(1 for r in train if r["_meta"].get("turns", 1) > 1)
    print(f"multi-turn: {multi}/{total} ({multi/total*100:.0f}%)")
    refusal_forms = {"refusal", "safety", "honest_limit", "boundary", "clarify"}
    refusals = sum(1 for r in train if r["_meta"]["form"] in refusal_forms)
    pidgin = sum(1 for r in train if r["_meta"]["form"] == "pidgin")
    print(f"refusal/limit examples: {refusals}/{total} ({refusals/total*100:.1f}%)"
          f"   [v1 had 1, and the model advised on a child's fever]")
    print(f"pidgin examples: {pidgin}/{total} ({pidgin/total*100:.1f}%)"
          f"   [v1 had 2, and answered Pidgin in English]")
    print(f"\nwrote {out_dir}/train.jsonl and holdout.jsonl")


if __name__ == "__main__":
    main()
