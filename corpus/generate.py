"""Compose the ADTC agriculture instruction corpus from the curated fact base.

Methodology (this is also the REPORT.md section on data provenance):

We do not scrape the web and we do not distil from a larger model. Both routes put
claims into the training data that nobody can trace, and this domain is judged by
agronomists who will notice invented chemistry. Every pair is composed from
`facts.json`, which holds only established extension-service practice. If a fact is
not in that file it cannot appear in the corpus.

The cost of that choice is phrasing variety, which templating alone does not give.
We buy it back four ways:
  1. several question surface forms per topic, since farmers do not ask in one voice
  2. question FORMS (what / diagnose / howto / when / prevent / compare / improve)
     that force genuinely different answer shapes from the same facts
  3. structured composition via `compose.py` rather than fact concatenation, so
     answers land in the 80 to 220 word band a human judge scores well
  4. hand-written gold exemplars in `gold.jsonl`, mixed in, carrying the house style

Anti-overfitting: a held-out slice is never trained on, standing in for the judges'
3 hidden prompts. Our own 2 submitted test prompts are excluded from training.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compose import (  # noqa: E402
    CLOSERS_GENERAL, compose_diagnose, compose_howto, compose_prose, para,
    pick, word_count,
)
from multiturn import build_multiturn  # noqa: E402

FACTS = json.loads((HERE / "facts.json").read_text())

SEED = 20260812
SYSTEM = (
    "You are an agricultural advisor for smallholder farmers and extension officers "
    "in West and Central Africa. Answer with practical, specific guidance the farmer "
    "can act on. Never invent agrochemical doses, prices or yield figures."
)

# Display name and grammatical number per crop key. Without this, category entries
# generated broken questions like "What is vegetables and how is it grown?" and leads
# like "Vegetables is a staple", which read as incompetent to an agronomist judge.
CROP_DISPLAY = {
    "cassava": ("cassava", False),
    "maize": ("maize", False),
    "yam": ("yam", False),
    "rice": ("rice", False),
    "sorghum_millet": ("sorghum and millet", True),
    "cowpea": ("cowpea", False),
    "groundnut": ("groundnut", False),
    "tomato": ("tomato", False),
    "plantain_banana": ("plantain and banana", True),
    "vegetables": ("leafy vegetables", True),
    "soybean": ("soybean", False),
    "cocoa": ("cocoa", False),
    "oil_palm": ("oil palm", False),
    "sweet_potato": ("sweet potato", False),
    "pepper": ("pepper", False),
    "onion": ("onion", False),
}

ASK = {
    "howto_plant": [
        "How do I plant {c}?",
        "What is the right way to establish {a} {c} field?",
        "Walk me through planting {c} on a small plot.",
        "I want to start growing {c}. Where do I begin?",
    ],
    "when": [
        "When should I plant {c}?",
        "What is the right timing for {c}?",
        "I am not sure when to put my {c} in the ground. Any guidance?",
    ],
    "improve": [
        "My {c} yield is poor. How do I improve it?",
        "How can I get more out of my {c} this season?",
        "What am I most likely doing wrong with {c}?",
    ],
    # Number-agnostic phrasing, so it reads correctly for both singular crops and
    # category entries.
    "what": [
        "Tell me about growing {c}.",
        "Can you explain {c} farming for a beginner?",
        "How is {c} grown?",
    ],
    "diagnose_crop": [
        "Something is wrong with my {c} and I cannot tell what. What should I check?",
        "My {c} is not doing well. How do I work out the problem?",
    ],
}


def rec(question: str, answer: str, slice_name: str, topic: str, form: str,
        extra_turns: list[tuple[str, str]] | None = None) -> dict:
    """One training record. `extra_turns` appends further user/assistant pairs,
    which is how the multi-turn slice trains follow-up behaviour."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    words = word_count(answer)
    for uq, ua in (extra_turns or []):
        messages.append({"role": "user", "content": uq})
        messages.append({"role": "assistant", "content": ua})
        words += word_count(ua)
    return {
        "messages": messages,
        "_meta": {"slice": slice_name, "topic": topic, "form": form,
                  "words": words, "turns": (len(messages) - 1) // 2},
    }


def build_crops(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for crop, f in FACTS["crops"].items():
        name, is_plural = CROP_DISPLAY.get(crop, (crop.replace("_", " "), False))
        article = "a" if not is_plural else ""
        is_are = "are" if is_plural else "is"
        staple_word = "staples" if is_plural else "a staple"
        loc = ", ".join(f.get("local_names", []))

        def q(template: str) -> str:
            return template.format(c=name, a=article).replace("  ", " ").strip()

        # --- establishment, procedural -------------------------------------
        steps = [s for s in (
            f.get("planting_material"), f.get("land_prep"),
            f.get("spacing"), f.get("planting_position"),
            f.get("fertiliser"), f.get("staking"), f.get("transplanting"),
        ) if s]
        if len(steps) >= 2:
            lead = (f"Getting {name} established well matters more than anything you do "
                    f"later in the season, because a weak stand cannot be rescued by "
                    f"fertiliser")
            for tpl in ASK["howto_plant"][:3]:
                out.append(rec(q(tpl),
                               compose_howto(rng, lead, steps,
                                             caveat=f.get("critical_window", "")),
                               "crop_agronomy", crop, "howto"))

        # --- timing ---------------------------------------------------------
        if f.get("planting_time") or f.get("harvest"):
            lead = f"Timing for {name} is driven by the rains, not by the calendar"
            out_body = para(f.get("planting_time", ""),
                            FACTS["zones"]["planting_rule"],
                            f.get("harvest", ""), f.get("storage", ""))
            for tpl in ASK["when"][:2]:
                out.append(rec(q(tpl),
                               compose_prose(rng, lead, out_body),
                               "crop_agronomy", crop, "when"))

        # --- yield improvement ---------------------------------------------
        levers = [s for s in (
            f.get("critical_window"), f.get("spacing"), f.get("fertiliser"),
            f.get("varieties"), f.get("seed_choice"), f.get("management"),
            f.get("training"), f.get("rotation_value"), f.get("rotation"),
        ) if s]
        if len(levers) >= 2:
            lead = (f"Poor {name} yield almost always traces to one of a few things, and "
                    f"they are worth checking in order of cost before you buy any input")
            for tpl in ASK["improve"]:
                out.append(rec(q(tpl),
                               compose_howto(rng, lead, levers),
                               "crop_agronomy", crop, "improve"))

        # --- what / overview -------------------------------------------------
        lead = (f"{name.capitalize()} {is_are} {staple_word} of West and Central African "
                f"smallholder systems" + (f", known locally as {loc}" if loc else ""))
        body = para(f.get("spacing", ""), f.get("harvest", ""),
                    f.get("systems", ""), f.get("cash_flow", ""),
                    f.get("processing_note", ""))
        if body:
            for tpl in ASK["what"][:2]:
                out.append(rec(q(tpl), compose_prose(rng, lead, body),
                               "crop_agronomy", crop, "what"))

        # --- diagnose from the crop's own problem list -----------------------
        probs = f.get("common_problems", [])
        if probs:
            listed = ", ".join(probs)
            lead = (f"On {name} the usual causes are a short list, so work through them "
                    f"before treating anything")
            signs = [f"Check for {p}" for p in probs]
            cause = (f"The common problems on {name} are {listed}.")
            action = ("Match the symptom to the cause before you spend money. The wrong "
                      "treatment costs cash and loses the window for the right one.")
            out.append(rec(q(rng.choice(ASK["diagnose_crop"])),
                           compose_diagnose(rng, lead, signs, cause, action),
                           "crop_agronomy", crop, "diagnose"))
    return out


def build_pests(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for key, f in FACTS["pests_diseases"].items():
        name = key.replace("_", " ")
        host = f.get("crop") or f.get("species") or f.get("target", "the crop")
        chemical = bool(f.get("chemical_note"))

        # --- identification --------------------------------------------------
        if f.get("identify"):
            signs = [f["identify"]]
            if f.get("scouting"):
                signs.append(f["scouting"])
            lead = f"That sounds like it could be {name} on {host}, but confirm before treating"
            cause = f.get("spread", "") or f.get("driver", "") or f.get("cause_signal", "")
            action = f.get("control", "") or f.get("non_chemical", "")
            for q in (f"I think I have {name} on my {host}. How do I confirm it?",
                      f"My {host} is showing damage and I suspect {name}. How do I be sure?"):
                out.append(rec(q, compose_diagnose(rng, lead, signs, cause, action, chemical),
                               "pests_diseases", key, "diagnose"))

        # --- control, procedural ---------------------------------------------
        steps = [s for s in (
            f.get("non_chemical"), f.get("control"), f.get("window"),
            f.get("biosecurity"), f.get("seed_bank"), f.get("chemical_note"),
        ) if s]
        if steps:
            lead = f"Control of {name} on {host} works best as a sequence, cheapest first"
            for q in (f"How do I control {name}?",
                      f"What is the right way to deal with {name} on {host}?"):
                out.append(rec(q, compose_howto(rng, lead, steps,
                                                caveat=f.get("cause_signal", ""),
                                                chemical=chemical),
                               "pests_diseases", key, "howto"))

            lead_p = f"Preventing {name} is far cheaper than treating it once it is established"
            out.append(rec(f"How do I stop {name} coming back next season?",
                           compose_howto(rng, lead_p, steps, chemical=chemical),
                           "pests_diseases", key, "prevent"))

        # --- misconception ----------------------------------------------------
        if f.get("misconception"):
            out.append(rec(
                f"Can I just spray something to cure {name}?",
                compose_prose(rng,
                              f"No, and this is worth being clear about because it wastes money",
                              f["misconception"], f.get("control", "")),
                "pests_diseases", key, "what"))
    return out


def build_livestock(rng: random.Random) -> list[dict]:
    """Every livestock fact becomes pairs. Phrasing overrides keep the questions
    natural; anything without an override falls back to the key itself."""
    ACTION = {
        "brooding": "brood day old chicks",
        "poultry_density": "stock my broiler house",
        "layer_calcium": "keep egg shell quality good",
        "small_ruminant_housing": "house goats",
        "deworming": "deworm my small ruminants",
        "catfish_density": "stock a catfish pond",
        "catfish_sorting": "manage catfish as they grow",
        "broiler_feeding": "feed broilers through to market weight",
        "layer_lighting": "manage lighting for laying birds",
        "local_chicken": "keep backyard chickens profitably",
        "cattle_dry_season": "feed cattle through the dry season",
        "rabbits": "start keeping rabbits",
        "beekeeping": "start beekeeping",
        "tilapia": "raise tilapia in a pond",
    }
    LEAD = "This is one of the places where a small holding gains or loses money fastest"
    out: list[dict] = []
    block = FACTS["livestock"]
    for key, fact in block.items():
        action = ACTION.get(key, key.replace("_", " "))
        related = [v for k, v in block.items() if k != key]
        rng.shuffle(related)
        for tpl in (f"How do I {action}?", f"What is the right way to {action}?",
                    f"I want to {action}. What should I know?"):
            out.append(rec(tpl, compose_prose(rng, LEAD, fact, *related[:2]),
                           "livestock_poultry_fish", key, "howto"))
        out.append(rec(f"What goes wrong most often when farmers {action}?",
                       compose_prose(rng, LEAD, fact, *related[:1]),
                       "livestock_poultry_fish", key, "diagnose"))
    return out


def build_soil_post(rng: random.Random) -> list[dict]:
    ACTION = {
        "soil_testing": "decide which fertiliser to buy",
        "nitrogen_placement": "apply fertiliser so it is not wasted",
        "poultry_manure": "use poultry manure safely",
        "acidity": "deal with soil that stays poor despite fertiliser",
        "erosion": "protect a sloping field from erosion",
        "rotation": "plan a rotation",
        "irrigation": "decide about dry season production",
        "mulching": "use mulch on my plots",
        "cover_crops": "use cover crops between seasons",
        "compost": "make and use compost",
        "agroforestry": "combine trees with my crops",
        "grain_moisture": "know when grain is dry enough to store",
        "drying_surface": "dry my harvest properly",
        "aflatoxin": "avoid aflatoxin in groundnut and maize",
        "hermetic": "store grain without chemicals",
        "cassava_processing": "handle cassava after harvest",
        "tomato_handling": "get tomatoes to market without losses",
        "market_timing": "avoid selling into the harvest glut",
        "solar_drying": "dry produce faster and cleaner",
        "packaging": "package produce for market",
        "cooperatives": "sell through a cooperative",
        "record_keeping": "keep farm records",
    }
    LEAD = "This is one of the cheapest places on a smallholding to gain or lose money"
    out: list[dict] = []
    for fact_key, slice_name in (("soil_water", "soil_fertiliser_water"),
                                 ("postharvest", "postharvest_storage_market")):
        block = FACTS[fact_key]
        for key, fact in block.items():
            action = ACTION.get(key, key.replace("_", " "))
            related = [v for k, v in block.items() if k != key]
            rng.shuffle(related)
            for tpl in (f"How do I {action}?", f"What is the right way to {action}?"):
                out.append(rec(tpl, compose_prose(rng, LEAD, fact, *related[:2]),
                               slice_name, key, "howto"))
            out.append(rec(f"Why does it matter how I {action}?",
                           compose_prose(rng, LEAD, fact, *related[:1]),
                           slice_name, key, "what"))
            out.append(rec(f"What is the risk if I do not {action}?",
                           compose_prose(rng, LEAD, fact),
                           slice_name, key, "prevent"))
    return out


def build_zones(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for zone, desc in FACTS["zones"].items():
        if zone == "planting_rule":
            continue
        z = zone.replace("_", " ")
        lead = f"What grows well in the {z} zone is decided by the length and reliability of the rains"
        for q in (f"I farm in the {z} zone. What should I be growing?",
                  f"Which crops suit the {z} zone?"):
            out.append(rec(q, compose_prose(rng, lead, desc,
                                            FACTS["zones"]["planting_rule"]),
                           "crop_agronomy", f"zone_{zone}", "compare"))
    return out


def load_gold() -> list[dict]:
    p = HERE / "gold.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in rows:
        r.setdefault("_meta", {}).update(
            {"slice": r["_meta"].get("slice", "gold"), "form": r["_meta"].get("form", "gold"),
             "topic": r["_meta"].get("topic", "gold"),
             "words": word_count(r["messages"][2]["content"])})
    return rows


def main() -> None:
    rng = random.Random(SEED)
    pairs: list[dict] = []
    for builder in (build_crops, build_pests, build_livestock, build_soil_post, build_zones):
        pairs += builder(rng)
    # Judges chat live, so single-turn-only training leaves the model losing the
    # thread on the second question.
    pairs += build_multiturn(rng, FACTS, rec)
    gold = load_gold()
    pairs += gold

    # Key on every user turn, not just the first. Keying on the opener alone
    # collapsed all four multi-turn follow-up intents into one, because they
    # deliberately share an opening question.
    seen, deduped = set(), []
    for p in pairs:
        key = " || ".join(
            m["content"].strip().lower()
            for m in p["messages"] if m["role"] == "user"
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    rng.shuffle(deduped)
    n_hold = max(20, int(len(deduped) * 0.08))
    holdout, train = deduped[:n_hold], deduped[n_hold:]

    out_dir = HERE / "build"
    out_dir.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("holdout", holdout)):
        with (out_dir / f"{name}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(train)
    by_slice: dict[str, int] = {}
    by_form: dict[str, int] = {}
    for r in train:
        by_slice[r["_meta"]["slice"]] = by_slice.get(r["_meta"]["slice"], 0) + 1
        by_form[r["_meta"]["form"]] = by_form.get(r["_meta"]["form"], 0) + 1

    print(f"train={total}  holdout={len(holdout)}  gold={len(gold)}\n")
    print("by slice:")
    for k, v in sorted(by_slice.items(), key=lambda x: -x[1]):
        print(f"   {k:<32}{v:>5}  {v/total*100:5.1f}%")
    print("\nby form:")
    for k, v in sorted(by_form.items(), key=lambda x: -x[1]):
        print(f"   {k:<32}{v:>5}  {v/total*100:5.1f}%")

    per_turn = []
    for r in train:
        for m in r["messages"]:
            if m["role"] == "assistant":
                per_turn.append(word_count(m["content"]))
    lens = sorted(per_turn)
    in_band = sum(1 for w in lens if 80 <= w <= 220)
    print(f"\nanswer words: min={lens[0]} p10={lens[len(lens)//10]} "
          f"p50={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} max={lens[-1]}")
    print(f"assistant turns in 80-220 band: {in_band}/{len(lens)} "
          f"({in_band/len(lens)*100:.0f}%)")
    multi = sum(1 for r in train if r["_meta"].get("turns", 1) > 1)
    print(f"multi-turn conversations: {multi}/{total} ({multi/total*100:.0f}%)")
    print(f"\nwrote {out_dir}/train.jsonl and holdout.jsonl")


if __name__ == "__main__":
    main()
