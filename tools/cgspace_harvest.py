#!/usr/bin/env python3
"""Build the Tier A source list from CGSpace metadata (no full text is downloaded here).

Tier A means the item states an open licence that allows commercial reuse with attribution:
CC-BY-4.0, CC-BY-3.0, CC0-1.0 or public domain. Items with no stated licence are NOT Tier A, even
when the publisher's policy says "free to reuse with acknowledgment". NC and SA licences are counted
for transparency and never emitted.

    python3 tools/cgspace_harvest.py --out data/sources/cgspace_tier_a.jsonl
"""
from __future__ import annotations

import argparse, collections, json, time, urllib.parse, urllib.request

API = "https://cgspace.cgiar.org/server/api/discover/search/objects"
TIER_A = ["CC-BY-4.0", "CC-BY-3.0", "CC0-1.0", "Public Domain"]
EXCLUDED_COUNTED = ["CC-BY-NC-4.0", "CC-BY-NC-SA-3.0", "CC-BY-SA-4.0", "CC-BY-NC-3.0"]
TYPES = {"Report", "Brief", "Journal Article", "Book", "Book Chapter", "Manual", "Training Material",
         "Working Paper", "Extension Material", "Poster", "Thesis", "Conference Paper", "Case Study",
         "Newsletter", "Annual Report", "Guide", "Handbook"}

TOPICS = {
    "maize": "maize", "cassava": "cassava", "yam": "yam", "rice": "rice", "sorghum": "sorghum OR millet",
    "cowpea": "cowpea", "groundnut": "groundnut OR peanut", "tomato": "tomato", "plantain": "plantain OR banana",
    "soybean": "soybean", "cocoa": "cocoa", "oil_palm": "oil palm", "sweet_potato": "sweet potato",
    "pepper": "pepper AND (chilli OR capsicum)", "onion": "onion",
    "fall_armyworm": "fall armyworm", "stem_borer": "stem borer", "striga": "striga", "aflatoxin": "aflatoxin",
    "cassava_mosaic": "cassava mosaic", "cassava_brown_streak": "cassava brown streak", "whitefly": "whitefly",
    "blight": "late blight OR early blight", "bacterial_wilt": "bacterial wilt", "nematode": "nematode",
    "poultry": "poultry OR chicken", "newcastle": "newcastle disease", "coccidiosis": "coccidiosis",
    "goat_sheep": "goat OR sheep", "ppr": "peste des petits ruminants", "cattle": "cattle", "pig": "pig OR swine",
    "aquaculture": "catfish OR aquaculture OR tilapia",
    "soil_fertility": "soil fertility", "fertilizer": "fertilizer OR fertiliser", "nitrogen": "nitrogen deficiency",
    "intercropping": "intercropping", "seed_system": "seed system OR improved varieties",
    "postharvest": "postharvest OR post-harvest OR storage", "drying": "grain drying", "weed": "weed management",
    "pesticide_safety": "pesticide safety OR pesticide use", "extension": "extension advisory services",
    "climate_adaptation": "climate-smart agriculture", "irrigation": "irrigation", "market_access": "agricultural markets smallholder",
}
AFRICA_HINT = ("africa", "nigeria", "ghana", "benin", "cameroon", "niger", "mali", "burkina", "senegal", "togo", "cote", "ivoire",
               "tanzania", "kenya", "uganda", "ethiopia", "malawi", "zambia", "mozambique", "rwanda", "congo", "sierra leone", "liberia")


def get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "agbe-source-registry/1.0 (nwokoloebube21@gmail.com)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def vals(md, key):
    return [m["value"] for m in md.get(key, [])]


def record(it, topic, lic):
    md = it["metadata"]
    regions = vals(md, "cg.coverage.region") + vals(md, "cg.coverage.country")
    return {
        "handle": it.get("handle"), "uri": (vals(md, "dc.identifier.uri") or [None])[0], "title": it.get("name"),
        "authors": vals(md, "dc.contributor.author")[:12], "issued": (vals(md, "dcterms.issued") or [None])[0],
        "type": (vals(md, "dcterms.type") or [None])[0], "licence": lic, "language": (vals(md, "dcterms.language") or [None])[0],
        "regions": regions, "affiliation": vals(md, "cg.contributor.affiliation")[:6],
        "subjects": (vals(md, "dcterms.subject") + vals(md, "cg.subject.iita"))[:12],
        "abstract": " ".join(vals(md, "dcterms.abstract"))[:2500], "doi": (vals(md, "cg.identifier.doi") or [None])[0],
        "citation": (vals(md, "dcterms.bibliographicCitation") or [None])[0], "topics": [topic],
    }


def fetch(topic, q, lic, page):
    query = f'({q}) AND dcterms.license:"{lic}"'
    for attempt in range(3):
        try:
            d = get({"query": query, "size": 100, "page": page, "dsoType": "ITEM"})
            return topic, lic, page, d["_embedded"]["searchResult"]["_embedded"]["objects"]
        except Exception:
            time.sleep(3 * (attempt + 1))
    print("giving up", topic, lic, page, flush=True)
    return topic, lic, page, []


def keep(it):
    md = it["metadata"]
    if (vals(md, "dcterms.type") or [None])[0] not in TYPES:
        return False
    if (vals(md, "dcterms.language") or ["en"])[0] not in ("en", "fr"):
        return False
    reg = " ".join(vals(md, "cg.coverage.region") + vals(md, "cg.coverage.country")).lower()
    return not reg or any(h in reg for h in AFRICA_HINT)


def main():
    from concurrent.futures import ThreadPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sources/cgspace_tier_a.jsonl")
    ap.add_argument("--pages", type=int, default=2, help="pages of 100 per topic for CC-BY-4.0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary", default="data/sources/cgspace_summary.json")
    a = ap.parse_args()
    # CC-BY-4.0 is the bulk, so it is searched per topic. The rare Tier A licences are fetched whole and filtered here.
    jobs = [(t, q, "CC-BY-4.0", p) for t, q in TOPICS.items() for p in range(a.pages)]
    jobs += [("any", "agriculture OR crop OR livestock OR farming OR soil", lic, p) for lic in ("CC0-1.0", "CC-BY-3.0", "Public Domain") for p in range(2)]
    items, counts = {}, collections.Counter()
    with ThreadPoolExecutor(a.workers) as ex:
        for n, (topic, lic, page, objs) in enumerate(ex.map(lambda j: fetch(*j), jobs), 1):
            for o in objs:
                it = o["_embedded"]["indexableObject"]
                if not keep(it):
                    continue
                h = it.get("handle")
                if h in items:
                    if topic not in items[h]["topics"]:
                        items[h]["topics"].append(topic)
                else:
                    items[h] = record(it, topic, lic)
                counts[topic] += 1
            print(f"{n}/{len(jobs)} {topic:20s} {lic:10s} p{page} unique={len(items)}", flush=True)
    excluded = {}
    for lic in EXCLUDED_COUNTED:
        try:
            d = get({"query": f'dcterms.license:"{lic}"', "size": 1, "dsoType": "ITEM"})
            excluded[lic] = d["_embedded"]["searchResult"]["page"]["totalElements"]
        except Exception:
            excluded[lic] = None
    with open(a.out, "w") as f:
        for r in sorted(items.values(), key=lambda r: (r["issued"] or "", r["handle"] or "")):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"harvested": time.strftime("%Y-%m-%d"), "unique_items": len(items),
               "by_licence": collections.Counter(r["licence"] for r in items.values()),
               "by_topic_hits": counts, "top_affiliations": collections.Counter(x for r in items.values() for x in r["affiliation"]).most_common(15),
               "excluded_not_tier_a_total_items_on_cgspace_counted_only": excluded,
               "note": "Metadata only. Full text is fetched in a later step, and only for items listed here. Limited to 100 x pages per topic, so this is a sample, not the whole of Tier A."}
    json.dump(summary, open(a.summary, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
