#!/usr/bin/env python3
"""List every source behind a passages or claims file, with its licence and a count, for attribution and audits.

    python3 tools/licence_manifest.py data/fulltext_pilot/passages.jsonl.gz > data/LICENCE_MANIFEST.md
"""
import collections, gzip, json, sys

path = sys.argv[1]
op = gzip.open if path.endswith(".gz") else open
rows = [json.loads(l) for l in op(path, "rt") if l.strip()]
by = collections.OrderedDict()
for r in rows:
    d = by.setdefault(r["handle"], {"title": r["title"], "uri": r.get("uri"), "licence": r["licence"], "issued": (r.get("issued") or "")[:4], "n": 0})
    d["n"] += 1
bad = [d for d in by.values() if d["licence"] not in ("CC-BY-4.0", "CC-BY-3.0", "CC0-1.0", "Public Domain")]
print(f"# Licence manifest\n\n{len(rows)} records from {len(by)} sources. Records outside Tier A: {len(bad)}.\n")
print("| Source | Year | Licence | Records | Link |\n|---|---|---|---|---|")
for h, d in sorted(by.items(), key=lambda kv: kv[1]["title"]):
    print(f"| {d['title'][:100].replace('|', '/')} | {d['issued']} | {d['licence']} | {d['n']} | {d['uri']} |")
if bad:
    sys.exit("NOT TIER A: " + ", ".join(d["title"] for d in bad))
