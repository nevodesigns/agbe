#!/usr/bin/env python3
"""Fetch the full text of Tier A items and cut it into citable passages.

Input is the list made by cgspace_harvest.py. For every selected item the licence is read again from
CGSpace at fetch time and the item is skipped unless it still states a Tier A licence. PDFs are
downloaded politely (few workers, retries), converted with pdftotext, cleaned, and cut into passages
that keep the handle, page and licence, so every later claim can point at its source.

    python3 tools/fetch_fulltext.py --pilot 80          # a balanced sample to prove the pipeline
    python3 tools/fetch_fulltext.py --all --limit 2000   # larger runs later
Outputs (git-ignored): data/fulltext/pdf/, data/fulltext/passages.jsonl, data/fulltext/manifest.jsonl
"""
from __future__ import annotations

import argparse, collections, gzip, hashlib, json, os, pathlib, re, subprocess, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://cgspace.cgiar.org/server/api"
UA = {"User-Agent": "agbe-source-registry/1.0 (nwokoloebube21@gmail.com)"}
TIER_A = {"CC-BY-4.0", "CC-BY-3.0", "CC0-1.0", "Public Domain"}
PREFER = ["Manual", "Brief", "Report", "Book Chapter", "Book", "Journal Article"]
PILOT_TOPICS = ["maize", "cassava", "yam", "cowpea", "tomato", "rice", "groundnut", "striga", "aflatoxin", "fall_armyworm",
                "cassava_mosaic", "stem_borer", "poultry", "newcastle", "coccidiosis", "goat_sheep", "ppr", "soil_fertility",
                "postharvest", "pesticide_safety", "weed", "plantain", "soybean", "intercropping"]
ROOT = pathlib.Path("data/fulltext")


def get(url, params=None, binary=False, timeout=120):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data if binary else json.loads(data)


def retry(fn, *a, **k):
    for i in range(3):
        try:
            return fn(*a, **k)
        except Exception as e:
            err = e
            time.sleep(3 * (i + 1))
    raise err


def find_pdf(handle):
    item = retry(get, f"{API}/pid/find", {"id": f"hdl:{handle}"})
    lic = [m["value"] for m in item["metadata"].get("dcterms.license", [])]
    if not lic or lic[0] not in TIER_A:
        return None, lic[0] if lic else None, None
    uuid = item["uuid"]
    bundles = retry(get, f"{API}/core/items/{uuid}/bundles")["_embedded"]["bundles"]
    for b in bundles:
        if b["name"] != "ORIGINAL":
            continue
        bs = retry(get, b["_links"]["bitstreams"]["href"], {"size": 50})["_embedded"]["bitstreams"]
        pdfs = [x for x in bs if x["name"].lower().endswith(".pdf") and x.get("sizeBytes", 0) < 30_000_000]
        if pdfs:
            pdfs.sort(key=lambda x: -x.get("sizeBytes", 0))
            return pdfs[0]["_links"]["content"]["href"], lic[0], pdfs[0]["name"]
    return None, lic[0], None


def pdf_pages(path):
    # default pdftotext keeps reading order, so two-column pages do not interleave
    out = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True, timeout=180).stdout
    return out.split("\f")


REF = re.compile(r"\b(19|20)\d\d\b")


def clean_pages(pages):
    # drop lines that repeat on many pages (running headers and footers)
    counts = collections.Counter(l.strip() for p in pages for l in set(p.split("\n")) if 8 < len(l.strip()) < 90)
    junk = {l for l, c in counts.items() if c >= max(4, len(pages) // 3)}
    cleaned = []
    for p in pages:
        lines = [l.rstrip() for l in p.split("\n") if l.strip() not in junk]
        t = "\n".join(lines)
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)          # de-hyphenate
        t = re.sub(r"[ \t]{2,}", " ", t)
        cleaned.append(t)
    return cleaned


def page_passages(text, target=170):
    """Cut one page into passages of roughly `target` words, dropping reference lists."""
    paras = [re.sub(r"\s*\n\s*", " ", x).strip() for x in re.split(r"\n\s*\n", text)]
    paras = [x for x in paras if len(x.split()) >= 8]
    out, buf, words = [], [], 0

    def flush():
        nonlocal buf, words
        if buf:
            t = " ".join(buf)
            is_refs = len(REF.findall(t)) >= 4 and ("et al" in t or t.count("(") >= 4)
            if not is_refs and len(t.split()) >= 25:
                out.append(t)
        buf, words = [], 0

    for x in paras:
        n = len(x.split())
        if words + n > target * 1.4 and buf:
            flush()
        buf.append(x)
        words += n
        if words >= target:
            flush()
    flush()
    return out


def build_passages(pages, meta):
    rows = []
    for pno, text in enumerate(pages, 1):
        for k, t in enumerate(page_passages(text), 1):
            pid = hashlib.sha1(f"{meta['handle']}|{pno}|{k}|{t[:40]}".encode()).hexdigest()[:12]
            rows.append({"passage_id": pid, "handle": meta["handle"], "page": pno, "text": t, "licence": meta["licence"],
                         "title": meta["title"], "uri": meta["uri"], "issued": meta["issued"], "topics": meta["topics"]})
    return rows


def process(meta):
    h = meta["handle"]
    safe = h.replace("/", "_")
    pdf = ROOT / "pdf" / f"{safe}.pdf"
    note = {"handle": h, "title": meta["title"]}
    try:
        if not pdf.exists():
            href, lic, name = find_pdf(h)
            if not href:
                return note | {"status": "skipped", "reason": f"no tier-a pdf (licence now: {lic})"}, []
            pdf.write_bytes(retry(get, href, binary=True, timeout=300))
            meta = dict(meta, licence=lic)
        pages = clean_pages(pdf_pages(pdf))
        rows = build_passages(pages, meta)
        return note | {"status": "ok", "pages": len(pages), "passages": len(rows), "licence": meta["licence"]}, rows
    except Exception as e:
        return note | {"status": "error", "reason": str(e)[:120]}, []


GOOD = re.compile(r"\b(manage|management|control|guide|manual|practice|practices|pest|disease|planting|storage|seed|spacing|fertili[sz]er|soil|weed|harvest|processing|extension|handbook|training|production|agronom|diagnos|symptom|vaccin|feed|housing|hygiene)\w*", re.I)
BAD = re.compile(r"\b(impact|econom|gender|policy|policies|value chain|adoption|willingness|household|livelihood|genome|genetic diversity|qtl|gwas|phylogen|model(l)?ing|remote sensing|satellite|survey)\w*", re.I)


def practical(r):
    t = r["title"] + " " + r["abstract"][:600]
    return len(GOOD.findall(t)) - 2 * len(BAD.findall(t))


def pick(rows, n, pilot):
    ok = [r for r in rows if r["language"] in ("en", None) and r["type"] in PREFER and len(r["abstract"]) > 300 and (r["issued"] or "0") >= "2014"]
    ok.sort(key=lambda r: (-practical(r), PREFER.index(r["type"])))
    ok = [r for r in ok if practical(r) >= 2]
    if not pilot:
        return ok[:n]
    chosen, seen = [], set()
    per = max(1, n // len(PILOT_TOPICS))
    for t in PILOT_TOPICS:
        c = 0
        for r in ok:
            if t in r["topics"] and r["handle"] not in seen and ("IITA" in " ".join(r["affiliation"]) or "Tropical Agriculture" in " ".join(r["affiliation"]) or "Livestock" in " ".join(r["affiliation"]) or "AfricaRice" in " ".join(r["affiliation"])):
                chosen.append(r); seen.add(r["handle"]); c += 1
                if c >= per:
                    break
    return chosen[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="data/sources/cgspace_tier_a.jsonl.gz")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (ROOT / "pdf").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in gzip.open(a.registry, "rt")]
    sel = pick(rows, a.pilot or a.limit, bool(a.pilot))
    print(f"selected {len(sel)} items", flush=True)
    done = set()
    mf = ROOT / "manifest.jsonl"
    if mf.exists():
        done = {json.loads(l)["handle"] for l in open(mf) if '"status": "ok"' in l}
    sel = [r for r in sel if r["handle"] not in done]
    with ThreadPoolExecutor(a.workers) as ex, open(mf, "a") as m, open(ROOT / "passages.jsonl", "a") as p:
        for n, (note, prow) in enumerate(ex.map(process, sel), 1):
            m.write(json.dumps(note) + "\n"); m.flush()
            for r in prow:
                p.write(json.dumps(r, ensure_ascii=False) + "\n")
            p.flush()
            print(f"{n}/{len(sel)} {note['status']:8s} {note.get('passages', '')} {note['title'][:60]}", flush=True)


if __name__ == "__main__":
    main()
