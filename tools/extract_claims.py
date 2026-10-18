#!/usr/bin/env python3
"""Turn source passages into verified, citable claims.

A claim is one atomic fact. It is accepted only if code can prove it came from the passage:
  1. its quote appears verbatim in the passage (after harmless normalisation),
  2. every number in the claim appears in the quote,
  3. most of the claim's content words appear in the quote, so a paraphrase cannot add facts,
  4. it is not a pesticide or agrochemical rate, which AGBE never states (those go to review).

Backends:
  extractive    no model: picks instruction-like sentences. Claim equals quote. CPU only, a baseline.
  transformers  a Hugging Face instruct model on a GPU (Kaggle or Colab).
  openai        any OpenAI-compatible endpoint, for example a llama.cpp server.
  llamacli      llama-cli on a local GGUF (--model path.gguf), CPU only and slow.

    python3 tools/extract_claims.py --passages data/fulltext/passages.jsonl --out data/claims/claims.jsonl --backend extractive
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, re, sys

STOP = set("""a an the and or but if then than so to of in on at by for with from as is are was were be been being it its this that these those
i you he she we they me my your our their do does did not no can could should would will may might must have has had into out up down over
under about after before while when where which who whom whose what how why also very more most some any each other such only just too
there here""".split())
WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
WORDS.update({"fifteen": "15", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100"})
NUM = re.compile(r"\d+(?:[.,]\d+)?")
# agrochemical rates: a quantity of product per area, volume or plant. Never stated, always reviewed.
PESTICIDE = re.compile(r"\b(pesticid|insecticid|herbicid|fungicid|acaricid|nematicid|rodenticid|molluscicid|glyphosate|paraquat|cypermethrin|"
                       r"lambda|imidacloprid|mancozeb|chlorpyrifos|dimethoate|atrazine|2,4-d|neem oil|spray(ed|ing)?|dilut\w+)\b", re.I)
QUANTITY = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:ml|mL|l|L|litres?|liters?|g|kg|cc|oz|ounces?|tablespoons?|teaspoons?|tbsp|tsp|%|ppm)\b|\bper\s+(?:ha|hectare|acre|litre|liter|plant|knapsack|tank)\b|/\s*(?:ha|l|L)\b", re.I)
DOSE_WORDS = re.compile(r"\b(a\.?i\.?|active ingredient|dosage|application rate|dose)\b", re.I)


def dose_like(s: str) -> bool:
    """Conservative: any chemical word with a quantity, any mass or volume quantity per area or volume, or dose wording."""
    return bool(DOSE_WORDS.search(s) or (PESTICIDE.search(s) and QUANTITY.search(s)) or re.search(r"\d\s*(?:ml|mL|g|kg)\b.{0,40}\b(?:per|/|in|to|with)\b.{0,12}\b(?:litre|liter|l|L|ha|hectare|acre|plant|tank)", s, re.I))


FIRST_PERSON = re.compile(r"\b(we|our|this study|this paper|the authors|in this report|figure \d|table \d|respondents?|survey(ed)?|sampled?|regression|accessions?|"
                          r"biplot|choice sets?|statistical(ly)?|significant(ly)?|hypothes[ie]s|questionnaire|enumerators?|focus group|plots?|treatments?|replicat\w+|"
                          r"split[- ]plot|experiment(al|s)?|trials?|WAP|field observation)\b|p\s*[<=]\s*0\.|\u2022|\b(19|20)\d\d\b", re.I)
SIGNAL = re.compile(r"\b(should|must|can be|recommended|use|apply|plant|harvest|store|dry|control|scout|rotate|space|spacing|remove|prevent|avoid|"
                    r"symptoms?|caused by|spread|feeds?|infest|seeds?|cuttings?|weed|mulch|fertili[sz]er|manure|vaccin|clean|moisture|days|weeks)\b", re.I)


def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    s = re.sub(r"[­​]", "", s)
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 2}


def numbers(s: str) -> set[str]:
    found = {n.replace(",", "") for n in NUM.findall(s)}
    return found | {WORDS[w] for w in re.findall(r"[a-z]+", s.lower()) if w in WORDS}


def verify(claim: str, quote: str, passage: str, min_cover: float = 0.5) -> tuple[str, list[str]]:
    """Return ('accepted' | 'review' | 'rejected', reasons)."""
    why = []
    if not quote or norm(quote) not in norm(passage):
        why.append("quote_not_in_passage")
    if not numbers(claim) <= numbers(quote):
        why.append("number_not_in_quote")
    ct = tokens(claim)
    if ct and len(ct & tokens(quote)) / len(ct) < min_cover:
        why.append("claim_adds_words")
    n = len(claim.split())
    if n < 6 or n > 45:
        why.append("length")
    if FIRST_PERSON.search(claim):
        why.append("study_language")
    if why:
        return "rejected", why
    if dose_like(claim) or dose_like(quote):
        return "review", ["dose_like"]
    return "accepted", []


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def extractive(passage: dict) -> list[dict]:
    out = []
    for s in sentences(passage["text"]):
        w = len(s.split())
        if 8 <= w <= 40 and SIGNAL.search(s) and not s.endswith("?") and not re.search(r"\(.*\d{4}\)|et al", s):
            out.append({"claim": s, "quote": s, "crop_or_animal": "", "type": "other"})
    return out


PROMPT = """Extract atomic agronomic facts from the passage for an advisor that answers farmers.
Rules:
- One fact per item, one plain sentence, self-contained (no 'it', 'this', 'the study').
- 'quote' must be an exact span copied from the passage that supports the fact.
- Only facts a farmer or extension officer can act on or use to recognise a problem.
- Never include pesticide or agrochemical doses, rates or product names.
- Skip how an experiment was run (dates, plot design, statistics), results tables, funding, and background about a country or a study.
- Prefer general practices, how to recognise a problem, thresholds, timings and storage. At most 5 facts.
- If there are none, return [].
Return only JSON: [{"claim": "...", "quote": "...", "crop_or_animal": "...", "type": "practice|diagnosis|timing|spacing|storage|safety|other"}]

Passage:
"""


def parse_string_list(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return [x.strip() for x in json.loads(m.group(0)) if isinstance(x, str) and 8 <= len(x) <= 200]
    except Exception:
        return []


def parse_json_list(text: str) -> list[dict]:
    """Parse a JSON array of claim objects, recovering whole objects when the model was cut off mid-list."""
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            return [d for d in data if isinstance(d, dict) and "claim" in d and "quote" in d]
        except Exception:
            pass
    out = []
    for o in re.finditer(r"\{[^{}]*\}", text):
        try:
            d = json.loads(o.group(0))
        except Exception:
            continue
        if isinstance(d, dict) and "claim" in d and "quote" in d:
            out.append(d)
    return out


def make_transformers(model_id: str, prompt: str = None, parse=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, device_map="auto")

    prompt = prompt or PROMPT
    parse = parse or parse_json_list

    def run(passage):
        msgs = [{"role": "user", "content": prompt + passage["text"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=700, do_sample=False)
        return parse(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    return run


def make_openai(url: str, model: str, prompt: str = None, parse=None):
    import urllib.request
    prompt = prompt or PROMPT
    parse = parse or parse_json_list

    def run(passage):
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": 700,
                           "messages": [{"role": "user", "content": prompt + passage["text"]}]}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return parse(json.load(r)["choices"][0]["message"]["content"])
    return run


def make_llamacli(gguf: str, prompt: str = None, parse=None):
    """Local CPU backend: llama-cli on a GGUF. Slow, but needs no GPU and no server."""
    import subprocess
    base_prompt = prompt or PROMPT
    parse = parse or parse_json_list

    def run(passage):
        prompt = base_prompt + passage["text"]
        out = subprocess.run(["llama-cli", "-m", gguf, "-t", "4", "-ngl", "0", "-c", "4096", "-n", "520", "--temp", "0", "-st",
                              "--simple-io", "--no-warmup", "-p", prompt], capture_output=True, text=True, timeout=1200,
                             stdin=subprocess.DEVNULL).stdout
        out = out.split("[ Prompt:")[0]
        i = out.rfind("Passage:")  # skip any echo of the instructions and their example object
        got = parse(out[i:] if i >= 0 else out)
        return [d for d in got if not (isinstance(d, dict) and d.get("claim") == "...")]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["extractive", "transformers", "openai", "llamacli"], default="extractive")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run = extractive if a.backend == "extractive" else None
    if a.backend == "transformers":
        run = make_transformers(a.model)
    elif a.backend == "openai":
        run = make_openai(a.url, a.model)
    elif a.backend == "llamacli":
        run = make_llamacli(a.model)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, stats = set(), {"accepted": 0, "review": 0, "rejected": 0}
    with open(a.passages) as f, open(out, "w") as o:
        for n, line in enumerate(f, 1):
            if a.limit and n > a.limit:
                break
            p = json.loads(line)
            try:
                items = run(p)
            except Exception as e:
                print("error", p["passage_id"], str(e)[:80], file=sys.stderr)
                continue
            for it in items:
                status, why = verify(it["claim"], it["quote"], p["text"])
                key = norm(it["claim"])
                if status == "accepted" and key in seen:
                    status, why = "rejected", ["duplicate"]
                if status == "accepted":
                    seen.add(key)
                stats[status] += 1
                cid = hashlib.sha1((p["passage_id"] + key).encode()).hexdigest()[:12]
                o.write(json.dumps({"claim_id": cid, "claim": it["claim"], "quote": it["quote"], "status": status, "reasons": why,
                                    "crop_or_animal": it.get("crop_or_animal", ""), "type": it.get("type", "other"),
                                    "handle": p["handle"], "page": p["page"], "passage_id": p["passage_id"], "licence": p["licence"],
                                    "uri": p["uri"], "title": p["title"]}, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                print(n, stats, flush=True)
    print("done", stats)


if __name__ == "__main__":
    main()
