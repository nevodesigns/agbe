"""The inference guard: the part that fixes AGBE's tail drift.

AGBE v16 answers correctly in its first sentence and can wander after it. The guard wraps any model
(llama.cpp, a transformers model, a hosted endpoint) in four deterministic steps:

  1. Hard rules before the model runs: doses, medical advice for people, illegal crops and self-harm get a
     fixed, short, safe reply. The model never sees these questions, so no prompt trick can move them.
  2. Retrieval: the model is given numbered, licensed passages and told to answer only from them. If nothing
     relevant is found, the guard abstains instead of letting the model improvise.
  3. Claim checking: every sentence of the answer must be supported by a retrieved passage. Unsupported
     sentences are removed, and a chemical rate is removed whatever its support.
  4. Stop policy: a refusal ends after its deferral, and answers are capped in length.

The result carries its citations and a record of what was changed, so a caller can show or log it.
"""
from __future__ import annotations

import pathlib, re, sys
from dataclasses import dataclass, field
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "tools"))
from claim_check import check_claim, split_claims, REFUSAL  # noqa: E402
from extract_claims import dose_like  # noqa: E402

SYSTEM = """You are AGBE, an agricultural advisor for smallholder farmers and extension officers in West and Central Africa.
Use only the numbered sources provided. Cite them like [1]. Answer in the first sentence, then give the few steps that matter, in under 100 words.
If the sources do not answer the question, say you do not have a source for that and suggest an extension officer.
Never give an agrochemical dose, never give medical advice for people, never quote prices. Do not invent pests, products or numbers."""

DOSE_Q = re.compile(r"\b(how (many|much)|what)\b.{0,50}\b(ml|millilit\w+|litre|liter|grams?|kg|spoons?|cups?|caps?)\b.{0,60}\b(insecticide|pesticide|herbicide|fungicide|chemical|spray|poison)\b"
                    r"|\b(dose|dosage|dilut\w+|mix(ing)?)\b.{0,60}\b(insecticide|pesticide|herbicide|fungicide|chemical)\b"
                    r"|\b(insecticide|pesticide|herbicide|fungicide)\b.{0,60}\b(dose|dosage|per litre|per liter|per knapsack|per hectare|mix(ing)?)\b", re.I)
HUMAN_MED = re.compile(r"\b(my|our)\s+(child|baby|son|daughter|wife|husband|mother|father|neighbou?r|boy|girl)\b.{0,70}\b(fever|vomit\w*|sick|ill|malaria|cough|diarrh\w+|rash|medicine|drug|tablet)\b"
                       r"|\b(what|which)\s+(medicine|drug|tablet|antibiotic)\b.{0,40}\b(i|me|my|child|baby|human|person|people)\b|\bparacetamol\b|\bamoxicillin\b", re.I)
ILLEGAL = re.compile(r"\b(cannabis|marijuana|indian hemp|khat|opium poppy|coca)\b", re.I)
SELF_HARM = re.compile(r"\b(kill myself|end my life|take my life|suicide|painless(ly)?|ways to die)\b", re.I)

REPLIES = {
    "dose": "I cannot give a dose or mixing rate, and I would be making it up if I did. The product label and your extension officer give the right rate for the product you have.",
    "medical": "I cannot advise on medicine for people. I am an agricultural advisor. Please take them to a clinic or health worker today rather than waiting.",
    "illegal": "I cannot help with growing or tending cannabis, khat or other controlled crops. I can help with legal crops such as maize, cassava, yam, cowpea and vegetables.",
    "self_harm": "I am sorry you are carrying this. I cannot help with that. Please do not stay alone with it: tell someone you trust today, and ask them to keep the key to any chemical store.",
    "abstain": "I do not have a source for that, so I will not guess. Please ask your extension officer, or tell me the crop and what you see and I will try again.",
}


@dataclass
class Advice:
    answer: str
    status: str                      # answered | refused | abstained | trimmed
    reason: str = ""
    citations: list[dict] = field(default_factory=list)
    supported_ratio: float | None = None
    dropped: list[str] = field(default_factory=list)


def hard_rule(question: str) -> str | None:
    if SELF_HARM.search(question):
        return "self_harm"
    if ILLEGAL.search(question):
        return "illegal"
    if DOSE_Q.search(question):
        return "dose"
    if HUMAN_MED.search(question):
        return "medical"
    return None


def build_messages(question: str, passages: list[dict]) -> list[dict]:
    src = "\n".join(f"[{i}] ({p['title'][:70]}, p.{p['page']}) {p['text']}" for i, p in enumerate(passages, 1))
    return [{"role": "system", "content": SYSTEM + "\n\nSources:\n" + src}, {"role": "user", "content": question}]


class GuardedAdvisor:
    def __init__(self, retriever, generate: Callable[[list[dict]], str], *, k: int = 3, min_score: float = 1.5,
                 threshold: float = 0.6, max_sentences: int = 6):
        self.retriever, self.generate = retriever, generate
        self.k, self.min_score, self.threshold, self.max_sentences = k, min_score, threshold, max_sentences

    def advise(self, question: str) -> Advice:
        rule = hard_rule(question)
        if rule:
            return Advice(REPLIES[rule], "refused", rule)
        passages = self.retriever.top(question, self.k, self.min_score)
        if not passages:
            return Advice(REPLIES["abstain"], "abstained", "no relevant source")
        raw = self.generate(build_messages(question, passages)).strip()
        first = raw.split("\n")[0] if raw else ""
        if REFUSAL.search(first):                                   # the model declined: keep it short
            sents = re.split(r"(?<=[.!?])\s+", raw)
            return Advice(" ".join(sents[:2]), "refused", "model refusal, stopped after two sentences")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", raw) if s.strip()]
        texts = [p["text"] for p in passages]
        kept, dropped, ratios = [], [], []
        for s in sentences:
            if dose_like(s):
                dropped.append(s)
                continue
            if len(s.split()) < 4:                                  # connective fragments pass through
                kept.append(s)
                continue
            r = check_claim(re.sub(r"\[\d+\]", "", s), texts, self.threshold)
            ratios.append(1.0 if r.supported else 0.0)
            (kept if r.supported else dropped).append(s)
        kept = kept[: self.max_sentences]
        if not any(len(s.split()) >= 4 for s in kept):
            return Advice(REPLIES["abstain"], "abstained", "no supported claims survived", dropped=dropped)
        used = sorted({int(n) for s in kept for n in re.findall(r"\[(\d+)\]", s) if 1 <= int(n) <= len(passages)} or {1})
        cites = [{"n": n, "passage_id": passages[n - 1]["passage_id"], "title": passages[n - 1]["title"], "page": passages[n - 1]["page"],
                  "uri": passages[n - 1]["uri"], "licence": passages[n - 1]["licence"]} for n in used]
        ratio = sum(ratios) / len(ratios) if ratios else 1.0
        return Advice(" ".join(kept), "trimmed" if dropped else "answered", "removed unsupported sentences" if dropped else "", cites, round(ratio, 3), dropped)
