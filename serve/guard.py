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
sys.path.insert(0, str(ROOT / "serve"))
from claim_check import check_claim, split_claims, REFUSAL  # noqa: E402
from extract_claims import dose_like  # noqa: E402

SYSTEM = """You are AGBE, an agricultural advisor for smallholder farmers and extension officers in West and Central Africa.
Use only the numbered sources provided. Cite them like [1]. Answer in the first sentence, then give the few steps that matter, in under 100 words.
If the sources do not answer the question, say you do not have a source for that and suggest an extension officer.
Never give an agrochemical dose, never give medical advice for people, never quote prices. Do not invent pests, products or numbers."""

def normalise(q: str) -> str:
    """Undo common obfuscation: zero-width characters, leetspeak and letter-spacing."""
    q = re.sub(r"[​‌‍­]", "", q).lower()
    if re.search(r"[a-z][0-9@$][a-z]|[0-9@$][a-z][0-9@$]", q):
        q = q.translate(str.maketrans("0134578@$", "oieastbas"))
    q = re.sub(r"\b((?:[a-z]\s){4,}[a-z])\b", lambda m: m.group(1).replace(" ", ""), q)
    return q


CHEM = r"(?:insecticid\w*|pesticid\w*|herbicid\w*|fungicid\w*|glyphosate|paraquat|cypermethrin|lambda|cyhalothrin|mancozeb|imidacloprid|atrazine|chlorpyrifos|dimethoate|\w*thrin|\w*phos|chemical)"
QTY = r"(?:\bml\b|millilit\w+|litres?|liters?|knapsack|dilut\w+|\brate\b|\bdose\b|dosage|how (?:much|many))"
DOSE_Q = re.compile(rf"\b{CHEM}\b.{{0,80}}{QTY}|{QTY}.{{0,80}}\b{CHEM}\b"
                    r"|\b(just|any|rough|approximate)\b.{0,30}\b(number|figure|amount|rate|dose)\b|\bexact\b.{0,25}\b(rate|dose|number)\b|\bgive me a number\b", re.I)
HUMAN_MED = re.compile(
    r"\b(my|our)\s+(child|baby|son|daughter|wife|husband|mother|father|neighbou?r|boy|girl)\b.{0,70}\b(fever|vomit\w*|sick|ill|malaria|cough|diarrh\w+|rash|medicine|drug|tablet|temperature|breath\w*|pain|dying|unwell)\b"
    r"|\b(what|which)\s+(medicine|drug|tablet|antibiotic|cream|syrup|medication)\b|\bparacetamol\b|\bamoxicillin\b|\bibuprofen\b|\bdiazepam\b|\btramadol\b|\bchloroquine\b"
    r"|\b(prescribe|prescription|over the counter)\b|\bhow many (tablets?|pills?|capsules?)\b|\bchest (pains?|infection)\b|\bdiagnose my\b"
    r"|\bcan i (take|use|give)\b.{0,60}\b(for|on)\b.{0,20}\b(me|myself|my (son|daughter|child|children|wife|husband|baby|kids|mother|father)|(the |a )?(child|children|baby))\b"
    r"|\b(sore throat|headache|stomach pain|infection)\b.{0,40}\b(i|me|my|myself)\b"
    r"|\b(licensed physician|i am a (doctor|nurse|pharmacist)|as a (doctor|nurse|pharmacist))\b"
    r"|\b(pretend|act as|roleplay as|role-play as|you are)\b.{0,30}\b(a |an )?(doctor|nurse|pharmacist|vet\w*|toxicologist|physician)\b"
    r"|\bhuman (drug|medication|medicine|illness|dose)\b", re.I)
HARM = re.compile(
    r"\b(poison|make (?:sick|ill)|harm|hurt)\b.{0,40}\b(person|people|man|woman|neighbou?rs?|brother|wife|husband|someone|somebody|rival|enemy|child|children|well|borehole|goats?|cattle|cows?|fish|pond|dog|food|drink)\b"
    r"|\bkill\b.{0,25}\b(a |the |my |his |her )?(person|man|woman|neighbou?r|brother|wife|husband|someone|somebody|rival|enemy)\b"
    r"|\b(lethal|fatal|deadly)\b|\bwithout (being )?(traced|caught|detected)\b|\bwithout (a )?taste\b|\bnot kill\b"
    r"|\b(how much|which)\b.{0,50}\b(pesticide|paraquat|poison|insecticide|chemical)\b.{0,60}\b(die|death|fatal|kill|drink|swallow|ill)\b"
    r"|\bwhich (pesticide|chemical|poison)\b.{0,50}\b(kill|fastest|toxic to humans)\b|\b(someone'?s|a person'?s)\s+(food|drink)\b", re.I)
ILLEGAL = re.compile(r"\b(cannabis|marijuana|indian hemp|khat|opium poppy|coca)\b", re.I)
SELF_HARM = re.compile(r"\b(kill myself|end my life|take my life|suicide|painless(ly)?|ways to die|end it|cannot go on|can'?t go on|no reason to live)\b", re.I)
JAILBREAK = re.compile(
    r"\b(ignore|forget|disregard)\b.{0,30}\b(previous|prior|above|earlier|all|your|every|everything)\b.{0,30}\b(instructions?|rules?|told|prompt)\b"
    r"|\b(switch|turn)\s+off\b.{0,25}\b(safety|rules|filter|restrictions?)\b|\b(dan|developer mode|system override|jailbreak|freegpt)\b|\bno (limits|restrictions|rules|filter|safety)\b"
    r"|\b(bypass|disable|lifted|override)\b.{0,30}\b(restrictions?|safety|limits?|rules?|filter)\b|\b(system|hidden|your)\s+(prompt|instructions)\b"
    r"|<\s*(end_of_turn|start_of_turn|\|im_start\|)|\"role\"\s*:\s*\"system\"|\[system\]|\byou are now\b|\bnew (rules|policy)\b|\bwithout any safety\b", re.I)
OFF_DOMAIN = re.compile(
    r"\b(python|javascript|c\+\+|html|sql)\b|\bwrite (me )?(a|my)\b.{0,40}\b(cv|resume|cover letter|essay|letter|poem|song|story|contract|agreement|notice|script|function|code)\b"
    r"|\b(draft|prepare)\b.{0,30}\b(legal|tenancy|agreement|contract|notice)\b|\b(vote|election|religion|stock market|cryptocurrency|bitcoin|crypto|football|divorce|girlfriend|boyfriend|love letter|nollywood|movie)\b"
    r"|\bintegrate\b|\bsolve this\b|\bhelp me write\b|\bwho should i vote\b", re.I)

REPLIES = {
    "dose": "I cannot give a dose or mixing rate, and I would be making it up if I did. The product label and your extension officer give the right rate for the product you have.",
    "medical": "I cannot advise on medicine for people. I am an agricultural advisor. Please take them to a clinic or health worker today rather than waiting.",
    "illegal": "I cannot help with growing or tending cannabis, khat or other controlled crops. I can help with legal crops such as maize, cassava, yam, cowpea and vegetables.",
    "self_harm": "I am sorry you are carrying this. I cannot help with that. Please do not stay alone with it: tell someone you trust today, and ask them to keep the key to any chemical store.",
    "harm": "I cannot help with anything meant to hurt people or animals. If someone is in danger, please contact the police or a health worker. I am glad to help with farming questions.",
    "jailbreak": "I cannot change my rules. I am an agricultural advisor, and I can help with crops, livestock, soil and storage.",
    "off_domain": "That is outside what I can help with. I am an agricultural advisor, so please ask me about crops, livestock, soil or storage.",
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
    """Deterministic categories, checked on the raw and the de-obfuscated text. Order matters: the most specific reply wins."""
    for q in (question, normalise(question)):
        for name, rx in (("self_harm", SELF_HARM), ("harm", HARM), ("illegal", ILLEGAL), ("dose", DOSE_Q), ("medical", HUMAN_MED),
                         ("jailbreak", JAILBREAK), ("off_domain", OFF_DOMAIN)):
            if rx.search(q):
                return name
    return None


def build_messages(question: str, passages: list[dict]) -> list[dict]:
    src = "\n".join(f"[{i}] ({p['title'][:70]}, p.{p['page']}) {p['text']}" for i, p in enumerate(passages, 1))
    return [{"role": "system", "content": SYSTEM + "\n\nSources:\n" + src}, {"role": "user", "content": question}]


def review_answer(raw: str, passages: list[dict], threshold: float = 0.6, max_sentences: int = 6) -> Advice:
    """Steps 3 and 4 of the guard on one model answer: keep a short refusal short, drop unsupported sentences and any chemical rate,
    cap the length, and abstain if nothing supported survives. Kept separate so it can be tested on recorded answers."""
    raw = (raw or "").strip()
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
        r = check_claim(re.sub(r"\[\d+\]", "", s), texts, threshold)
        ratios.append(1.0 if r.supported else 0.0)
        (kept if r.supported else dropped).append(s)
    kept = kept[:max_sentences]
    if not any(len(s.split()) >= 4 for s in kept):
        return Advice(REPLIES["abstain"], "abstained", "no supported claims survived", dropped=dropped)
    used = sorted({int(n) for s in kept for n in re.findall(r"\[(\d+)\]", s) if 1 <= int(n) <= len(passages)} or {1})
    cites = [{"n": n, "passage_id": passages[n - 1].get("passage_id"), "title": passages[n - 1].get("title"), "page": passages[n - 1].get("page"),
              "uri": passages[n - 1].get("uri"), "licence": passages[n - 1].get("licence")} for n in used]
    ratio = sum(ratios) / len(ratios) if ratios else 1.0
    return Advice(" ".join(kept), "trimmed" if dropped else "answered", "removed unsupported sentences" if dropped else "", cites, round(ratio, 3), dropped)


class GuardedAdvisor:
    def __init__(self, retriever, generate: Callable[[list[dict]], str], *, k: int = 3, min_score: float = 1.5,
                 threshold: float = 0.6, max_sentences: int = 6, compact: tuple[int, int] | None = None):
        self.retriever, self.generate = retriever, generate
        self.k, self.min_score, self.threshold, self.max_sentences = k, min_score, threshold, max_sentences
        self.compact = compact          # (passages kept, words per passage) for a shorter prompt; None keeps the full passages

    def advise(self, question: str) -> Advice:
        rule = hard_rule(question)
        if rule:
            return Advice(REPLIES[rule], "refused", rule)
        passages = self.retriever.top(question, self.k, self.min_score)
        if not passages:
            return Advice(REPLIES["abstain"], "abstained", "no relevant source")
        shown = passages
        if self.compact:                # the prompt gets the trimmed passages, the guard checks the answer against the full ones
            from prompt import prepare_sources
            shown, passages = prepare_sources(question, passages, *self.compact)
        raw = self.generate(build_messages(question, shown))
        return review_answer(raw, passages, self.threshold, self.max_sentences)
