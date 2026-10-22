"""A shorter prompt: rank the retrieved passages, keep the best few, and trim each to the sentences that matter for the question.

On a laptop CPU most of an answer's time is the model reading its prompt (about 750 tokens: three passages and the instructions), not writing
the answer. Fewer and shorter passages cut that time. The guard still checks the answer against the FULL text of the passages that were
shown, so trimming the prompt does not weaken the check.
"""
from __future__ import annotations

import re

STOP = set("""the and for are but not you your with that this from have has had was were can will would should could may might into onto than then
them they their there what when where which while who whom whose why how does did doing done its it's our out off over under about after before
between during each more most other some such only own same too very just also any all both few many much per via use used using one two""".split())


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOP}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def relevance(question: str, text: str) -> float:
    q = words(question)
    return len(q & words(text)) / len(q) if q else 0.0


def compact(question: str, text: str, max_words: int, must_keep: list[str] | None = None) -> str:
    """Keep the sentences that share the most words with the question, in their original order, up to max_words (at least one).
    Sentences that hold any of `must_keep` (verbatim quotes the answer is built from) are always kept, so training answers stay supported
    by the trimmed text; at serving time there is no such list and relevance alone decides."""
    sents = sentences(text)
    if sum(len(s.split()) for s in sents) <= max_words:
        return text.strip()
    norm = lambda t: " ".join(t.split()).lower()
    quotes = [norm(q) for q in (must_keep or []) if q]
    keep = {i for i, s in enumerate(sents) if any(q[:40] in norm(s) or norm(s) in q or norm(s)[:40] in q for q in quotes)}   # a quote may span sentences
    total = sum(len(sents[i].split()) for i in keep)
    for i in sorted(range(len(sents)), key=lambda i: (-relevance(question, sents[i]), i)):
        if i in keep:
            continue
        n = len(sents[i].split())
        if keep and total + n > max_words:
            continue
        keep.add(i); total += n
        if total >= max_words:
            break
    return " ".join(sents[i] for i in sorted(keep))


def prepare_sources(question: str, passages: list[dict], k: int = 2, max_words: int = 80) -> tuple[list[dict], list[dict]]:
    """Return (shown, full): the ranked, trimmed passages for the prompt and the same passages untrimmed for the guard, in the same order."""
    ranked = sorted(passages, key=lambda p: -relevance(question, p["text"]))[: k or len(passages)]
    shown = [dict(p, text=compact(question, p["text"], max_words) if max_words else p["text"]) for p in ranked]
    return shown, [dict(p) for p in ranked]
