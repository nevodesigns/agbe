"""Small BM25 retriever over cited passages. No dependencies, so it runs on a laptop or in a container."""
from __future__ import annotations

import collections, json, math, pathlib, statistics, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))
from claim_check import content_tokens  # noqa: E402


class Retriever:
    def __init__(self, path):
        path = pathlib.Path(path)
        self.rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
        self.toks = [content_tokens(r["text"]) for r in self.rows]
        df = collections.Counter(t for ts in self.toks for t in ts)
        n = max(1, len(self.rows))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.avg = statistics.mean([len(t) for t in self.toks]) if self.toks else 1

    def top(self, q: str, k: int = 3, min_score: float = 0.0):
        qt = content_tokens(q)
        scored = []
        for i, ts in enumerate(self.toks):
            s = sum(self.idf.get(t, 0) * 2.5 / (1 + 1.5 * (0.25 + 0.75 * len(ts) / self.avg)) for t in qt if t in ts)
            if s > min_score:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [dict(self.rows[i], score=round(s, 3)) for s, i in scored[:k]]
