# serve: AGBE as an endpoint

`guard.py` wraps any model with four deterministic steps: hard rules for doses, medical advice, illegal crops and self-harm; retrieval over cited, licensed passages; a claim-level check that removes unsupported sentences; a stop policy. `app.py` serves it as `/advise` and an OpenAI-compatible `/v1/chat/completions`. `backends.py` calls llama.cpp on CPU or any OpenAI-style server.

```bash
pip install -r serve/requirements.txt
AGBE_GGUF=model/agbe-1b-q4_k_m.gguf AGBE_PASSAGES=data/fulltext/passages.jsonl uvicorn serve.app:app --port 8080
curl -s localhost:8080/advise -H 'Content-Type: application/json' -d '{"question":"How do I keep stored grain safe from rodents?"}'
```

Tests: `AGBE_AUTOSTART=0 python3 -m unittest serve.test_guard serve.test_app` (the app tests need fastapi and httpx).

## What the first real runs showed (18 Oct 2026)

- With **v16** as the generator, the guard abstained on every question. v16 was trained to answer from its own memory and ignores the sources it is given, so its sentences are not in the passages and are removed. The guard did its job (it also stripped a llama.cpp banner that a backend bug leaked into an answer), but v16 plus retrieval is not a product.
- With an instruction-following model (**Qwen2.5-3B Q4**, CPU) the guard produced a cited, fully supported answer on stored-grain rodents, trimmed half of a banana bunchy top answer, and abstained when the model's cassava numbers were not in the sources.
- Conclusion: the v2 model has to be trained to answer from sources (source-grounded SFT with citations, then preference training on supported versus unsupported answers). The bake-off in `rag` mode measures which base does this best.
