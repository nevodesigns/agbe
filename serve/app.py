"""AGBE as an endpoint.

    pip install fastapi uvicorn
    AGBE_GGUF=model/agbe-1b-q4_k_m.gguf AGBE_PASSAGES=data/fulltext/passages.jsonl uvicorn serve.app:app --port 8080

    POST /advise                {"question": "..."}  ->  answer, status, citations, supported_ratio, dropped
    POST /v1/chat/completions   OpenAI-compatible; the last user message goes through the guard
    GET  /health

Set AGBE_API_KEYS (comma separated) to require `Authorization: Bearer <key>`. Nothing is logged by default and no
question leaves the machine unless AGBE_BACKEND_URL points at a remote model.
"""
from __future__ import annotations

import os, pathlib, sys, time
from dataclasses import asdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from serve.guard import GuardedAdvisor  # noqa: E402
from serve.retrieval import Retriever  # noqa: E402

MAX_Q = 600


class Ask(BaseModel):
    question: str


class Msg(BaseModel):
    role: str
    content: str


class Chat(BaseModel):
    model: str = "agbe"
    messages: list[Msg]


def default_advisor() -> GuardedAdvisor:
    from serve import backends
    if os.environ.get("AGBE_BACKEND_URL"):
        gen = backends.openai_compatible(os.environ["AGBE_BACKEND_URL"], api_key=os.environ.get("AGBE_BACKEND_KEY", ""))
    else:
        gen = backends.llama_cli(os.environ.get("AGBE_GGUF", str(ROOT / "model/agbe-1b-q4_k_m.gguf")))
    return GuardedAdvisor(Retriever(os.environ.get("AGBE_PASSAGES", str(ROOT / "data/fulltext/passages.jsonl"))), gen)


def create_app(advisor: GuardedAdvisor | None = None, api_keys: set[str] | None = None) -> FastAPI:
    app = FastAPI(title="AGBE", version="2.0.0-dev")
    keys = api_keys if api_keys is not None else {k for k in os.environ.get("AGBE_API_KEYS", "").split(",") if k}
    state = {"advisor": advisor}

    def get_advisor() -> GuardedAdvisor:
        if state["advisor"] is None:
            state["advisor"] = default_advisor()
        return state["advisor"]

    def auth(authorization: str | None = Header(default=None)):
        if keys and (not authorization or authorization.removeprefix("Bearer ").strip() not in keys):
            raise HTTPException(401, "missing or wrong API key")

    @app.get("/health")
    def health():
        return {"ok": True, "version": app.version}

    @app.post("/advise", dependencies=[Depends(auth)])
    def advise(body: Ask):
        q = body.question.strip()
        if not q or len(q) > MAX_Q:
            raise HTTPException(422, f"question must be 1 to {MAX_Q} characters")
        return asdict(get_advisor().advise(q))

    @app.post("/v1/chat/completions", dependencies=[Depends(auth)])
    def chat(body: Chat):
        users = [m.content for m in body.messages if m.role == "user"]
        if not users or not users[-1].strip() or len(users[-1]) > MAX_Q:
            raise HTTPException(422, f"last user message must be 1 to {MAX_Q} characters")
        a = get_advisor().advise(users[-1].strip())
        return {"id": f"agbe-{int(time.time() * 1000)}", "object": "chat.completion", "created": int(time.time()), "model": body.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": a.answer}, "finish_reason": "stop"}],
                "agbe": {"status": a.status, "citations": a.citations, "supported_ratio": a.supported_ratio, "dropped": a.dropped}}

    return app


app = create_app() if os.environ.get("AGBE_AUTOSTART", "1") == "1" else None
