"""Ways to call a model. Each returns a function from chat messages to text."""
from __future__ import annotations

import json, re, subprocess, urllib.request


def extract_answer(out: str, user: str) -> str:
    """llama-cli prints a banner, then '> <user message>', then the answer, then timing lines."""
    out = out.split("[ Prompt:")[0]
    marker = "> " + user.strip()
    i = out.rfind(marker)
    body = out[i + len(marker):] if i >= 0 else out
    return re.sub(r"\n?Exiting\.\.\.\s*$", "", body).strip()


def llama_cli(gguf: str, threads: int = 4, ctx: int = 2048, max_new: int = 220):
    """CPU llama.cpp, greedy and seeded, the settings AGBE's answers can be reproduced with."""
    def gen(messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = [m["content"] for m in messages if m["role"] == "user"][-1]
        cmd = ["llama-cli", "-m", gguf, "-t", str(threads), "-ngl", "0", "-c", str(ctx), "-n", str(max_new), "--temp", "0", "--seed", "1",
               "-st", "--no-warmup"]
        if system:
            cmd += ["-sys", system]
        out = subprocess.run(cmd + ["-p", user], capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL).stdout
        return extract_answer(out, user)
    return gen


def openai_compatible(base_url: str, model: str = "agbe", api_key: str = "", max_new: int = 220):
    """Any OpenAI-style server: llama.cpp server, vLLM, LM Studio, a hosted endpoint."""
    def gen(messages):
        body = json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": max_new}).encode()
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = "Bearer " + api_key
        req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", body, h)
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()
    return gen
