#!/usr/bin/env python3
"""Score a GGUF file on the held-out cases and time it on this CPU, with llama.cpp's own server.

The merged float16 model was scored before it was shrunk to 4-bit. This scores the file that would actually ship, with the same
prompts, the same greedy decoding and the same scorer, and records speed and peak memory. llama.cpp is built from the same source
that made the GGUF, so a version mismatch cannot refuse the file. One model per process, so peak memory is that model's alone.

    python3 train/gguf_eval.py --server llama.cpp/build/bin/llama-server --gguf agbe-v2-qwen17-q4_k_m.gguf \\
        --tokenizer Qwen/Qwen3-1.7B --eval rag_eval.jsonl --out gguf_qwen17.json
"""
from __future__ import annotations

import argparse, json, pathlib, platform, subprocess, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "train"))
from rag_eval import score, summarise  # noqa: E402
from serve.guard import build_messages  # noqa: E402
from serve.prompt import prepare_sources  # noqa: E402


def render(tok, messages) -> str:
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def post(base: str, path: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(base: str, seconds: int = 600):
    end = time.time() + seconds
    while time.time() < end:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("llama-server did not become ready")


def case(rec: dict, k: int, max_words: int):
    """The record the model is shown (ranked, trimmed passages) and the record it is scored against (the same passages, full text)."""
    if not k and not max_words:
        return rec, rec
    shown, full = prepare_sources(rec["question"], rec["sources"], k, max_words)
    idx = None
    if rec["kind"] == "grounded":
        idx = next((i for i, p in enumerate(full, 1) if p.get("passage_id") == rec.get("passage_id")), None)   # None if the retriever missed it
    return dict(rec, sources=shown), dict(rec, sources=full, expected_index=idx)


def run(base: str, tok, recs, max_new: int, k: int = 0, max_words: int = 0):
    rows, p_n, p_ms, g_n, g_ms = [], 0, 0.0, 0, 0.0
    for i, r0 in enumerate(recs, 1):
        r, scored = case(r0, k, max_words)
        text = render(tok, build_messages(r["question"], r["sources"]))
        # tokenise on the server without adding a BOS: the chat template already writes one
        ids = post(base, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
        out = post(base, "/completion", {"prompt": ids, "n_predict": max_new, "temperature": 0.0, "repeat_penalty": 1.05, "cache_prompt": False, "stream": False})
        ans = out["content"].strip()
        t = out.get("timings", {})
        p_n += t.get("prompt_n", len(ids)); p_ms += t.get("prompt_ms", 0.0); g_n += t.get("predicted_n", 0); g_ms += t.get("predicted_ms", 0.0)
        rows.append({"kind": r["kind"], "question": r["question"], "answer": ans, **score(scored, ans)})
        if i % 10 == 0:
            print(i, "/", len(recs), flush=True)
    speed = {"cases": len(recs), "mean_prompt_tokens": round(p_n / max(1, len(recs))),
             "prompt_tokens_per_s": round(p_n / (p_ms / 1000), 1) if p_ms else None,
             "generation_tokens_per_s": round(g_n / (g_ms / 1000), 2) if g_ms else None,
             "mean_seconds_per_answer": round((p_ms + g_ms) / 1000 / max(1, len(recs)), 1)}
    return rows, speed


def peak_memory_mb(pid: int):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM"):
                return round(int(line.split()[1]) / 1024)
    except OSError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True); ap.add_argument("--gguf", required=True); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--eval", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--max-new", type=int, default=200); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sources", type=int, default=0, help="keep the best K passages (0 keeps all)"); ap.add_argument("--max-words", type=int, default=0, help="trim each passage to about this many words (0 keeps it whole)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    recs = [json.loads(l) for l in open(a.eval) if l.strip()]
    if a.limit:
        recs = recs[: a.limit]
    base = f"http://127.0.0.1:{a.port}"
    cmd = [a.server, "-m", a.gguf, "-c", str(a.ctx), "-t", str(a.threads), "--host", "127.0.0.1", "--port", str(a.port), "-np", "1"]
    # recent llama-server keeps a RAM cache of earlier prompts (default up to 8 GB) and context checkpoints; they inflate the peak
    # memory far beyond what the model needs, so switch them off when this build knows the flags
    try:
        helptext = subprocess.run([a.server, "--help"], capture_output=True, text=True, timeout=60)
        helptext = helptext.stdout + helptext.stderr
    except Exception:
        helptext = ""
    if "--cache-ram" in helptext:
        cmd += ["--cache-ram", "0"]
    if "--ctx-checkpoints" in helptext:
        cmd += ["--ctx-checkpoints", "0"]
    print("server command:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(base)
        rows, speed = run(base, tok, recs, a.max_new, a.sources, a.max_words)
        speed.update(sources=a.sources, max_words=a.max_words, threads=a.threads, cpu=cpu_name(), peak_memory_mb=peak_memory_mb(proc.pid), file_mb=round(pathlib.Path(a.gguf).stat().st_size / 1e6))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    summ = summarise(rows)
    json.dump({"model": pathlib.Path(a.gguf).name, "summary": summ, "speed": speed, "rows": rows}, open(a.out, "w"), indent=1)
    print(pathlib.Path(a.gguf).name, summ); print("speed", speed)


if __name__ == "__main__":
    main()
