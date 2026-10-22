import json, os, sys, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_eval

SEEN = {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({"status": "ok"})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            SEEN["add_special"] = body["add_special"]; self._send({"tokens": list(range(len(body["content"].split())))})
        else:
            SEEN["prompt_is_tokens"] = isinstance(body["prompt"], list); SEEN["temp"] = body["temperature"]
            self._send({"content": " Use mulch to keep soil moist. [1]", "timings": {"prompt_n": 20, "prompt_ms": 200.0, "predicted_n": 10, "predicted_ms": 500.0}})


class FakeTok:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kw):
        return "<bos>" + messages[-1]["content"]


class T(unittest.TestCase):
    def test_run(self):
        srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        gguf_eval.wait_ready(base, 5)
        recs = [{"kind": "grounded", "question": "how to keep soil moist", "expected_index": 1, "claims": ["Use mulch to keep soil moist."], "passage_id": "p",
                 "sources": [{"title": "T", "page": 1, "text": "Use mulch to keep soil moist.", "passage_id": "p", "uri": "u", "licence": "CC-BY-4.0"}]}]
        rows, speed = gguf_eval.run(base, FakeTok(), recs, 50)
        srv.shutdown()
        self.assertEqual(rows[0]["answer"], "Use mulch to keep soil moist. [1]")
        self.assertIs(SEEN["add_special"], False)       # never add a second BOS
        self.assertTrue(SEEN["prompt_is_tokens"]); self.assertEqual(SEEN["temp"], 0.0)
        self.assertEqual(speed["generation_tokens_per_s"], 20.0); self.assertEqual(speed["prompt_tokens_per_s"], 100.0)

    def test_compact_prompt_scores_against_full_passages(self):
        long = "Maize is planted at the onset of the rains. " * 6 + "Space rows 75 cm apart and plants 25 cm apart within the row. " + "Bags stay on pallets. " * 6
        recs = [{"kind": "grounded", "question": "how far apart should I space maize rows", "expected_index": 2, "claims": [], "passage_id": "m",
                 "sources": [{"title": "X", "page": 1, "text": "Goats need water.", "passage_id": "g", "uri": "u", "licence": "CC-BY-4.0"},
                             {"title": "T", "page": 1, "text": long, "passage_id": "m", "uri": "u", "licence": "CC-BY-4.0"}]}]
        shown, scored = gguf_eval.case(recs[0], 1, 20)
        self.assertEqual(len(shown["sources"]), 1)
        self.assertLess(len(shown["sources"][0]["text"].split()), len(long.split()))
        self.assertEqual(scored["sources"][0]["text"], long)      # scored against the full passage
        self.assertEqual(scored["expected_index"], 1)             # the maize passage is now first


if __name__ == "__main__":
    unittest.main()
