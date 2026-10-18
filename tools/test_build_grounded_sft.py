import json, pathlib, sys, tempfile, unittest
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import build_grounded_sft as b
from serve.guard import REPLIES

P = [{"passage_id": f"p{i}", "handle": f"h/{i}", "page": i, "title": t, "uri": "u", "licence": "CC-BY-4.0", "topics": [], "text": x} for i, (t, x) in enumerate([
    ("Banana guide", "Banana aphids spread banana bunchy top virus by feeding on infected plants and then moving to healthy plants. Remove infected mats at once and use clean suckers."),
    ("Grain store guide", "Store bags in tidy stacks on pallets with a space of one metre all round. Keep the area around the store free of tall weeds so rodents have no cover."),
    ("Cassava guide", "Cassava stem cuttings of about 25 cm with at least five nodes sprout best when taken from healthy plants."),
    ("Yam guide", "Yam seed tubers should be treated and dried before storage so that they do not rot in the barn."),
], 1)]
C = [{"passage_id": "p1", "status": "accepted", "claim": "Banana aphids spread banana bunchy top virus by feeding on infected plants and then moving to healthy plants."},
     {"passage_id": "p1", "status": "accepted", "claim": "Remove infected mats at once and use clean suckers."},
     {"passage_id": "p2", "status": "accepted", "claim": "Store bags in tidy stacks on pallets with a space of one metre all round."},
     {"passage_id": "p2", "status": "rejected", "claim": "invented extra"}]
Q = [{"passage_id": "p1", "raw": ["How does banana bunchy top spread?", "What do I do about infected banana mats?"]},
     {"passage_id": "p2", "raw": ["How should I stack grain bags?", "How do I keep rats out of my store?"]}]


class Compose(unittest.TestCase):
    def run_compose(self, **kw):
        d = pathlib.Path(tempfile.mkdtemp())
        for n, rows in (("p", P), ("c", C), ("q", Q)):
            (d / f"{n}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        out = d / "o.jsonl"
        a = type("A", (), dict(passages=str(d / "p.jsonl"), claims=str(d / "c.jsonl"), questions=str(d / "q.jsonl"), out=str(out), abstain_share=kw.get("abstain", 0.0), refusal_share=0.12))
        b.stage_compose(a)
        return [json.loads(l) for l in out.read_text().splitlines()]

    def test_grounded_answers_only_use_accepted_claims_and_cite_the_right_source(self):
        rows = [r for r in self.run_compose() if r["kind"] == "grounded"]
        self.assertTrue(rows)
        for r in rows:
            ans = r["messages"][-1]["content"]
            self.assertNotIn("invented extra", ans)
            n = int(ans.rsplit("[", 1)[1].rstrip("]"))
            system = r["messages"][0]["content"]
            self.assertIn(f"[{n}] (", system)
            pid = r["passage_id"]
            src_line = [l for l in system.split("\n") if l.startswith(f"[{n}] (")][0]
            self.assertIn({"p1": "Banana", "p2": "Store bags"}[pid], src_line)

    def test_every_grounded_sentence_is_cited(self):
        for r in [r for r in self.run_compose() if r["kind"] == "grounded"]:
            for s in r["messages"][-1]["content"].split(". ["):
                pass
            self.assertTrue(r["messages"][-1]["content"].rstrip().endswith("]"))

    def test_abstain_examples_remove_the_answer_passage(self):
        rows = [r for r in self.run_compose(abstain=1.0) if r["kind"] == "abstain"]
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["messages"][-1]["content"], REPLIES["abstain"])
            self.assertNotIn({"p1": "Banana aphids spread", "p2": "Store bags in tidy"}[r["passage_id"]], r["messages"][0]["content"])

    def test_refusals_use_the_guard_replies(self):
        rows = [r for r in self.run_compose() if r["kind"] == "refusal"]
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["messages"][-1]["content"], REPLIES[r["rule"]])


if __name__ == "__main__":
    unittest.main()
