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
            if r["rule"] == "gold":      # the v16 hand-written refusals are reused as they are
                self.assertEqual(r["messages"][0]["role"], "system"); continue
            self.assertEqual(r["messages"][-1]["content"], REPLIES[r["rule"]])


class Holdout(unittest.TestCase):
    def test_held_out_passages_are_excluded_from_training_and_written_to_the_eval_file(self):
        d = pathlib.Path(tempfile.mkdtemp())
        for n, rows in (("p", P), ("c", C), ("q", Q)):
            (d / f"{n}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        a = type("A", (), dict(passages=str(d / "p.jsonl"), claims=str(d / "c.jsonl"), questions=str(d / "q.jsonl"), out=str(d / "o.jsonl"),
                               abstain_share=0.0, refusal_share=0.12, holdout=1.0, eval_out=str(d / "e.jsonl")))
        b.stage_compose(a)
        train = [json.loads(l) for l in (d / "o.jsonl").read_text().splitlines()]
        ev = [json.loads(l) for l in (d / "e.jsonl").read_text().splitlines()]
        self.assertEqual([r for r in train if r["kind"] != "refusal"], [])          # everything was held out
        self.assertTrue(any(r["kind"] == "grounded" for r in ev) and any(r["kind"] == "abstain" for r in ev))
        g = [r for r in ev if r["kind"] == "grounded"][0]
        self.assertEqual(g["sources"][g["expected_index"] - 1]["passage_id"], g["passage_id"])
        ab = [r for r in ev if r["kind"] == "abstain"][0]
        self.assertNotIn(ab["passage_id"], [s["passage_id"] for s in ab["sources"]])

    def test_no_holdout_by_default(self):
        d = pathlib.Path(tempfile.mkdtemp())
        for n, rows in (("p", P), ("c", C), ("q", Q)):
            (d / f"{n}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        a = type("A", (), dict(passages=str(d / "p.jsonl"), claims=str(d / "c.jsonl"), questions=str(d / "q.jsonl"), out=str(d / "o.jsonl"), abstain_share=0.0, refusal_share=0.12))
        b.stage_compose(a)
        self.assertTrue(any(json.loads(l)["kind"] == "grounded" for l in (d / "o.jsonl").read_text().splitlines()))


class CleanAbstainSources(unittest.TestCase):
    def test_sources_that_answer_are_never_used(self):
        mk = lambda pid, text: {"passage_id": pid, "handle": "h/" + pid, "page": 1, "title": "T", "uri": "u", "licence": "CC-BY-4.0", "topics": [], "text": text}
        answer = "Banana aphids spread banana bunchy top virus by feeding on infected plants and then moving to healthy plants."
        claims = [{"claim": answer}]
        cands = [mk("dup", "Another guide says: " + answer),                                  # a different document that repeats the advice
                 mk("n1", "Banana suckers should be taken from clean nurseries and planted at three metres spacing."),
                 mk("n2", "Banana plants need mulch and regular weeding to keep yields high in the dry season."),
                 mk("n3", "Cassava cuttings sprout best when taken from healthy plants.")]
        got = b.clean_sources(claims, cands)
        self.assertEqual([p["passage_id"] for p in got], ["n1", "n2", "n3"])                 # retrieval order kept, the copy skipped
        self.assertIsNone(b.clean_sources(claims, cands[:3]))                                  # not enough clean sources: no example


class CompactTraining(unittest.TestCase):
    def test_trimmed_examples_stay_supported_and_cite_the_right_index(self):
        import re
        sys.path.insert(0, str(HERE.parent / "eval"))
        from claim_check import check_claim
        long = ("Banana aphids spread banana bunchy top virus by feeding on infected plants and then moving to healthy plants. "
                + "Banana is grown across many regions and is a staple food for millions of households in the tropics. " * 4
                + "Remove infected mats at once and use clean suckers. " + "Spacing of three metres is common for banana mats in the field. " * 4)
        P2 = [dict(P[0], text=long, passage_id="p1")] + [dict(x) for x in P[1:]]
        C2 = [{"passage_id": "p1", "status": "accepted", "claim": "Remove infected mats at once and use clean suckers.", "quote": "Remove infected mats at once and use clean suckers."}]
        Q2 = [{"passage_id": "p1", "raw": ["What do I do about infected banana mats?"]}]
        d = pathlib.Path(tempfile.mkdtemp())
        for n, rows in (("p", P2), ("c", C2), ("q", Q2)):
            (d / f"{n}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        out = d / "o.jsonl"
        a = type("A", (), dict(passages=str(d / "p.jsonl"), claims=str(d / "c.jsonl"), questions=str(d / "q.jsonl"), out=str(out), abstain_share=0.0, refusal_share=0.12, compact="3,25"))
        b.stage_compose(a)
        rows = [json.loads(l) for l in out.read_text().splitlines() if json.loads(l)["kind"] == "grounded"]
        self.assertTrue(rows)
        for r in rows:
            system = r["messages"][0]["content"]
            srcs = re.findall(r"^\[(\d+)\] \(.*?\) (.*)$", system, re.M)
            ans = r["messages"][-1]["content"]
            n = int(re.findall(r"\[(\d+)\]", ans)[-1])
            shown = dict(srcs)[str(n)]
            self.assertLessEqual(len(shown.split()), len(long.split()) // 2)                     # really shorter than the whole passage
            self.assertTrue(check_claim(re.sub(r"\[\d+\]", "", ans).strip().rstrip("."), [shown]).supported)   # still supported by what the model sees


if __name__ == "__main__":
    unittest.main()
