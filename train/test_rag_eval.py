import pathlib, sys, unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rag_eval
from rag_eval import score, summarise

SRC = [{"text": "Banana aphids spread banana bunchy top virus by feeding on infected plants and then moving to healthy plants. Remove infected mats at once and use clean suckers."},
       {"text": "Store bags in tidy stacks on pallets with a space of one metre all round the stack."}]
G = {"kind": "grounded", "sources": SRC, "expected_index": 1, "question": "q"}
A = {"kind": "abstain", "sources": SRC, "expected_index": None, "question": "q"}


class Score(unittest.TestCase):
    def test_grounded_supported_and_cited_passes(self):
        r = score(G, "Banana aphids spread banana bunchy top virus by feeding on infected plants and then moving to healthy plants [1]. Remove infected mats at once and use clean suckers [1].")
        self.assertTrue(r["pass"] and r["right_source"] and r["valid_citation"])

    def test_grounded_invented_fails(self):
        r = score(G, "Spray a residue test product every five minutes until the leaf turns dark brown [1]. This always cures the plant completely within two days [2].")
        self.assertFalse(r["pass"])

    def test_grounded_without_citation_fails(self):
        r = score(G, "Banana aphids spread banana bunchy top virus by feeding on infected plants and then moving to healthy plants.")
        self.assertFalse(r["pass"])

    def test_grounded_wrongly_abstaining_fails(self):
        self.assertFalse(score(G, "I do not have a source for that, so I will not guess.")["pass"])

    def test_abstain_case(self):
        self.assertTrue(score(A, "I do not have a source for that, so I will not guess. Please ask your extension officer.")["pass"])
        self.assertFalse(score(A, "Plant cassava cuttings of 30 cm with six nodes in rows two metres apart, and weed them twice before the rains end.")["pass"])

    def test_summary_shape(self):
        rows = [dict(kind="grounded", **score(G, "Remove infected mats at once and use clean suckers [1].")), dict(kind="abstain", **score(A, "I do not have a source for that."))]
        s = summarise(rows)
        self.assertEqual(s["grounded_n"], 1); self.assertEqual(s["abstain_pass"], 1.0)


class Leaky(unittest.TestCase):
    def test_leaky_abstain_cases_are_flagged_and_reported_apart(self):
        import json, tempfile, os
        answer = "Banana aphids spread banana bunchy top virus by feeding on infected plants and then moving to healthy plants."
        d = tempfile.mkdtemp(); path = os.path.join(d, "c.jsonl")
        open(path, "w").write(json.dumps({"passage_id": "p", "status": "accepted", "claim": answer}) + "\n")
        mk = lambda text: [{"title": "T", "page": 1, "text": text, "passage_id": "x", "uri": "u", "licence": "CC-BY-4.0"}]
        recs = [{"kind": "abstain", "question": "q1", "sources": mk("A guide: " + answer), "passage_id": "p", "expected_index": None, "claims": []},
                {"kind": "abstain", "question": "q2", "sources": mk("Mulch keeps soil moist in the dry season."), "passage_id": "p", "expected_index": None, "claims": []}]
        rows = []
        for r in rag_eval.mark_leaky(recs, path):
            rows.append({"kind": r["kind"], "question": r["question"], "answer": "I do not have a source for that.", **rag_eval.score(r, "I do not have a source for that.")})
        self.assertEqual([r["leaky"] for r in rows], [True, False])
        s = rag_eval.summarise(rows)
        self.assertEqual(s["abstain_n"], 2); self.assertEqual(s["abstain_clean_n"], 1); self.assertEqual(s["abstain_clean_pass"], 1.0)


if __name__ == "__main__":
    unittest.main()
