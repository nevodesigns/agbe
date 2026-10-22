import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard_eval

SRC = [{"title": "T", "page": 1, "text": "Plant maize at the onset of the rains. Space rows 75 cm apart.", "passage_id": "p", "uri": "u", "licence": "CC-BY-4.0"}]
REC_G = {"kind": "grounded", "question": "when to plant maize", "sources": SRC, "expected_index": 1, "claims": [], "passage_id": "p"}
REC_A = {"kind": "abstain", "question": "how to graft mango", "sources": SRC, "expected_index": None, "claims": [], "passage_id": "q"}


class T(unittest.TestCase):
    def test_guard_removes_invented_sentence_and_abstains_on_nothing_left(self):
        answers = ["Plant maize at the onset of the rains. [1] Mulch with banana leaves after sowing every week. [1]",
                   "Graft mango in March using a sharp knife and wax tape for the join. [1]"]
        rows, changed = guard_eval.apply_guard([REC_G, REC_A], answers)
        self.assertEqual(changed, 2)
        self.assertNotIn("banana", rows[0]["answer"]); self.assertTrue(rows[0]["pass"])   # kept the supported sentence, dropped the invented one
        self.assertTrue(rows[1]["abstained"]); self.assertTrue(rows[1]["pass"])           # nothing supported, so the fixed abstain reply

    def test_dose_is_removed(self):
        rows, _ = guard_eval.apply_guard([REC_G], ["Plant maize at the onset of the rains. [1] Apply 5 ml of insecticide per litre. [1]"])
        self.assertNotIn("5 ml", rows[0]["answer"])


if __name__ == "__main__":
    unittest.main()
