import unittest
from serve import prompt

LONG = ("Maize is planted at the onset of the rains. " * 3 + "Space rows 75 cm apart and plants 25 cm apart within the row. " +
        "Storage bags should be kept on pallets away from walls. " * 3)
P1 = {"title": "A", "page": 1, "text": LONG, "passage_id": "a"}
P2 = {"title": "B", "page": 2, "text": "Cassava stems are cut into pieces of 25 cm for planting.", "passage_id": "b"}
P3 = {"title": "C", "page": 3, "text": "Goats need clean water every day and shelter from rain.", "passage_id": "c"}


class T(unittest.TestCase):
    def test_compact_keeps_the_relevant_sentence_and_shortens(self):
        out = prompt.compact("how far apart should I space maize rows", LONG, 25)
        self.assertIn("75 cm", out)
        self.assertLess(len(out.split()), len(LONG.split()))
        self.assertLessEqual(len(out.split()), 25)

    def test_short_passage_unchanged(self):
        self.assertEqual(prompt.compact("goats water", P3["text"], 80), P3["text"])

    def test_prepare_ranks_and_keeps_full_text_for_the_guard(self):
        shown, full = prompt.prepare_sources("how far apart should I space maize rows", [P3, P2, P1], k=2, max_words=25)
        self.assertEqual(len(shown), 2)
        self.assertEqual(shown[0]["passage_id"], "a")                       # the maize passage ranks first
        self.assertLess(len(shown[0]["text"].split()), len(full[0]["text"].split()))
        self.assertEqual(full[0]["text"], LONG)                              # the guard still sees the whole passage

    def test_advisor_prompts_with_trimmed_text_but_checks_against_full(self):
        from serve.guard import GuardedAdvisor, build_messages
        seen = {}
        class R:
            def top(self, q, k, min_score): return [P1, P2, P3]
        def gen(messages):
            seen["prompt"] = messages[0]["content"]
            return "Space rows 75 cm apart and plants 25 cm apart within the row. [1]"
        adv = GuardedAdvisor(R(), gen, compact=(2, 25)).advise("how far apart should I space maize rows")
        full = build_messages("how far apart should I space maize rows", [P1, P2, P3])[0]["content"]
        self.assertLess(len(seen["prompt"].split()), len(full.split()))                          # shorter than the full three-passage prompt
        self.assertEqual(adv.status, "answered")                                                  # still supported by the full passage


if __name__ == "__main__":
    unittest.main()
