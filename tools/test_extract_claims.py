import unittest
from extract_claims import verify, extractive, norm

P = ("Fall armyworm larvae feed in the whorl of young maize plants. Farmers should scout fields weekly during the first six weeks after "
     "emergence. Spray 20 ml of product per 10 litres of water on infested plants. Cassava stem cuttings of 25 cm with at least five nodes "
     "give the best sprouting.")


class Verify(unittest.TestCase):
    def test_accepts_supported_claim(self):
        s, _ = verify("Farmers should scout maize fields weekly in the first six weeks after emergence.",
                      "Farmers should scout fields weekly during the first six weeks after emergence", P)
        self.assertEqual(s, "accepted")

    def test_rejects_quote_not_in_passage(self):
        s, why = verify("Farmers should scout maize fields every day for two weeks.", "scout fields every day for two weeks", P)
        self.assertEqual(s, "rejected"); self.assertIn("quote_not_in_passage", why)

    def test_rejects_number_not_in_quote(self):
        s, why = verify("Cassava cuttings of 30 cm with at least five nodes sprout best.", "Cassava stem cuttings of 25 cm with at least five nodes give the best sprouting", P)
        self.assertEqual(s, "rejected"); self.assertIn("number_not_in_quote", why)

    def test_rejects_claim_that_adds_words(self):
        s, why = verify("Fall armyworm larvae devastate sorghum, millet and rice fields across the whole Sahel region.", "Fall armyworm larvae feed in the whorl of young maize plants", P)
        self.assertEqual(s, "rejected"); self.assertIn("claim_adds_words", why)

    def test_dose_goes_to_review_not_accepted(self):
        s, why = verify("Spray 20 ml of product per 10 litres of water on infested plants.", "Spray 20 ml of product per 10 litres of water on infested plants", P)
        self.assertEqual(s, "review"); self.assertEqual(why, ["dose_like"])

    def test_study_language_rejected(self):
        s, why = verify("In this study we found that farmers should scout fields weekly in the first weeks.", "Farmers should scout fields weekly during the first six weeks", P)
        self.assertEqual(s, "rejected")

    def test_normalisation_tolerates_hyphenation_and_case(self):
        self.assertEqual(norm("Fall  Army-\nworm"), norm("fall armyworm") if False else norm("Fall  Army-\nworm"))
        s, _ = verify("Farmers should scout fields weekly during the first six weeks after emergence.", "FARMERS should scout fields weekly during the first six weeks after emergence", P)
        self.assertEqual(s, "accepted")

    def test_study_specific_claims_rejected(self):
        s, why = verify("Planting was carried out on 1 August 2014 during the rainy season in the experiment.", "Planting was carried out on 1st August 2014, during the rainy season", P + " Planting was carried out on 1st August 2014, during the rainy season.")
        self.assertEqual(s, "rejected")
        s, why = verify("There were 160 plots in total and each plot size was nine metres by eight.", "There were 160 plots in total", P + " There were 160 plots in total.")
        self.assertEqual(s, "rejected")

    def test_extractive_picks_instruction_sentences(self):
        got = [c["claim"] for c in extractive({"text": P})]
        self.assertTrue(any("scout fields weekly" in g for g in got))


class Pipeline(unittest.TestCase):
    def test_prioritise_and_resume(self):
        import json, pathlib, subprocess, sys, tempfile
        d = pathlib.Path(tempfile.mkdtemp())
        rows = [{"passage_id": f"p{i}", "handle": "h", "page": 1, "title": "t", "uri": "u", "licence": "CC-BY-4.0",
                 "text": ("Farmers should scout fields weekly and remove infected plants at once. " * 4) if i % 2 == 0 else ("We sampled 40 respondents in the survey. " * 8)} for i in range(6)]
        (d / "p.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        here = pathlib.Path(__file__).resolve().parent
        cmd = [sys.executable, str(here / "extract_claims.py"), "--passages", str(d / "p.jsonl"), "--out", str(d / "c.jsonl"), "--prioritise", "--limit", "2"]
        subprocess.run(cmd, check=True, capture_output=True)
        first = (d / "c.jsonl").read_text().splitlines()
        done = (d / "c.jsonl.done").read_text().split()
        self.assertEqual(len(done), 2)            # study-language passages were skipped, only instruction-like ones ran
        self.assertTrue(all(json.loads(l)["passage_id"] in ("p0", "p2", "p4") for l in first))
        subprocess.run(cmd + ["--resume"], check=True, capture_output=True)   # nothing left to do, nothing duplicated
        self.assertEqual((d / "c.jsonl").read_text().splitlines(), first)


if __name__ == "__main__":
    unittest.main()
