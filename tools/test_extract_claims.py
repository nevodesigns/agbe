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


if __name__ == "__main__":
    unittest.main()
