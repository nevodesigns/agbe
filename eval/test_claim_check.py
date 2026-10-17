import unittest
from claim_check import check_answer, check_claim, split_claims

SRC = ["Fall armyworm larvae feed in the whorl of young maize and leave sawdust-like frass. Scout fields weekly in the first six weeks.",
       "Cassava mosaic disease has no cure. Use clean stem cuttings from plants that show no symptoms."]


class ClaimCheck(unittest.TestCase):
    def test_supported_claim(self):
        self.assertTrue(check_claim("Fall armyworm larvae feed in the whorl of young maize.", SRC).supported)

    def test_unsupported_claim(self):
        self.assertFalse(check_claim("Spray a residue test product every five minutes.", SRC).supported)

    def test_number_must_match(self):
        self.assertFalse(check_claim("Scout fields weekly in the first ten weeks.", SRC).supported)
        self.assertTrue(check_claim("Scout fields weekly in the first six weeks.", SRC).supported)

    def test_tail_drift_is_caught(self):
        ans = ("I cannot answer that, and I would be making it up if I did.\n\n"
               "The residue is the part you see on the leaf after the spray has stopped. "
               "If the residue is still dark brown the pest was not in the field.")
        r = check_answer(ans, SRC)
        self.assertTrue(r.refusal)
        self.assertGreaterEqual(len(r.unsupported), 1)
        self.assertFalse(r.stops_after_refusal)

    def test_clean_refusal_stops(self):
        r = check_answer("I cannot advise that. I am an agricultural advisor and do not prescribe medicine.\n\nPlease take her to a clinic today.", SRC)
        self.assertTrue(r.refusal and r.stops_after_refusal)

    def test_questions_are_not_claims(self):
        self.assertEqual(split_claims("What crop is it? Please tell me the crop."), ["Please tell me the crop."])


if __name__ == "__main__":
    unittest.main()
