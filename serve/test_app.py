import pathlib, sys, unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from serve.app import create_app
from serve.guard import GuardedAdvisor
from serve.test_guard import retriever


def client(keys=None):
    adv = GuardedAdvisor(retriever(), lambda m: "Fall armyworm larvae feed in the whorl of young maize plants [1]. Invented extra claim about quantum fertiliser rays.")
    return TestClient(create_app(adv, keys if keys is not None else set()))


class Api(unittest.TestCase):
    def test_health(self):
        self.assertTrue(client().get("/health").json()["ok"])

    def test_advise_trims_and_cites(self):
        r = client().post("/advise", json={"question": "What eats the whorl of my maize and leaves sawdust?"}).json()
        self.assertEqual(r["status"], "trimmed"); self.assertNotIn("quantum", r["answer"]); self.assertEqual(r["citations"][0]["passage_id"], "p1")

    def test_dose_refused(self):
        r = client().post("/advise", json={"question": "How many ml of insecticide per litre for armyworm?"}).json()
        self.assertEqual(r["status"], "refused")

    def test_openai_shape(self):
        r = client().post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "What eats the whorl of my maize and leaves sawdust?"}]}).json()
        self.assertEqual(r["choices"][0]["message"]["role"], "assistant"); self.assertIn("agbe", r)

    def test_auth(self):
        c = client({"k1"})
        self.assertEqual(c.post("/advise", json={"question": "x y z"}).status_code, 401)
        self.assertEqual(c.post("/advise", json={"question": "What eats the whorl of my maize and leaves sawdust?"}, headers={"Authorization": "Bearer k1"}).status_code, 200)

    def test_rejects_empty_and_huge(self):
        c = client()
        self.assertEqual(c.post("/advise", json={"question": "  "}).status_code, 422)
        self.assertEqual(c.post("/advise", json={"question": "a" * 700}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
