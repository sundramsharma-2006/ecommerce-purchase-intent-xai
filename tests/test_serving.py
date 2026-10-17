import unittest

import numpy as np

from tests import helpers  # noqa: F401
from src.config import load_settings
from src.data.loader import load_sessions, stratified_split
from src.serving.predictor import ModelNotFoundError, PurchasePredictor

S = load_settings()


class PredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = PurchasePredictor.load(settings=S)
        _, cls.test = stratified_split(load_sessions(S), S)
        cls.row = {k: (v.item() if hasattr(v, "item") else v) for k, v in cls.test.iloc[0][cls.pred.features].items()}

    def test_response_contract(self):
        r = self.pred.predict(self.row)
        self.assertTrue(0 <= r["probability"] <= 1)
        self.assertIn(r["decision"], {"high_intent", "low_intent"})
        self.assertIn(r["confidence"], {"high", "medium", "low"})
        self.assertEqual(r["prediction"], int(r["probability"] >= r["threshold"]))
        self.assertLessEqual(len(r["top_contributors"]), 5)
        self.assertTrue({"feature", "value", "effect", "direction"} <= set(r["top_contributors"][0]))

    def test_no_train_serve_skew(self):
        """Serving a row gives exactly what the training-time model gives for that row."""
        direct = self.pred.model.predict_proba(self.test.iloc[[0]])[0]
        self.assertAlmostEqual(self.pred.predict(self.row, explain=False)["probability"], direct, places=12)

    def test_explanation_adds_up(self):
        r = self.pred.predict(self.row)
        full = self.pred._explainer.explain(self.test.iloc[0][self.pred.features])
        self.assertAlmostEqual(full["efficiency_gap"], 0.0, places=9)
        self.assertAlmostEqual(r["baseline_probability"], full["base_value"], places=9)

    def test_unseen_category_does_not_crash(self):
        r = self.pred.predict({**self.row, "Browser": 999, "TrafficType": 777}, explain=False)
        self.assertTrue(0 <= r["probability"] <= 1)

    def test_missing_field_raises(self):
        bad = dict(self.row); bad.pop("ExitRates")
        with self.assertRaises(KeyError):
            self.pred.predict(bad)

    def test_batch_matches_single(self):
        batch = self.pred.predict_proba(self.test.iloc[:5][self.pred.features])
        single = [self.pred.predict(
            {k: (v.item() if hasattr(v, "item") else v) for k, v in self.test.iloc[i][self.pred.features].items()},
            explain=False)["probability"] for i in range(5)]
        np.testing.assert_allclose(batch, single)

    def test_higher_engagement_raises_probability(self):
        lo = self.pred.predict({**self.row, "ProductRelated": 1, "ProductRelated_Duration": 20.0}, explain=False)["probability"]
        hi = self.pred.predict({**self.row, "ProductRelated": 60, "ProductRelated_Duration": 2500.0}, explain=False)["probability"]
        self.assertGreater(hi, lo)

    def test_confidence_levels(self):
        t = self.pred.threshold
        self.assertEqual(self.pred.confidence(t), "low")
        self.assertEqual(self.pred.confidence(min(t + 0.5, 1.0)), "high")

    def test_missing_model_dir_fails_clearly(self):
        with self.assertRaises(ModelNotFoundError):
            PurchasePredictor.load(model_dir="/nonexistent_dir_xyz", settings=S)


if __name__ == "__main__":
    unittest.main()
