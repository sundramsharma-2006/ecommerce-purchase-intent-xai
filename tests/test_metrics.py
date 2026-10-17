import unittest

import numpy as np

from tests import helpers  # noqa: F401
from src.utils import metrics as M


class MetricsTest(unittest.TestCase):
    def test_roc_auc_hand_computed(self):
        # positives {0.8, 0.4}, negatives {0.6, 0.2}: pairs correctly ordered = 3 of 4
        self.assertAlmostEqual(M.roc_auc([1, 0, 1, 0], [0.8, 0.6, 0.4, 0.2]), 0.75)

    def test_roc_auc_perfect_and_reversed_and_ties(self):
        self.assertEqual(M.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(M.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertAlmostEqual(M.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_single_class_gives_nan(self):
        self.assertTrue(np.isnan(M.roc_auc([1, 1, 1], [0.1, 0.2, 0.3])))

    def test_average_precision_hand_computed(self):
        # ranking by score: 1,0,1,0 -> precision at positives: 1/1 and 2/3 -> AP = (1 + 2/3)/2
        self.assertAlmostEqual(M.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]), (1 + 2 / 3) / 2)

    def test_confusion_and_threshold_metrics(self):
        y = [1, 1, 0, 0, 0, 1]
        s = [0.9, 0.2, 0.6, 0.1, 0.3, 0.7]
        r = M.threshold_metrics(y, s, 0.5)
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(r["precision"], 2 / 3)
        self.assertAlmostEqual(r["recall"], 2 / 3)
        self.assertAlmostEqual(r["accuracy"], 4 / 6)

    def test_majority_baseline_accuracy_is_misleading(self):
        y = np.r_[np.ones(157), np.zeros(843)]
        r = M.threshold_metrics(y, np.zeros(1000), 0.5)
        self.assertAlmostEqual(r["accuracy"], 0.843)
        self.assertEqual(r["recall"], 0.0)

    def test_brier(self):
        self.assertAlmostEqual(M.brier([1, 0], [0.8, 0.3]), (0.04 + 0.09) / 2)

    def test_bootstrap_ci_contains_estimate(self):
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 400)
        s = y * 0.3 + rng.rand(400)
        lo, hi = M.bootstrap_ci(y, s, M.roc_auc, 200)
        self.assertLess(lo, M.roc_auc(y, s))
        self.assertGreater(hi, M.roc_auc(y, s))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            M.roc_auc([1, 0], [0.1])


if __name__ == "__main__":
    unittest.main()
