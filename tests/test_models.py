import unittest

import numpy as np
import pandas as pd

from tests import helpers  # noqa: F401
from src.config import load_settings
from src.data.loader import load_sessions, stratified_split
from src.features.engineering import add_engineered_features
from src.models.evaluation import choose_threshold, cross_validate, stratified_folds, value_curve
from src.models.smote import smote
from src.models.trainer import SessionClassifier, build_registry
from src.preprocessing.pipeline import build_preprocessor
from src.utils import metrics as M

S = load_settings()
DF = load_sessions(S)
TRAIN, TEST = stratified_split(DF, S)


class FeatureTest(unittest.TestCase):
    def test_engineered_features_and_safe_division(self):
        d = pd.DataFrame([{"Administrative": 0, "Informational": 0, "ProductRelated": 0, "Administrative_Duration": 0.0,
                           "Informational_Duration": 0.0, "ProductRelated_Duration": 0.0},
                          {"Administrative": 1, "Informational": 1, "ProductRelated": 2, "Administrative_Duration": 10.0,
                           "Informational_Duration": 20.0, "ProductRelated_Duration": 70.0}])
        e = add_engineered_features(d)
        self.assertEqual(e.loc[0, "avg_time_per_page"], 0.0)          # no pages -> no division error
        self.assertEqual(e.loc[1, "total_pages"], 4)
        self.assertAlmostEqual(e.loc[1, "avg_time_per_page"], 25.0)
        self.assertAlmostEqual(e.loc[1, "product_page_share"], 0.5)
        self.assertAlmostEqual(e.loc[1, "time_per_product_page"], 35.0)


class PreprocessorTest(unittest.TestCase):
    def test_fit_uses_only_given_rows_and_is_deterministic(self):
        a = build_preprocessor(S).fit(TRAIN.iloc[:2000])
        b = build_preprocessor(S).fit(TRAIN.iloc[:2000])
        np.testing.assert_allclose(a.transform(TEST.iloc[:50]), b.transform(TEST.iloc[:50]))
        c = build_preprocessor(S).fit(TRAIN.iloc[2000:4000])
        self.assertFalse(np.allclose(a.mean_, c.mean_))                 # statistics come from the fit rows

    def test_int_coded_ids_are_one_hot_not_scaled(self):
        p = build_preprocessor(S).fit(TRAIN)
        for col in S.cols("int_coded_categorical"):
            self.assertTrue(any(n.startswith(f"{col}=") for n in p.feature_names_), col)
            self.assertFalse(any(n == f"log1p({col})" or n == col for n in p.feature_names_), col)

    def test_unseen_category_becomes_all_zeros(self):
        p = build_preprocessor(S).fit(TRAIN)
        row = TEST.iloc[[0]].copy(); row["Browser"] = 999
        X = p.transform(row)
        idx = [i for i, n in enumerate(p.feature_names_) if n.startswith("Browser=")]
        self.assertEqual(X[0, idx].sum(), 0.0)

    def test_page_values_excluded_by_default_and_optional(self):
        self.assertNotIn("PageValues", build_preprocessor(S).input_columns)
        self.assertIn("PageValues", build_preprocessor(S, use_page_values=True).input_columns)

    def test_missing_input_column_raises(self):
        p = build_preprocessor(S).fit(TRAIN)
        with self.assertRaises(KeyError):
            p.transform(TEST.drop(columns=["ExitRates"]).iloc[:5])

    def test_exclude_removes_features(self):
        p = build_preprocessor(S, exclude=("Region", "Month")).fit(TRAIN)
        self.assertFalse(any(n.startswith(("Region=", "Month=")) for n in p.feature_names_))


class SmoteTest(unittest.TestCase):
    def test_balances_and_keeps_originals(self):
        rng = np.random.RandomState(0)
        X = rng.randn(300, 4); y = np.r_[np.ones(30), np.zeros(270)].astype(int)
        X2, y2 = smote(X, y, np.random.RandomState(1))
        self.assertEqual((y2 == 1).sum(), (y2 == 0).sum())
        np.testing.assert_array_equal(X2[:300], X)
        lo, hi = X[y == 1].min(0), X[y == 1].max(0)
        self.assertTrue(((X2[300:] >= lo - 1e-9) & (X2[300:] <= hi + 1e-9)).all())   # synthetic points lie inside the minority hull box


class CvTest(unittest.TestCase):
    def test_folds_are_stratified_and_disjoint(self):
        y = TRAIN.Converted.to_numpy()
        seen = np.zeros(len(y))
        for tr, va in stratified_folds(y, 5, 42):
            self.assertAlmostEqual(y[va].mean(), y.mean(), places=2)
            self.assertEqual(len(np.intersect1d(tr, va)), 0)
            seen[va] += 1
        self.assertTrue((seen == 1).all())

    def test_cv_is_deterministic_and_beats_random(self):
        spec = build_registry()["LogisticRegression"]
        sub = TRAIN.iloc[:3000].reset_index(drop=True)
        a = cross_validate(spec, S, sub, sub.Converted, "none", folds=3)
        b = cross_validate(spec, S, sub, sub.Converted, "none", folds=3)
        np.testing.assert_allclose(a["oof"], b["oof"])
        self.assertGreater(a["roc_auc_mean"], 0.7)
        self.assertGreater(a["pr_auc_mean"], sub.Converted.mean() * 2)

    def test_registry_always_has_four_core_models(self):
        self.assertTrue({"LogisticRegression", "DecisionTree", "RandomForest", "GradientBoosting"} <= set(build_registry()))

    def test_classifier_rejects_unknown_strategy_and_unfitted_predict(self):
        spec = build_registry()["LogisticRegression"]
        with self.assertRaises(ValueError):
            SessionClassifier(spec, S, "magic")
        with self.assertRaises(RuntimeError):
            SessionClassifier(spec, S).predict_proba(TEST.iloc[:2])


class ThresholdTest(unittest.TestCase):
    def test_value_curve_matches_hand_computation(self):
        y = np.array([1, 0, 1, 0]); p = np.array([0.9, 0.8, 0.3, 0.1])
        vc = value_curve(y, p, value=10, cost=1, grid=99)
        row = vc.iloc[(vc.threshold - 0.5).abs().argmin()]          # flags the two highest scores: 1 TP, 1 FP
        self.assertEqual((row.tp, row.flagged), (1, 2))
        self.assertAlmostEqual(row.net_value, 10 * 1 - 1 * 2)

    def test_threshold_tracks_cost_ratio_and_is_not_default_half(self):
        rng = np.random.RandomState(3)
        y = (rng.rand(4000) < 0.16).astype(int)
        p = np.clip(0.16 + 0.35 * (y - 0.16) + 0.15 * rng.randn(4000), 0.001, 0.999)
        cheap = choose_threshold(y, p, 40, 2)["threshold"]
        dear = choose_threshold(y, p, 40, 20)["threshold"]
        self.assertLess(cheap, dear)
        self.assertNotAlmostEqual(cheap, 0.5, places=1)


class EndToEndTest(unittest.TestCase):
    def test_model_beats_single_feature_baseline_not_by_magic(self):
        spec = build_registry()["LogisticRegression"]
        m = SessionClassifier(spec, S, "none", {"C": 0.03}).fit(TRAIN, TRAIN.Converted)
        p = m.predict_proba(TEST)
        self.assertGreater(M.roc_auc(TEST.Converted, p), 0.80)
        self.assertTrue(((p >= 0) & (p <= 1)).all())


if __name__ == "__main__":
    unittest.main()
