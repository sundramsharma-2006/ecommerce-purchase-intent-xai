import unittest

import numpy as np
import pandas as pd

from tests import helpers  # noqa: F401
from src.explain.lime_style import LimeStyleExplainer
from src.explain.shapley import ShapleyExplainer


def additive(df):
    return 2 * df["a"].to_numpy() + 3 * df["b"].to_numpy() ** 2 - df["c"].to_numpy()


BG = pd.DataFrame({"a": [0., 1., 2., 3.], "b": [0., 1., 0., 1.], "c": [5., 1., 2., 0.]})
X = pd.Series({"a": 4., "b": 2., "c": 3.})


class ShapleyTest(unittest.TestCase):
    def test_efficiency_exact(self):
        r = ShapleyExplainer(additive, BG, ["a", "b", "c"], n_permutations=60, seed=0).explain(X)
        self.assertAlmostEqual(r["efficiency_gap"], 0.0, places=9)
        self.assertAlmostEqual(sum(r["phi"].values()), r["prediction"] - r["base_value"], places=9)

    def test_exact_for_additive_model_with_one_background_row(self):
        bg = BG.iloc[[1]]
        r = ShapleyExplainer(additive, bg, ["a", "b", "c"], n_permutations=10, seed=0).explain(X)
        self.assertAlmostEqual(r["phi"]["a"], 2 * (4 - 1))
        self.assertAlmostEqual(r["phi"]["b"], 3 * (4 - 1))
        self.assertAlmostEqual(r["phi"]["c"], -(3 - 1))

    def test_close_to_closed_form_with_full_background(self):
        r = ShapleyExplainer(additive, BG, ["a", "b", "c"], n_permutations=800, seed=1).explain(X)
        exact = {"a": 2 * (4 - BG.a.mean()), "b": 3 * (4 - (BG.b ** 2).mean()), "c": -(3 - BG.c.mean())}
        for k in exact:
            self.assertAlmostEqual(r["phi"][k], exact[k], delta=0.2)

    def test_irrelevant_feature_gets_zero(self):
        bg = BG.assign(d=[1., 2., 3., 4.])
        r = ShapleyExplainer(additive, bg, ["a", "b", "c", "d"], n_permutations=40, seed=0).explain(X.append(pd.Series({"d": 9.})) if hasattr(X, "append") else pd.concat([X, pd.Series({"d": 9.})]))
        self.assertAlmostEqual(r["phi"]["d"], 0.0, places=9)

    def test_empty_background_rejected(self):
        with self.assertRaises(ValueError):
            ShapleyExplainer(additive, BG.iloc[:0], ["a", "b", "c"])


class LimeTest(unittest.TestCase):
    def test_recovers_signs_of_a_linear_model(self):
        lin = lambda df: 0.5 * df["a"].to_numpy() - 0.3 * df["c"].to_numpy() + 0.0 * df["b"].to_numpy()  # noqa: E731
        r = LimeStyleExplainer(lin, BG, ["a", "b", "c"], n_samples=2000, seed=0).explain(X)
        self.assertGreater(r["weights"]["a"], 0)
        self.assertLess(r["weights"]["c"], 0)
        self.assertLess(abs(r["weights"]["b"]), 0.05)


if __name__ == "__main__":
    unittest.main()
