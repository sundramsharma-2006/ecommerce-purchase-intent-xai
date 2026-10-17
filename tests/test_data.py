import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests import helpers  # noqa: F401
from src.config import load_settings
from src.data.loader import DataValidationError, load_sessions, stratified_split, validate_sessions

S = load_settings()


class LoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_sessions(S)

    def test_loads_expected_shape(self):
        self.assertEqual(self.df.shape, (12000, 18))
        self.assertAlmostEqual(self.df["Converted"].mean(), 0.1572, places=3)

    def test_missing_file_fails_loudly(self):
        with self.assertRaises(FileNotFoundError):
            load_sessions(S, path="does_not_exist.csv")

    def test_missing_column_rejected(self):
        with self.assertRaises(DataValidationError):
            validate_sessions(self.df.drop(columns=["ExitRates"]), S)

    def test_bad_values_rejected(self):
        bad = self.df.copy(); bad.loc[0, "Month"] = "Smarch"
        with self.assertRaises(DataValidationError):
            validate_sessions(bad, S)
        bad = self.df.copy(); bad.loc[0, "ExitRates"] = 1.5
        with self.assertRaises(DataValidationError):
            validate_sessions(bad, S)
        bad = self.df.copy(); bad.loc[0, "Converted"] = 2
        with self.assertRaises(DataValidationError):
            validate_sessions(bad, S)
        bad = self.df.copy(); bad.loc[0, "ProductRelated_Duration"] = None
        with self.assertRaises(DataValidationError):
            validate_sessions(bad, S)

    def test_malformed_csv_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text("a,b\n1,2\n")
            with self.assertRaises(DataValidationError):
                load_sessions(S, path=p)

    def test_split_is_stratified_deterministic_and_disjoint(self):
        a1, b1 = stratified_split(self.df, S)
        a2, b2 = stratified_split(self.df, S)
        pd.testing.assert_frame_equal(a1, a2); pd.testing.assert_frame_equal(b1, b2)
        self.assertEqual(len(a1) + len(b1), len(self.df))
        self.assertAlmostEqual(a1.Converted.mean(), b1.Converted.mean(), places=3)

    def test_known_data_properties(self):
        # documented in the README: no purchase ever happens when PageValues == 0
        self.assertEqual(self.df.loc[self.df.PageValues == 0, "Converted"].sum(), 0)
        self.assertEqual(self.df.duplicated().sum(), 0)


if __name__ == "__main__":
    unittest.main()
