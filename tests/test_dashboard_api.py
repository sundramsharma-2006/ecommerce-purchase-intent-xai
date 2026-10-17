import unittest

from tests import helpers  # noqa: F401
from dashboard import services as D
from src.config import load_settings

S = load_settings()


class DashboardServicesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = D.sessions(S)

    def test_kpis(self):
        k = D.kpis(self.df)
        self.assertEqual(k["sessions"], 12000)
        self.assertEqual(k["purchases"], 1886)

    def test_funnel_is_monotone_in_size_and_engagement_lifts_conversion(self):
        f = D.funnel(self.df)
        self.assertTrue((f["sessions"].diff().dropna() <= 0).all())
        self.assertGreater(f["conversion_rate"].iloc[3], f["conversion_rate"].iloc[0])

    def test_month_order_is_calendar(self):
        g = D.month_order(self.df, S)
        months = list(g["Month"].astype(str))
        self.assertEqual(months, [m for m in S.cols("months") if m in months])

    def test_cohort_counts_sum_to_total(self):
        self.assertEqual(int(D.cohort(self.df, "VisitorType")["sessions"].sum()), 12000)


class ApiSchemaTest(unittest.TestCase):
    def test_schema_validation_if_pydantic_installed(self):
        try:
            from pydantic import ValidationError

            from src.api.schemas import SessionIn
        except ImportError:
            self.skipTest("pydantic not installed in this environment")
        ok = dict(SessionIn.model_config["json_schema_extra"]["example"])
        SessionIn(**ok)
        with self.assertRaises(ValidationError):
            SessionIn(**{**ok, "ExitRates": 3})
        with self.assertRaises(ValidationError):
            SessionIn(**{**ok, "Month": "Smarch"})
        with self.assertRaises(ValidationError):
            SessionIn(**{**ok, "PageValues": 10})    # extra fields (PageValues) are rejected


if __name__ == "__main__":
    unittest.main()
