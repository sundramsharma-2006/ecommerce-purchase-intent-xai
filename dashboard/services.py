"""Data/logic layer for the dashboard (no Streamlit import -> unit-testable). Never trains a model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.data.loader import load_sessions
from src.features.engineering import add_engineered_features
from src.models.evaluation import value_curve
from src.serving.predictor import PurchasePredictor, load_metrics


def sessions(settings: Settings | None = None) -> pd.DataFrame:
    return load_sessions(settings or load_settings())


def kpis(df: pd.DataFrame) -> dict:
    return {"sessions": int(len(df)), "purchases": int(df["Converted"].sum()), "conversion_rate": float(df["Converted"].mean()),
            "avg_product_pages": float(df["ProductRelated"].mean())}


def funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Engagement funnel: how conversion changes as sessions go deeper."""
    d = add_engineered_features(df)
    stages = [("All sessions", d), ("Viewed >= 1 product page", d[d["ProductRelated"] >= 1]),
              ("Viewed >= 10 product pages", d[d["ProductRelated"] >= 10]),
              ("Viewed >= 25 product pages", d[d["ProductRelated"] >= 25]),
              ("... and ExitRates below median", d[(d["ProductRelated"] >= 25) & (d["ExitRates"] < d["ExitRates"].median())])]
    return pd.DataFrame([{"stage": n, "sessions": len(s), "conversion_rate": float(s["Converted"].mean()) if len(s) else np.nan,
                          "purchases": int(s["Converted"].sum())} for n, s in stages])


def cohort(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by)["Converted"].agg(sessions="size", conversion_rate="mean").reset_index()
    return g.sort_values("conversion_rate", ascending=False)


def month_order(df: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    order = settings.cols("months")
    g = cohort(df, "Month")
    g["Month"] = pd.Categorical(g["Month"], categories=order, ordered=True)
    return g.sort_values("Month")


def sensitivity(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics["threshold_sensitivity"])


def value_table(y, p, value: float, cost: float) -> pd.DataFrame:
    return value_curve(y, p, value, cost, 99)


def load_predictor() -> PurchasePredictor:
    return PurchasePredictor.load()


def metrics_summary() -> dict:
    return load_metrics()
