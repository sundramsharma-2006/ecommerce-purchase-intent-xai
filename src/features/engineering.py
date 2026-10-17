"""Behavioural feature engineering (pure function of one frame, so training and serving share it)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED = ["total_pages", "total_duration", "avg_time_per_page", "product_page_share", "time_per_product_page"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ratio features that describe how the visitor browsed.

    All ratios use a safe division (0 when the denominator is 0) so a session with no pages is valid.
    """
    out = df.copy()
    out["total_pages"] = out["Administrative"] + out["Informational"] + out["ProductRelated"]
    out["total_duration"] = out["Administrative_Duration"] + out["Informational_Duration"] + out["ProductRelated_Duration"]
    out["avg_time_per_page"] = np.where(out["total_pages"] > 0, out["total_duration"] / out["total_pages"].clip(lower=1), 0.0)
    out["product_page_share"] = np.where(out["total_pages"] > 0, out["ProductRelated"] / out["total_pages"].clip(lower=1), 0.0)
    out["time_per_product_page"] = np.where(
        out["ProductRelated"] > 0, out["ProductRelated_Duration"] / out["ProductRelated"].clip(lower=1), 0.0
    )
    return out
