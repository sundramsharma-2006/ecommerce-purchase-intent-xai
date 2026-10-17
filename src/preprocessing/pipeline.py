"""Reusable, fit-once preprocessing (PRD 4.2): the same fitted object is used in training, API and dashboard.

Equivalent in behaviour to a scikit-learn ColumnTransformer inside a Pipeline:
  * skewed counts/durations      -> log1p -> standardise
  * bounded rates / ratios       -> standardise
  * categoricals (incl. the int-coded IDs OperatingSystems/Browser/Region/TrafficType) -> one-hot,
    categories learned from the *training* fold only, unseen categories -> all zeros (handle_unknown='ignore')
It is fitted inside every cross-validation fold by the trainer, never on the full data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Settings
from src.features.engineering import ENGINEERED, add_engineered_features

LOG_COLS_BASE = [
    "Administrative", "Informational", "ProductRelated",
    "Administrative_Duration", "Informational_Duration", "ProductRelated_Duration",
    "total_pages", "total_duration", "avg_time_per_page", "time_per_product_page",
]
PLAIN_COLS_BASE = ["BounceRates", "ExitRates", "SpecialDay", "product_page_share"]


@dataclass
class SessionPreprocessor:
    categorical: list[str]
    use_page_values: bool = False
    exclude: tuple[str, ...] = ()          # raw/engineered columns removed by feature selection
    log_cols: list[str] = field(default_factory=list)
    plain_cols: list[str] = field(default_factory=list)
    levels_: dict[str, list[str]] = field(default_factory=dict)
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    feature_names_: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        ex = set(self.exclude)
        self.log_cols = [c for c in LOG_COLS_BASE if c not in ex]
        self.plain_cols = [c for c in PLAIN_COLS_BASE if c not in ex]
        if self.use_page_values and "PageValues" not in ex:
            self.log_cols.append("PageValues")
        self.categorical = [c for c in self.categorical if c not in ex]

    # ------------------------------------------------------------------ helpers
    @property
    def input_columns(self) -> list[str]:
        """Raw columns the caller must provide (engineered ones are derived)."""
        raw_needed = {
            "Administrative", "Informational", "ProductRelated", "Administrative_Duration",
            "Informational_Duration", "ProductRelated_Duration",
        }
        raw_needed |= {c for c in self.log_cols + self.plain_cols if c not in ENGINEERED}
        raw_needed |= set(self.categorical)
        return sorted(raw_needed)

    @staticmethod
    def _as_str(s: pd.Series) -> pd.Series:
        return s.astype(str)

    def _numeric(self, df: pd.DataFrame) -> np.ndarray:
        d = add_engineered_features(df)
        parts = [np.log1p(d[self.log_cols].to_numpy(dtype=float)), d[self.plain_cols].to_numpy(dtype=float)]
        return np.hstack(parts)

    # ------------------------------------------------------------------ API
    def fit(self, df: pd.DataFrame) -> "SessionPreprocessor":
        if len(df) == 0:
            raise ValueError("Cannot fit the preprocessor on an empty frame")
        num = self._numeric(df)
        self.mean_ = num.mean(axis=0)
        self.std_ = num.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        self.levels_ = {c: sorted(self._as_str(df[c]).unique().tolist()) for c in self.categorical}
        self.feature_names_ = (
            [f"log1p({c})" for c in self.log_cols] + list(self.plain_cols)
            + [f"{c}={lvl}" for c in self.categorical for lvl in self.levels_[c]]
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("SessionPreprocessor is not fitted")
        missing = [c for c in self.input_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing input columns: {missing}")
        num = (self._numeric(df) - self.mean_) / self.std_
        blocks = [num]
        for c in self.categorical:
            col = self._as_str(df[c]).to_numpy()
            for lvl in self.levels_[c]:
                blocks.append((col == lvl).astype(float)[:, None])
        return np.hstack(blocks)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def group_of(self, feature_name: str) -> str:
        """Map an expanded column name back to the original feature (for explanations)."""
        if "=" in feature_name:
            return feature_name.split("=")[0]
        if feature_name.startswith("log1p("):
            return feature_name[6:-1]
        return feature_name


def build_preprocessor(settings: Settings, use_page_values: bool | None = None, exclude=()) -> SessionPreprocessor:
    upv = (not settings.drop_page_values) if use_page_values is None else use_page_values
    return SessionPreprocessor(categorical=settings.cols("categorical"), use_page_values=upv, exclude=tuple(exclude))
