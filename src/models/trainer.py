"""Model registry + a single trainer interface (PRD 4.5).

Every algorithm is wrapped in the same `SessionClassifier` (preprocessor + imbalance strategy + estimator), so
cross-validation clones the *whole* pipeline and nothing is fitted on held-out rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.config import Settings
from src.models import ml_compat as mc
from src.models.smote import smote
from src.preprocessing.pipeline import SessionPreprocessor, build_preprocessor
from src.features.engineering import add_engineered_features  # noqa: F401  (re-exported for callers)

IMBALANCE = ("none", "class_weight", "smote")


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[str, int, dict], object]       # (imbalance, seed, params) -> estimator
    default_params: dict
    grid: list[dict]                                   # small tuning grid
    native: str = "mini"                               # which backend provides it


def _lr(imb, seed, p):
    return mc.LogisticRegression(C=p.get("C", 1.0), class_weight="balanced" if imb == "class_weight" else None, max_iter=200)


def _dt(imb, seed, p):
    return mc.DecisionTreeClassifier(max_depth=p.get("max_depth", 6), min_samples_leaf=p.get("min_samples_leaf", 20),
                                     class_weight="balanced" if imb == "class_weight" else None, random_state=seed)


def _rf(imb, seed, p):
    return mc.RandomForestClassifier(n_estimators=p.get("n_estimators", 150), max_depth=p.get("max_depth", 10),
                                     min_samples_leaf=p.get("min_samples_leaf", 10), max_features="sqrt",
                                     class_weight="balanced" if imb == "class_weight" else None, random_state=seed)


def _gb(imb, seed, p):
    return mc.GradientBoostingClassifier(n_estimators=p.get("n_estimators", 150), learning_rate=p.get("learning_rate", 0.08),
                                         max_depth=p.get("max_depth", 3), min_samples_leaf=p.get("min_samples_leaf", 20),
                                         class_weight="balanced" if imb == "class_weight" else None, random_state=seed)


def _xgb(imb, seed, p):  # pragma: no cover - only when xgboost is installed
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.05),
                         max_depth=p.get("max_depth", 4), subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=5.4 if imb == "class_weight" else 1.0, eval_metric="aucpr",
                         random_state=seed, n_jobs=-1, verbosity=0)


def _lgbm(imb, seed, p):  # pragma: no cover
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.05),
                          num_leaves=p.get("num_leaves", 15), subsample=0.8, colsample_bytree=0.8,
                          class_weight="balanced" if imb == "class_weight" else None, random_state=seed, verbose=-1)


def _cat(imb, seed, p):  # pragma: no cover
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=p.get("iterations", 300), learning_rate=p.get("learning_rate", 0.05),
                              depth=p.get("depth", 5), auto_class_weights="Balanced" if imb == "class_weight" else None,
                              random_seed=seed, verbose=0)


def build_registry() -> dict[str, ModelSpec]:
    """Always: LogisticRegression, DecisionTree, RandomForest, GradientBoosting.
    Added only if installed: XGBoost, LightGBM, CatBoost."""
    reg = {
        "LogisticRegression": ModelSpec("LogisticRegression", _lr, {"C": 1.0}, [{"C": 0.03}, {"C": 0.3}, {"C": 3.0}]),
        "DecisionTree": ModelSpec("DecisionTree", _dt, {"max_depth": 6}, [{"max_depth": 4}, {"max_depth": 8, "min_samples_leaf": 30}]),
        "RandomForest": ModelSpec("RandomForest", _rf, {"max_depth": 10},
                                  [{"max_depth": 8, "min_samples_leaf": 20}, {"max_depth": 12, "min_samples_leaf": 10}]),
        "GradientBoosting": ModelSpec("GradientBoosting", _gb, {"max_depth": 3},
                                      [{"max_depth": 2, "n_estimators": 200}, {"max_depth": 3, "learning_rate": 0.05}, {"max_depth": 4, "n_estimators": 100}]),
    }
    boosters = mc.optional_boosters()
    if "XGBoost" in boosters:
        reg["XGBoost"] = ModelSpec("XGBoost", _xgb, {}, [{"max_depth": 3}, {"max_depth": 5}], native="xgboost")
    if "LightGBM" in boosters:
        reg["LightGBM"] = ModelSpec("LightGBM", _lgbm, {}, [{"num_leaves": 7}, {"num_leaves": 31}], native="lightgbm")
    if "CatBoost" in boosters:
        reg["CatBoost"] = ModelSpec("CatBoost", _cat, {}, [{"depth": 4}, {"depth": 6}], native="catboost")
    return reg


class SessionClassifier:
    """preprocessor -> (optional SMOTE) -> estimator. fit() learns everything from the rows it is given."""

    def __init__(self, spec: ModelSpec, settings: Settings, imbalance: str = "class_weight", params: dict | None = None,
                 use_page_values: bool | None = None, exclude=()):
        if imbalance not in IMBALANCE:
            raise ValueError(f"imbalance must be one of {IMBALANCE}")
        self.spec, self.settings, self.imbalance = spec, settings, imbalance
        self.params = {**spec.default_params, **(params or {})}
        self.use_page_values, self.exclude = use_page_values, tuple(exclude)
        self.pre: SessionPreprocessor | None = None
        self.est = None

    def fit(self, df: pd.DataFrame, y) -> "SessionClassifier":
        y = np.asarray(y).astype(int)
        self.pre = build_preprocessor(self.settings, self.use_page_values, self.exclude)
        X = self.pre.fit_transform(df)
        if self.imbalance == "smote":
            X, y = smote(X, y, np.random.RandomState(self.settings.seed))
        self.est = self.spec.factory(self.imbalance, self.settings.seed, self.params)
        self.est.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probability of the positive (purchase) class."""
        if self.est is None or self.pre is None:
            raise RuntimeError("Model is not fitted")
        return np.asarray(self.est.predict_proba(self.pre.transform(df)))[:, 1]

    def feature_importance(self) -> dict[str, float] | None:
        est = self.est
        if hasattr(est, "feature_importances_"):
            vals = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            vals = np.abs(np.asarray(est.coef_, dtype=float).ravel())
        else:
            return None
        agg: dict[str, float] = {}
        for n, v in zip(self.pre.feature_names_, vals):
            g = self.pre.group_of(n)
            agg[g] = agg.get(g, 0.0) + float(v)
        s = sum(agg.values()) or 1.0
        return dict(sorted(((k, v / s) for k, v in agg.items()), key=lambda kv: -kv[1]))
