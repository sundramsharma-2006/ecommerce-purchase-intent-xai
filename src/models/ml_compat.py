"""One import point for estimators: real libraries when installed, NumPy fallback otherwise.

BACKEND records what actually ran so every report can state it honestly.
Boosters (XGBoost / LightGBM / CatBoost) are optional: they are registered only when importable.
"""
from __future__ import annotations

try:  # real scikit-learn
    from sklearn.ensemble import GradientBoostingClassifier as _SKGB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    import sklearn

    BACKEND = f"scikit-learn {sklearn.__version__}"

    class GradientBoostingClassifier(_SKGB):
        """scikit-learn's GB has no class_weight; emulate it with sample weights."""

        def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=1, class_weight=None, random_state=None):
            super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                             min_samples_leaf=min_samples_leaf, random_state=random_state)
            self.class_weight = class_weight

        def fit(self, X, y, sample_weight=None):
            if self.class_weight == "balanced":
                sample_weight = compute_sample_weight("balanced", y)
            return super().fit(X, y, sample_weight=sample_weight)

except ImportError:  # offline fallback
    from src.models.mini_sklearn import (  # noqa: F401
        DecisionTreeClassifier, GradientBoostingClassifier, LogisticRegression, RandomForestClassifier,
    )

    BACKEND = "mini_sklearn (NumPy fallback implementing the scikit-learn API subset)"


def optional_boosters() -> dict[str, object]:
    """Return {name: importable module} for the boosters that are installed."""
    found = {}
    for name, mod in (("XGBoost", "xgboost"), ("LightGBM", "lightgbm"), ("CatBoost", "catboost")):
        try:
            found[name] = __import__(mod)
        except Exception:  # noqa: BLE001 - any import problem means "not available"
            continue
    return found
