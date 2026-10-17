"""Serving-side scoring: loads the saved bundle ONCE and scores single sessions or batches (PRD 4.7).

The same SessionClassifier (fitted preprocessor + estimator) object that training saved is used here, so there
is no train/serve skew.  The FastAPI app and the Streamlit dashboard are thin wrappers around this class.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.explain.shapley import ShapleyExplainer
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when no trained bundle exists (run `python -m src.train` first)."""


class PurchasePredictor:
    def __init__(self, bundle: dict, settings: Settings):
        self.settings = settings
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.meta = bundle["metadata"]
        self.background: pd.DataFrame = bundle["background"]
        self.features: list[str] = self.model.pre.input_columns
        self._explainer = ShapleyExplainer(self.model.predict_proba, self.background, self.features,
                                           n_permutations=settings.section("explain")["n_permutations"], seed=0)

    @classmethod
    def load(cls, model_dir: str | Path | None = None, settings: Settings | None = None) -> "PurchasePredictor":
        settings = settings or load_settings()
        model_dir = Path(model_dir) if model_dir else settings.path("model_dir")
        path = model_dir / "model_bundle.pkl"
        if not path.exists():
            raise ModelNotFoundError(f"No trained model at {path}. Run: python -m src.train")
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)  # trusted local artifact produced by src.train
        log.info("Loaded model %s (threshold %.3f)", bundle["metadata"].get("model"), bundle["threshold"])
        return cls(bundle, settings)

    # ------------------------------------------------------------------ scoring
    def _frame(self, sessions: list[dict] | pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(sessions) if not isinstance(sessions, pd.DataFrame) else sessions.copy()
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise KeyError(f"Missing fields: {missing}")
        return df

    def predict_proba(self, sessions) -> np.ndarray:
        return self.model.predict_proba(self._frame(sessions))

    def confidence(self, p: float) -> str:
        margin = abs(p - self.threshold)
        hi = float(self.settings.section("serving")["high_confidence_margin"])
        return "high" if margin >= hi else "medium" if margin >= hi / 2 else "low"

    def predict(self, session: dict, explain: bool = True) -> dict:
        df = self._frame([session])
        p = float(self.model.predict_proba(df)[0])
        out = {
            "probability": p,
            "prediction": int(p >= self.threshold),
            "decision": "high_intent" if p >= self.threshold else "low_intent",
            "threshold": self.threshold,
            "confidence": self.confidence(p),
            "model": self.meta.get("model"),
        }
        if explain:
            ex = self._explainer.explain(df.iloc[0])
            k = int(self.settings.section("serving")["top_k_contributors"])
            top = sorted(ex["phi"].items(), key=lambda kv: -abs(kv[1]))[:k]
            out["top_contributors"] = [
                {"feature": f, "value": _plain(df.iloc[0][f]), "effect": float(v),
                 "direction": "raises" if v > 0 else "lowers"} for f, v in top
            ]
            out["baseline_probability"] = ex["base_value"]
        return out


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def load_metrics(settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    p = settings.path("model_dir") / "metrics.json"
    if not p.exists():
        raise ModelNotFoundError(f"No metrics at {p}. Run: python -m src.train")
    return json.loads(p.read_text())
