"""LIME-style local surrogate explanation (second, differently-constructed lens; PRD 4.6).

Real `lime` was not installable in the build sandbox.  This follows LIME's recipe with one difference that
should be stated plainly: the interpretable representation is "feature keeps this session's value (1) or is
replaced by a random background value (0)" instead of LIME's quantile-discretised bins.
Perturbed sessions are weighted by closeness to the original (exponential kernel on the share of removed
features) and a weighted ridge regression of predicted probability on the 0/1 mask gives the local weights.
`r2` reports how well that linear surrogate fits the model locally (low r2 = treat the explanation with care).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


class LimeStyleExplainer:
    def __init__(self, predict_fn: Callable[[pd.DataFrame], np.ndarray], background: pd.DataFrame, features: list[str],
                 n_samples: int = 1000, kernel_width: float = 0.35, alpha: float = 1.0, seed: int = 0):
        self.predict_fn, self.features = predict_fn, list(features)
        self.background = background.reset_index(drop=True)
        self.n_samples, self.kernel_width, self.alpha, self.seed = n_samples, kernel_width, alpha, seed

    def explain(self, x: pd.Series | dict) -> dict:
        x = pd.Series(x)
        d = len(self.features)
        rng = np.random.RandomState(self.seed)
        mask = (rng.rand(self.n_samples, d) < 0.5).astype(float)
        mask[0] = 1.0                                              # the original session itself
        bg_idx = rng.randint(0, len(self.background), self.n_samples)
        cols = {}
        for j, f in enumerate(self.features):
            cols[f] = np.where(mask[:, j] == 1, x[f], self.background[f].to_numpy()[bg_idx])
        frame = pd.DataFrame(cols)
        for c in self.background.columns:
            if c not in frame.columns:
                frame[c] = self.background[c].to_numpy()[bg_idx]
        y = np.asarray(self.predict_fn(frame[self.background.columns]), dtype=float)
        dist = (1 - mask).mean(axis=1)                              # share of features removed
        w = np.exp(-(dist ** 2) / (self.kernel_width ** 2))
        A = np.c_[np.ones(len(mask)), mask]
        W = np.diag(w) if self.n_samples <= 2000 else None
        reg = np.eye(d + 1) * self.alpha
        reg[0, 0] = 0
        Aw = A * w[:, None]
        beta = np.linalg.solve(A.T @ Aw + reg, Aw.T @ y)
        fit = A @ beta
        ybar = np.average(y, weights=w)
        r2 = 1 - np.sum(w * (y - fit) ** 2) / max(np.sum(w * (y - ybar) ** 2), 1e-12)
        return {"weights": dict(zip(self.features, beta[1:].tolist())), "intercept": float(beta[0]), "r2": float(r2),
                "prediction": float(y[0])}
