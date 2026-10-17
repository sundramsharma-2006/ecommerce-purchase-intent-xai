"""Model-agnostic Shapley values at the level of the ORIGINAL session features (PRD 4.6).

Why not the `shap` package here? It was not installable in the build sandbox.  This is the standard
permutation-sampling estimator of Shapley values (Strumbelj & Kononenko, 2014): for random feature orderings,
start from a random background session and switch features to the explained session's values one at a time;
the change in predicted probability is credited to the feature switched.  Properties verified in tests/:
  * efficiency  - contributions sum exactly to  f(x) - mean f(background rows used)
  * exactness   - equals the closed form for an additive model
Players are the raw input columns (e.g. ProductRelated), not one-hot columns, so explanations read naturally.
Values are in PROBABILITY units (percentage points of purchase probability).
If `shap` is installed, `shap_tree_values()` below can be used for tree models as an independent check.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


class ShapleyExplainer:
    def __init__(self, predict_fn: Callable[[pd.DataFrame], np.ndarray], background: pd.DataFrame, features: list[str],
                 n_permutations: int = 40, seed: int = 0):
        if len(background) == 0:
            raise ValueError("background must not be empty")
        self.predict_fn = predict_fn
        self.background = background.reset_index(drop=True)
        self.features = list(features)
        self.n_perm = int(n_permutations)
        self.seed = seed

    def _perturbed(self, x: pd.Series, orders: np.ndarray, bg_idx: np.ndarray) -> pd.DataFrame:
        d = len(self.features)
        n = len(orders)
        # rank[m, j] = position of feature j in permutation m
        rank = np.empty_like(orders)
        for m in range(n):
            rank[m, orders[m]] = np.arange(d)
        steps = np.arange(d + 1)[None, :, None]                       # k = number of switched features
        use_x = rank[:, None, :] < steps                              # (n, d+1, d)
        cols = {}
        bg = self.background
        for j, f in enumerate(self.features):
            xv = x[f]
            zv = bg[f].to_numpy()[bg_idx][:, None]                    # (n, 1)
            cols[f] = np.where(use_x[:, :, j], xv, zv).reshape(-1)
        out = pd.DataFrame(cols)
        for c in self.background.columns:                             # any non-player columns keep background values
            if c not in out.columns:
                out[c] = np.repeat(bg[c].to_numpy()[bg_idx], d + 1)
        return out[self.background.columns]

    def explain(self, x: pd.Series | dict) -> dict:
        x = pd.Series(x)
        d = len(self.features)
        rng = np.random.RandomState(self.seed)
        half = max(self.n_perm // 2, 1)
        base_orders = np.array([rng.permutation(d) for _ in range(half)])
        orders = np.vstack([base_orders, base_orders[:, ::-1]])       # antithetic pairs reduce variance
        # cycle through a shuffled background so every background row is used evenly (lower variance than iid draws)
        bg_cycle = rng.permutation(len(self.background))
        bg_idx = np.tile(bg_cycle[np.arange(half) % len(bg_cycle)], 2)
        frame = self._perturbed(x, orders, bg_idx)
        preds = np.asarray(self.predict_fn(frame)).reshape(len(orders), d + 1)
        diffs = preds[:, 1:] - preds[:, :-1]                          # (n, d): credit for the j-th switched feature
        phi = np.zeros(d)
        for m in range(len(orders)):
            phi[orders[m]] += diffs[m]
        phi /= len(orders)
        base = float(preds[:, 0].mean())
        fx = float(preds[:, -1].mean())
        return {"phi": dict(zip(self.features, phi.tolist())), "base_value": base, "prediction": fx,
                "efficiency_gap": float(phi.sum() - (fx - base))}

    def explain_many(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = [self.explain(frame.iloc[i])["phi"] for i in range(len(frame))]
        return pd.DataFrame(rows, index=frame.index)

    def global_importance(self, sample: pd.DataFrame) -> pd.DataFrame:
        """Mean |Shapley value| per feature over a sample (the usual SHAP 'global bar chart')."""
        phi = self.explain_many(sample)
        out = pd.DataFrame({"mean_abs_phi": phi.abs().mean(), "mean_phi": phi.mean()})
        return out.sort_values("mean_abs_phi", ascending=False)


def shap_tree_values(model, X):  # pragma: no cover - requires the shap package
    """Optional cross-check with the real SHAP TreeExplainer when `shap` is installed."""
    import shap

    return shap.TreeExplainer(model).shap_values(X)
