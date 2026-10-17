"""Minimal SMOTE (applied inside each CV fold on the training part only)."""
from __future__ import annotations

import numpy as np


def smote(X: np.ndarray, y: np.ndarray, rng: np.random.RandomState, k: int = 5, target_ratio: float = 1.0):
    """Oversample the minority class with synthetic points between a sample and one of its k nearest neighbours.

    target_ratio = desired minority/majority ratio after resampling (1.0 = balanced).
    Note: one-hot columns are interpolated too, so synthetic rows can carry fractional category weights.
    """
    y = np.asarray(y).astype(int)
    minority = X[y == 1]
    n_maj, n_min = int((y == 0).sum()), len(minority)
    n_new = int(round(target_ratio * n_maj)) - n_min
    if n_new <= 0 or n_min < 2:
        return X, y
    k = min(k, n_min - 1)
    d2 = (minority ** 2).sum(1)[:, None] + (minority ** 2).sum(1)[None, :] - 2 * minority @ minority.T
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    base = rng.randint(0, n_min, n_new)
    neigh = nn[base, rng.randint(0, k, n_new)]
    gap = rng.rand(n_new, 1)
    synth = minority[base] + gap * (minority[neigh] - minority[base])
    return np.vstack([X, synth]), np.r_[y, np.ones(n_new, dtype=int)]
