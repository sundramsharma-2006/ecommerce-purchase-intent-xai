"""Stratified cross-validation, tuning and value-based threshold selection."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import Settings
from src.models.trainer import ModelSpec, SessionClassifier
from src.utils import metrics as M


def stratified_folds(y, k: int, seed: int):
    """Yield (train_idx, val_idx) with the class ratio preserved in every fold."""
    y = np.asarray(y).astype(int)
    rng = np.random.RandomState(seed)
    fold_of = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % k
    for f in range(k):
        yield np.where(fold_of != f)[0], np.where(fold_of == f)[0]


def cross_validate(spec: ModelSpec, settings: Settings, df: pd.DataFrame, y, imbalance="class_weight", params=None,
                   use_page_values=None, exclude=(), folds: int | None = None, quiet=True) -> dict:
    """Return pooled out-of-fold probabilities plus per-fold ROC-AUC / PR-AUC and fit time."""
    y = np.asarray(y).astype(int)
    k = folds or settings.cv_folds
    oof = np.zeros(len(y))
    per_fold = []
    t0 = time.time()
    for tr, va in stratified_folds(y, k, settings.seed):
        clf = SessionClassifier(spec, settings, imbalance, params, use_page_values, exclude).fit(df.iloc[tr], y[tr])
        p = clf.predict_proba(df.iloc[va])
        oof[va] = p
        per_fold.append({"roc_auc": M.roc_auc(y[va], p), "pr_auc": M.average_precision(y[va], p)})
    return {
        "oof": oof,
        "roc_auc_mean": float(np.mean([f["roc_auc"] for f in per_fold])),
        "roc_auc_std": float(np.std([f["roc_auc"] for f in per_fold])),
        "pr_auc_mean": float(np.mean([f["pr_auc"] for f in per_fold])),
        "pr_auc_std": float(np.std([f["pr_auc"] for f in per_fold])),
        "per_fold": per_fold,
        "seconds": time.time() - t0,
    }


def tune(spec: ModelSpec, settings: Settings, df: pd.DataFrame, y, imbalance="class_weight", use_page_values=None,
         exclude=(), folds: int = 3) -> tuple[dict, list[dict]]:
    """Pick the grid entry with the best mean PR-AUC (3-fold, training data only)."""
    results = []
    for params in [{}] + spec.grid:
        cv = cross_validate(spec, settings, df, y, imbalance, params, use_page_values, exclude, folds=folds)
        results.append({"params": params, "pr_auc": cv["pr_auc_mean"], "roc_auc": cv["roc_auc_mean"]})
    best = max(results, key=lambda r: r["pr_auc"])
    return best["params"], results


# ---------------------------------------------------------------------------- threshold
def value_curve(y, p, value: float, cost: float, grid: int = 199) -> pd.DataFrame:
    """Expected net value of flagging every session with p >= t.

    net = value * (captured purchases among flagged) - cost * (all flagged sessions)
    This is an UPPER BOUND on real value: it assumes the intervention captures every flagged would-be buyer.
    value / cost are assumptions supplied by the business (config.yaml), not measured here.
    """
    y = np.asarray(y).astype(int)
    rows = []
    for t in np.linspace(0.005, 0.995, grid):
        flagged = p >= t
        tp = int((flagged & (y == 1)).sum())
        n_flag = int(flagged.sum())
        rows.append({"threshold": float(t), "flagged": n_flag, "tp": tp,
                     "precision": tp / n_flag if n_flag else 0.0, "recall": tp / max(int(y.sum()), 1),
                     "net_value": value * tp - cost * n_flag})
    return pd.DataFrame(rows)


def choose_threshold(y, p, value: float, cost: float, grid: int = 199) -> dict:
    curve = value_curve(y, p, value, cost, grid)
    best = curve.loc[curve["net_value"].idxmax()]
    return {"threshold": float(best["threshold"]), "net_value": float(best["net_value"]),
            "precision": float(best["precision"]), "recall": float(best["recall"]), "flagged": int(best["flagged"])}
