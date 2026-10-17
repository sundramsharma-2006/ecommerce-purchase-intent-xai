"""Backend-independent metrics (identical numbers whether or not scikit-learn is installed).

Implemented by hand and unit-tested against hand-computed values, so results do not depend on the
installed library version.
"""
from __future__ import annotations

import numpy as np


def _clean(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y and scores must have the same shape")
    return y, s


def roc_auc(y, s) -> float:
    """ROC-AUC via the rank-sum identity (ties get average ranks)."""
    y, s = _clean(y, s)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    r = np.empty(len(s))
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks[order] = r
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_curve(y, s):
    """Precision/recall at every distinct score threshold (descending)."""
    y, s = _clean(y, s)
    order = np.argsort(-s, kind="mergesort")
    y_s, s_s = y[order], s[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    distinct = np.r_[np.where(np.diff(s_s))[0], len(s_s) - 1]
    tp, fp, thr = tp[distinct], fp[distinct], s_s[distinct]
    precision = tp / (tp + fp)
    recall = tp / max(int(y.sum()), 1)
    return precision, recall, thr


def average_precision(y, s) -> float:
    """PR-AUC as average precision (step-wise, as in scikit-learn)."""
    precision, recall, _ = pr_curve(y, s)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def confusion(y, pred) -> dict[str, int]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    return {
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
        "tp": int(((y == 1) & (pred == 1)).sum()),
    }


def threshold_metrics(y, s, threshold: float) -> dict[str, float]:
    pred = (np.asarray(s) >= threshold).astype(int)
    c = confusion(y, pred)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (c["tp"] + c["tn"]) / max(len(pred), 1)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "flagged_share": float(pred.mean()), **c}


def brier(y, p) -> float:
    y, p = _clean(y, p)
    return float(np.mean((p - y) ** 2))


def full_report(y, s, threshold: float) -> dict[str, float]:
    out = {"roc_auc": roc_auc(y, s), "pr_auc": average_precision(y, s), "brier": brier(y, s)}
    out.update(threshold_metrics(y, s, threshold))
    return out


def bootstrap_ci(y, s, fn, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for any metric fn(y, s)."""
    y, s = _clean(y, s)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        v = fn(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
