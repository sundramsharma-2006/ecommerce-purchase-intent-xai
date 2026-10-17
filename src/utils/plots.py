"""Shared matplotlib figures (static, saved to reports/figures)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils import metrics as M  # noqa: E402

PALETTE = {"main": "#1F3A5F", "accent": "#E07A1F", "grey": "#8A94A6", "good": "#2E8B57", "bad": "#C0392B"}


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def pr_roc_figure(y, scores: dict[str, np.ndarray], path: Path, title: str = "") -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, s in scores.items():
        pr, rc, _ = M.pr_curve(y, s)
        ax[0].plot(rc, pr, label=f"{name} (AP {M.average_precision(y, s):.3f})")
        order = np.argsort(-s)
        ys = np.asarray(y)[order]
        tpr = np.cumsum(ys) / ys.sum()
        fpr = np.cumsum(1 - ys) / (1 - ys).sum()
        ax[1].plot(fpr, tpr, label=f"{name} (AUC {M.roc_auc(y, s):.3f})")
    ax[0].axhline(np.mean(y), ls="--", c=PALETTE["grey"], label=f"base rate {np.mean(y):.3f}")
    ax[1].plot([0, 1], [0, 1], ls="--", c=PALETTE["grey"])
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.25)
    fig.suptitle(title)
    return save(fig, path)


def calibration_figure(y, p, path: Path, bins: int = 10) -> Path:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    xs = [p[idx == b].mean() for b in range(bins) if (idx == b).any()]
    ys = [y[idx == b].mean() for b in range(bins) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", c=PALETTE["grey"], label="perfect")
    ax.plot(xs, ys, "o-", c=PALETTE["main"], label="model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed purchase rate", title="Calibration (test set)")
    ax.legend()
    ax.grid(alpha=0.25)
    return save(fig, path)


def bar_figure(labels, values, path: Path, title: str, xlabel: str, color=None) -> Path:
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(labels) + 1.2))
    ax.barh(np.array(labels)[order], np.array(values)[order], color=color or PALETTE["main"])
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="x", alpha=0.25)
    return save(fig, path)
