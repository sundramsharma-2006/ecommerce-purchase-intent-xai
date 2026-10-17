"""Architecture and workflow diagrams (matplotlib). python tools/make_diagrams.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs"


def box(ax, x, y, w, h, text, fc="#E8EEF6", ec="#1F3A5F", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#1F3A5F", lw=1.3))


def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.9)); ax.set_xlim(0, 11); ax.set_ylim(0, 6.9); ax.axis("off")
    layers = [("DATA", 5.0, "#E8EEF6", ["ecommerce_sessions.csv", "Loader + validation", "Stratified split (80/20)", "Preprocessor\n(log1p, scale, one-hot)"]),
              ("ML", 3.55, "#FFF1E0", ["Feature engineering", "4-6 models\n+ stratified CV", "Value-based\nthreshold", "Shapley +\nLIME-style", "Model card"]),
              ("SERVING", 2.1, "#E6F4EA", ["PurchasePredictor\n(loads bundle once)", "FastAPI  /predict /health", "Streamlit dashboard"]),
              ("MLOPS", 0.65, "#F3E8F9", ["Docker + Compose", "MLflow / JSONL run log", "GitHub Actions CI", "AWS ECR/ECS notes"])]
    for name, y, col, items in layers:
        ax.add_patch(FancyBboxPatch((0.2, y), 10.6, 1.2, boxstyle="round,pad=0.02,rounding_size=0.1", fc=col, ec="#8A94A6", lw=1))
        ax.text(0.45, y + 0.6, name, fontsize=11, fontweight="bold", va="center", color="#1F3A5F")
        n = len(items); w = 8.9 / n
        for i, t in enumerate(items):
            box(ax, 1.7 + i * w, y + 0.2, w - 0.15, 0.8, t, fc="white", fs=8)
    for y in (5.0, 3.55, 2.1):
        arrow(ax, 5.5, y, 5.5, y - 0.25)
    ax.text(5.5, 6.65, "Purchase-intent system: four layers (config + logging cut across all)", ha="center", fontsize=12, fontweight="bold", color="#1F3A5F")
    fig.savefig(OUT / "architecture.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(12, 3.4)); ax.set_xlim(0, 12); ax.set_ylim(0, 3.4); ax.axis("off")
    steps = ["CSV row /\nsession JSON", "Validate\n(schema, ranges)", "Engineer + encode\n(saved preprocessor)", "Model\nP(purchase)", "Threshold\n(value-based)", "Shapley\ncontributors", "Response:\nprob, decision,\nconfidence, top 5"]
    w = 1.5
    for i, s in enumerate(steps):
        x = 0.2 + i * 1.68
        box(ax, x, 1.1, w, 1.2, s, fc="#E8EEF6" if i not in (3, 6) else "#FFF1E0", fs=8.5)
        if i:
            arrow(ax, x - 0.18, 1.7, x, 1.7)
    ax.text(6, 3.0, "Scoring workflow: one session, from input to explained decision", ha="center", fontsize=12, fontweight="bold", color="#1F3A5F")
    ax.text(6, 0.45, "Same fitted preprocessor object in training and serving (no train/serve skew); unseen categories -> all-zero one-hot.", ha="center", fontsize=9, color="#555")
    fig.savefig(OUT / "workflow.png", dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    architecture(); workflow(); print("diagrams written")
