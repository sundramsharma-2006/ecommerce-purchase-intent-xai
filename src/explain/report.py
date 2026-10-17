"""Builds the explainability artifacts: global Shapley, local examples, LIME agreement, latency. `python -m src.explain.report`"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import load_settings
from src.data.loader import load_sessions, stratified_split
from src.explain.lime_style import LimeStyleExplainer
from src.explain.shapley import ShapleyExplainer
from src.serving.predictor import PurchasePredictor
from src.utils import plots
from src.utils.logging_utils import get_logger

log = get_logger("explain")


def spearman(a, b) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> dict:
    s = load_settings()
    pred = PurchasePredictor.load(settings=s)
    df = load_sessions(s)
    _, test = stratified_split(df, s)
    feats = pred.features
    ex_cfg = s.section("explain")
    shap = ShapleyExplainer(pred.model.predict_proba, pred.background, feats, ex_cfg["n_permutations"], seed=1)
    sample = test.sample(n=int(ex_cfg["global_sample"]), random_state=s.seed)[feats]
    t0 = time.time()
    glob = shap.global_importance(sample)
    out = {"global_seconds": time.time() - t0, "global": glob.reset_index().rename(columns={"index": "feature"}).to_dict("records")}
    plots.bar_figure(list(glob.index), glob["mean_abs_phi"].tolist(), s.path("figures_dir") / "shapley_global.png",
                     "Global importance: mean |Shapley value| (test sample)", "change in purchase probability")

    metrics = json.loads((s.path("model_dir") / "metrics.json").read_text())
    perm = {r["feature"]: r["mean_drop"] for r in metrics["permutation_importance"] if r["feature"] in feats}
    common = [f for f in feats if f in perm]
    out["rank_agreement_shapley_vs_permutation"] = spearman([glob.loc[f, "mean_abs_phi"] for f in common], [perm[f] for f in common])

    # local examples: highest / lowest scoring + one near the threshold
    probs = pred.model.predict_proba(test)
    idx = {"highest": int(np.argmax(probs)), "near_threshold": int(np.argmin(np.abs(probs - pred.threshold))), "lowest": int(np.argmin(probs))}
    local = {}
    for name, i in idx.items():
        row = test.iloc[i][feats]
        e = shap.explain(row)
        l = LimeStyleExplainer(pred.model.predict_proba, pred.background, feats, n_samples=1500, seed=2).explain(row)
        local[name] = {"actual_converted": int(test.iloc[i]["Converted"]), "probability": float(probs[i]), "base": e["base_value"],
                       "shapley": dict(sorted(e["phi"].items(), key=lambda kv: -abs(kv[1]))[:6]),
                       "lime": dict(sorted(l["weights"].items(), key=lambda kv: -abs(kv[1]))[:6]), "lime_r2": l["r2"],
                       "efficiency_gap": e["efficiency_gap"], "values": {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}}
    out["local"] = local

    # Shapley vs LIME agreement over 25 sessions
    ag, top1, r2s = [], [], []
    for i in range(25):
        row = sample.iloc[i]
        a = shap.explain(row)["phi"]
        b = LimeStyleExplainer(pred.model.predict_proba, pred.background, feats, n_samples=1000, seed=i).explain(row)
        ag.append(spearman([abs(a[f]) for f in feats], [abs(b["weights"][f]) for f in feats]))
        top1.append(max(a, key=lambda k: abs(a[k])) == max(b["weights"], key=lambda k: abs(b["weights"][k])))
        r2s.append(b["r2"])
    out["shapley_vs_lime"] = {"mean_rank_corr": float(np.mean(ag)), "top1_agreement": float(np.mean(top1)),
                              "mean_lime_r2": float(np.mean(r2s)), "n": 25}

    # serving latency
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in test.iloc[0][feats].items()}
    t0 = time.time()
    for _ in range(5):
        pred.predict(row, explain=True)
    out["latency_ms_with_explanation"] = (time.time() - t0) / 5 * 1000
    t0 = time.time()
    for _ in range(20):
        pred.predict(row, explain=False)
    out["latency_ms_score_only"] = (time.time() - t0) / 20 * 1000
    (s.path("reports_dir") / "explain_results.json").write_text(json.dumps(out, indent=2, default=float))
    log.info("done: %s", {k: out[k] for k in ("rank_agreement_shapley_vs_permutation", "shapley_vs_lime", "latency_ms_with_explanation", "latency_ms_score_only")})
    return out


if __name__ == "__main__":
    main()
