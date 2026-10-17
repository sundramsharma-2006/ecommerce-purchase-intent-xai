"""End-to-end training pipeline:  python -m src.train

Stages: load -> stratified split -> imbalance benchmark (CV) -> tuning (CV) -> model comparison (CV) ->
feature selection (permutation importance + ablation) -> value-based threshold (on out-of-fold predictions) ->
ONE evaluation on the untouched test set -> save versioned artifacts + metrics.

The test set is used for the final champion (and a clearly-labelled PageValues reference run) only.
Everything that chooses something (strategy, hyper-parameters, features, threshold) uses training data only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.data.loader import load_sessions, stratified_split
from src.models import ml_compat
from src.models.evaluation import choose_threshold, cross_validate, stratified_folds, tune, value_curve
from src.models.trainer import IMBALANCE, SessionClassifier, build_registry
from src.utils import metrics as M
from src.utils import plots
from src.utils.logging_utils import get_logger
from src.utils.tracking import log_run

log = get_logger("train")

_T0 = time.time()
BUDGET_S = float(__import__("os").environ.get("TRAIN_BUDGET_S", "0"))   # 0 = unlimited; >0 lets long runs resume in slices


class BudgetExceeded(RuntimeError):
    """Raised between work units when the time slice is used up; re-run the same command to resume."""


def unit(settings: Settings, key: str, fn):
    """Run fn() once and cache the result on disk (experiments/cache); re-runs resume from the cache."""
    d = settings.path("experiments_dir") / "cache"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{key}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    if BUDGET_S and time.time() - _T0 > BUDGET_S:
        raise BudgetExceeded(key)
    val = fn()
    with open(f, "wb") as fh:
        pickle.dump(val, fh)
    return val


def _json(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def permutation_importance(spec, settings, df, y, params, imbalance, exclude, repeats=3, folds=5):
    """Per-fold drop in PR-AUC when one RAW input column is shuffled in the validation rows."""
    rng = np.random.RandomState(settings.seed)
    feats = None
    drops: dict[str, list[float]] = {}
    for tr, va in stratified_folds(y, folds, settings.seed):
        clf = SessionClassifier(spec, settings, imbalance, params, None, exclude).fit(df.iloc[tr], y[tr])
        feats = clf.pre.input_columns
        dv, yv = df.iloc[va].reset_index(drop=True), y[va]
        base = M.average_precision(yv, clf.predict_proba(dv))
        for f in feats:
            vals = []
            for _ in range(repeats):
                d2 = dv.copy()
                d2[f] = rng.permutation(d2[f].to_numpy())
                vals.append(base - M.average_precision(yv, clf.predict_proba(d2)))
            drops.setdefault(f, []).append(float(np.mean(vals)))
    rows = [{"feature": f, "mean_drop": float(np.mean(v)), "std_drop": float(np.std(v)),
             "folds_positive": int(sum(x > 0 for x in v))} for f, v in drops.items()]
    return pd.DataFrame(rows).sort_values("mean_drop", ascending=False).reset_index(drop=True)


def run(settings: Settings | None = None, quick: bool = False) -> dict:
    t_start = time.time()
    settings = settings or load_settings()
    np.random.seed(settings.seed)
    df = load_sessions(settings)
    train, test = stratified_split(df, settings)
    y_tr, y_te = train[settings.target].to_numpy(), test[settings.target].to_numpy()
    reg = build_registry()
    biz = settings.section("business")
    V, C = float(biz["value_per_conversion"]), float(biz["cost_per_intervention"])
    out: dict = {
        "backend": ml_compat.BACKEND, "models_available": list(reg), "n_train": len(train), "n_test": len(test),
        "train_rate": float(y_tr.mean()), "test_rate": float(y_te.mean()), "page_values_in_primary_model": not settings.drop_page_values,
        "data_sha256": hashlib.sha256(open(settings.path("raw_data"), "rb").read()).hexdigest()[:16],
        "business_assumptions": {"value_per_conversion": V, "cost_per_intervention": C},
    }
    folds = 3 if quick else settings.cv_folds

    # ---- 1. imbalance strategy benchmark ------------------------------------------------------------------
    log.info("Stage 1: imbalance strategy benchmark")
    bench = []
    for name, spec in reg.items():
        for imb in IMBALANCE:
            cv = unit(settings, f"bench_{name}_{imb}_{folds}", lambda: cross_validate(spec, settings, train, y_tr, imb, folds=folds))
            bench.append({"model": name, "imbalance": imb, "roc_auc": cv["roc_auc_mean"], "roc_auc_std": cv["roc_auc_std"],
                          "pr_auc": cv["pr_auc_mean"], "pr_auc_std": cv["pr_auc_std"], "seconds": cv["seconds"]})
            log.info("  %-18s %-13s PR-AUC %.4f ROC-AUC %.4f", name, imb, cv["pr_auc_mean"], cv["roc_auc_mean"])
    out["imbalance_benchmark"] = bench
    best_imb = {}
    for name in reg:
        rows = [b for b in bench if b["model"] == name]
        best_imb[name] = max(rows, key=lambda r: r["pr_auc"])["imbalance"]
    out["best_imbalance"] = best_imb

    # ---- 2. tuning + 3. comparison ---------------------------------------------------------------------------
    log.info("Stage 2: tuning and model comparison")
    tuned, tuning_log, comp = {}, {}, []
    oofs = {}
    for name, spec in reg.items():
        params, res = unit(settings, f"tune_{name}", lambda: tune(spec, settings, train, y_tr, best_imb[name], folds=3))
        tuned[name], tuning_log[name] = params, res
        cv = unit(settings, f"cmp_{name}_{folds}", lambda: cross_validate(spec, settings, train, y_tr, best_imb[name], params, folds=folds))
        oofs[name] = cv["oof"]
        comp.append({"model": name, "imbalance": best_imb[name], "params": params, "roc_auc": cv["roc_auc_mean"],
                     "roc_auc_std": cv["roc_auc_std"], "pr_auc": cv["pr_auc_mean"], "pr_auc_std": cv["pr_auc_std"],
                     "brier_oof": M.brier(y_tr, cv["oof"]), "seconds": cv["seconds"]})
        log.info("  %-18s CV PR-AUC %.4f +- %.4f  ROC-AUC %.4f", name, cv["pr_auc_mean"], cv["pr_auc_std"], cv["roc_auc_mean"])
    out["comparison"], out["tuning"] = comp, tuning_log
    # One-standard-error rule: among models whose CV PR-AUC is within 1 SE of the best, take the simplest.
    # (Adopted after the first run showed LogisticRegression and RandomForest tied to 3 decimals; the
    #  max-PR-AUC alternative is reported too: see out["champion_by_max_pr_auc"].)
    complexity = {"LogisticRegression": 0, "DecisionTree": 1, "RandomForest": 2, "GradientBoosting": 3,
                  "XGBoost": 4, "LightGBM": 4, "CatBoost": 4}
    best_row = max(comp, key=lambda r: r["pr_auc"])
    se = best_row["pr_auc_std"] / np.sqrt(folds)
    eligible = [r for r in comp if r["pr_auc"] >= best_row["pr_auc"] - se]
    champ = min(eligible, key=lambda r: (complexity.get(r["model"], 9), -r["pr_auc"]))["model"]
    out["champion"] = champ
    out["champion_by_max_pr_auc"] = best_row["model"]
    out["champion_rule"] = {"rule": "simplest model within 1 standard error of the best CV PR-AUC", "se": float(se),
                            "eligible": [r["model"] for r in eligible]}
    spec = reg[champ]
    params, imb = tuned[champ], best_imb[champ]

    # ---- 4. feature selection --------------------------------------------------------------------------------
    log.info("Stage 3: feature importance and selection for %s", champ)
    pi = unit(settings, f"perm_{champ}_{folds}", lambda: permutation_importance(spec, settings, train, y_tr, params, imb, (), repeats=2 if quick else 3, folds=folds))
    out["permutation_importance"] = pi.to_dict("records")
    droppable = set(settings.cols("categorical")) | {"BounceRates", "ExitRates", "SpecialDay"}
    drop = [r["feature"] for r in out["permutation_importance"]
            if r["feature"] in droppable and r["mean_drop"] < 0.001 and r["folds_positive"] <= folds // 2]
    cv_full = unit(settings, f"full_{champ}_{folds}", lambda: cross_validate(spec, settings, train, y_tr, imb, params, folds=folds))
    cv_sel = unit(settings, f"sel_{champ}_{folds}_{len(drop)}", lambda: cross_validate(spec, settings, train, y_tr, imb, params, exclude=drop, folds=folds))
    cv_noeng = None
    out["feature_selection"] = {
        "dropped": drop, "pr_auc_full": cv_full["pr_auc_mean"], "pr_auc_selected": cv_sel["pr_auc_mean"],
        "pr_auc_full_std": cv_full["pr_auc_std"], "pr_auc_selected_std": cv_sel["pr_auc_std"],
        "rule": "drop a categorical/rate input if shuffling it costs < 0.001 PR-AUC and hurts in <= half of the folds",
    }
    # keep the reduced set only if it does not lose more than 0.002 PR-AUC
    keep_exclude = drop if cv_sel["pr_auc_mean"] >= cv_full["pr_auc_mean"] - 0.002 else []
    out["feature_selection"]["applied"] = bool(keep_exclude)
    oof = cv_sel["oof"] if keep_exclude else cv_full["oof"]
    log.info("  dropped %s (applied=%s): PR-AUC %.4f -> %.4f", drop, bool(keep_exclude), cv_full["pr_auc_mean"], cv_sel["pr_auc_mean"])

    # ---- 5. threshold from business value (training OOF only) -----------------------------------------------------
    thr = choose_threshold(y_tr, oof, V, C, int(biz["threshold_grid_points"]))
    out["threshold"] = thr
    sens = []
    for ratio in (2, 5, 10, 20, 40):
        t = choose_threshold(y_tr, oof, V, V / ratio, int(biz["threshold_grid_points"]))
        sens.append({"value_to_cost_ratio": ratio, **t})
    out["threshold_sensitivity"] = sens
    out["oof_metrics_at_threshold"] = M.full_report(y_tr, oof, thr["threshold"])

    # ---- 6. single evaluation on the untouched test set -----------------------------------------------------------
    log.info("Stage 4: final fit and one test evaluation")
    final = SessionClassifier(spec, settings, imb, params, exclude=keep_exclude).fit(train, y_tr)
    p_te = final.predict_proba(test)
    rep = M.full_report(y_te, p_te, thr["threshold"])
    rep["roc_auc_ci95"] = M.bootstrap_ci(y_te, p_te, M.roc_auc, 400)
    rep["pr_auc_ci95"] = M.bootstrap_ci(y_te, p_te, M.average_precision, 400)
    rep["net_value_test"] = float(V * rep["tp"] - C * (rep["tp"] + rep["fp"]))
    rep["net_value_flag_everyone"] = float(V * y_te.sum() - C * len(y_te))
    rep["net_value_flag_nobody"] = 0.0
    out["test"] = rep
    out["test_baselines"] = {
        "majority_class_accuracy": float(1 - y_te.mean()), "majority_class_recall": 0.0,
        "random_pr_auc": float(y_te.mean()),
        "single_feature_pr_auc": {f: M.average_precision(y_te, test[f].to_numpy(float) * s)
                                  for f, s in (("ProductRelated", 1), ("ExitRates", -1), ("BounceRates", -1))},
    }
    # reference only (not used to choose anything): same recipe WITH PageValues
    ref = SessionClassifier(spec, settings, imb, params, use_page_values=True, exclude=keep_exclude).fit(train, y_tr)
    p_ref = ref.predict_proba(test)
    out["test_with_page_values_reference"] = {"roc_auc": M.roc_auc(y_te, p_ref), "pr_auc": M.average_precision(y_te, p_ref),
                                              "note": "reference only; PageValues==0 never converts in this data"}
    # all models on the test set, for the dashboard comparison (selection already made on CV)
    per_model_test = {}
    for name, sp in reg.items():
        m = SessionClassifier(sp, settings, best_imb[name], tuned[name]).fit(train, y_tr)
        pt = m.predict_proba(test)
        per_model_test[name] = {"roc_auc": M.roc_auc(y_te, pt), "pr_auc": M.average_precision(y_te, pt)}
    out["test_all_models"] = per_model_test

    # ---- 7. figures + artifacts ----------------------------------------------------------------------------------------
    fig_dir = settings.path("figures_dir")
    plots.pr_roc_figure(y_te, {champ: p_te}, fig_dir / "test_pr_roc.png", f"{champ} on the held-out test set")
    plots.calibration_figure(y_te, p_te, fig_dir / "test_calibration.png")
    vc = value_curve(y_tr, oof, V, C, int(biz["threshold_grid_points"]))
    vc.to_csv(settings.path("reports_dir") / "value_curve_oof.csv", index=False)
    pd.DataFrame({"y": y_tr, **{f"oof_{k}": v for k, v in oofs.items()}, "oof_champion_selected": oof}).to_csv(
        settings.path("reports_dir") / "oof_predictions.csv", index=False)
    pd.DataFrame(bench).to_csv(settings.path("reports_dir") / "imbalance_benchmark.csv", index=False)
    pd.DataFrame(comp).drop(columns=["params"]).to_csv(settings.path("reports_dir") / "model_comparison.csv", index=False)

    version = dt.date.today().strftime("%Y%m%d")
    bg = train.sample(n=min(int(settings.section("explain")["background_size"]), len(train)), random_state=settings.seed)
    bg = bg[final.pre.input_columns].reset_index(drop=True)
    meta = {"model": champ, "version": version, "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "backend": ml_compat.BACKEND, "imbalance": imb, "params": params, "excluded_features": keep_exclude,
            "uses_page_values": False, "input_columns": final.pre.input_columns, "seed": settings.seed,
            "data_sha256": out["data_sha256"], "threshold": thr["threshold"]}
    model_dir = settings.path("model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {"model": final, "threshold": thr["threshold"], "metadata": meta, "background": bg}
    for target in (model_dir / "model_bundle.pkl", model_dir / f"model_bundle_v{version}.pkl"):
        with open(target, "wb") as fh:
            pickle.dump(bundle, fh)
    with open(model_dir / "pipeline_with_page_values_reference.pkl", "wb") as fh:
        pickle.dump(ref, fh)
    out["metadata"] = meta
    out["seconds_total"] = time.time() - t_start
    (model_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=_json))
    log_run(settings, "final_champion", {"model": champ, "imbalance": imb, **{f"p_{k}": v for k, v in params.items()}},
            {"test_pr_auc": rep["pr_auc"], "test_roc_auc": rep["roc_auc"], "cv_pr_auc": cv_full["pr_auc_mean"]},
            artifact=model_dir / "model_bundle.pkl")
    log.info("Done in %.0fs. Test PR-AUC %.3f ROC-AUC %.3f", out["seconds_total"], rep["pr_auc"], rep["roc_auc"])
    return out


if __name__ == "__main__":
    try:
        run(quick="--quick" in sys.argv)
    except BudgetExceeded as exc:
        log.info("Time slice used up before %s - progress cached; re-run to continue.", exc)
        sys.exit(3)
