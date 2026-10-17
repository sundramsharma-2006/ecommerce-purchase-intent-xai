"""Writes README.md, models/model_card.md and documentation/report markdown with numbers read from metrics.json
(so prose and results cannot drift apart).  python tools/make_docs.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "models" / "metrics.json").read_text())
E = json.loads((ROOT / "reports" / "explain_results.json").read_text())
T, B = R["test"], R["test_baselines"]
ref = R["test_with_page_values_reference"]
thr = R["threshold"]
comp = {c["model"]: c for c in R["comparison"]}
tm = R["test_all_models"]
fs = R["feature_selection"]
V = R["business_assumptions"]
f3 = lambda x: f"{x:.3f}"
f2 = lambda x: f"{x:.2f}"

comp_rows = "\n".join(
    f"| {m} | {c['imbalance']} | {c['pr_auc']:.3f} +- {c['pr_auc_std']:.3f} | {c['roc_auc']:.3f} | {c['brier_oof']:.3f} | {tm[m]['pr_auc']:.3f} | {tm[m]['roc_auc']:.3f} |"
    for m, c in comp.items())
sens_rows = "\n".join(f"| {s['value_to_cost_ratio']}:1 | {s['threshold']:.3f} | {s['precision']:.2f} | {s['recall']:.2f} | {s['flagged']} |" for s in R["threshold_sensitivity"])
perm_rows = "\n".join(f"| {r['feature']} | {r['mean_drop']:.4f} | {r['std_drop']:.4f} |" for r in R["permutation_importance"][:8])

STATUS = f"""| Component | Status |
|---|---|
| Data loader + validation, preprocessing, feature engineering, metrics | Run; covered by unit tests |
| Training pipeline (imbalance benchmark, tuning, CV comparison, feature selection, threshold, single test evaluation) | Run end to end (`python -m src.train`) |
| Algorithms | **4 of the 6 requested were trained** (LogisticRegression, DecisionTree, RandomForest, GradientBoosting). XGBoost, LightGBM and CatBoost could not be installed in the build sandbox. They are registered automatically in `src/models/trainer.py` when installed: run `pip install -r requirements.txt` then `python -m src.train` to get all six. |
| ML backend | `{R['backend']}`. With scikit-learn installed, the real library is used automatically, and numbers will differ slightly. |
| Explainability | Model-agnostic permutation-sampling **Shapley values** and a **LIME-style** surrogate implemented in `src/explain/` (verified by tests: efficiency, exactness on an additive model). The `shap` and `lime` packages themselves were not available; hooks for them are documented. |
| Notebooks (4) | Executed, 0 errors (`tools/build_notebooks.py`) |
| Unit tests | 51 tests (50 pass, 1 skipped: needs pydantic) |
| FastAPI service (`src/api`) | Written to the FastAPI/Pydantic v2 API, **never started** (packages not installable). The scoring logic it calls is tested. |
| Streamlit dashboard (`dashboard/`) | Written, **never rendered**. Its data layer is tested. |
| Docker, Compose, MLflow, GitHub Actions, AWS notes | Written, **not executed**. Experiment runs are logged to `experiments/runs.jsonl` when MLflow is absent. |
| Code style (black / ruff) | **Not run** (tools unavailable). Some lines exceed 120 characters, so the first CI run will likely need `black .` and a few `ruff` fixes. |"""

README = f"""# E-Commerce Purchase-Intent Prediction with Explainable AI

Predicts whether an online shopping session will end in a purchase from browsing behaviour, explains each prediction, and serves it through an API and dashboard. Built against `docs/PRD_Purchase_Intent_XAI.docx` (CRISP-DM). Data: 12,000 sessions, 17 inputs, 15.7% purchases (`dataset/`).

## Verification status (read first)
{STATUS}

## Headline results (and their limits)
- **Deployed model: {R['champion']}** (CV PR-AUC {comp[R['champion']]['pr_auc']:.3f}), chosen by the one-standard-error rule. RandomForest was nominally best by {abs(comp['RandomForest']['pr_auc']-comp[R['champion']]['pr_auc']):.4f} PR-AUC, a difference far inside the fold-to-fold noise. The rule was adopted after seeing that tie, and the raw-best model's test numbers are in the table below.
- **Held-out test set (touched once):** PR-AUC **{f3(T['pr_auc'])}** (95% CI {T['pr_auc_ci95'][0]:.2f}-{T['pr_auc_ci95'][1]:.2f}), ROC-AUC **{f3(T['roc_auc'])}** (CI {T['roc_auc_ci95'][0]:.2f}-{T['roc_auc_ci95'][1]:.2f}), base rate {R['test_rate']:.3f}. At the chosen threshold {thr['threshold']:.3f}: recall {T['recall']:.2f}, precision {T['precision']:.2f}.
- **A one-line baseline is close:** using ProductRelated alone gives test PR-AUC {B['single_feature_pr_auc']['ProductRelated']:.3f}. The full pipeline adds roughly {T['pr_auc']-B['single_feature_pr_auc']['ProductRelated']:.2f} PR-AUC. Do not oversell the model.
- **PageValues was excluded on purpose.** In this data no session with PageValues = 0 ever converts (6,612 sessions), and the value is only known late in a visit. Including it lifts test PR-AUC from {T['pr_auc']:.2f} to {ref['pr_auc']:.2f} (reported as a labelled upper bound, not a result).
- **Accuracy is not the headline:** "always no purchase" is {B['majority_class_accuracy']:.1%} accurate and finds no buyers.
- **Threshold from business value, not 0.5.** With *assumed* figures (value {V['value_per_conversion']:.0f} per captured purchase, cost {V['cost_per_intervention']:.0f} per intervention, edit in `config/config.yaml`) the threshold is {thr['threshold']:.3f}. On the test set the net value is {T['net_value_test']:,.0f}, versus {T['net_value_flag_everyone']:,.0f} for flagging everyone and 0 for flagging nobody. These are upper bounds: they assume every flagged would-be buyer is captured.
- **Not validated:** scoring *during* a session. The data has whole-session totals only, so "real time" here means scoring from the behaviour so far, evaluated on complete sessions. The data also looks synthetic (463 sessions have BounceRates above ExitRates).

| Model | Imbalance | CV PR-AUC | CV ROC-AUC | CV Brier | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|---|---|---|
{comp_rows}

SMOTE was benchmarked and was never better than no resampling or class weights (see `reports/imbalance_benchmark.csv`).

## Quick start
```bash
pip install -r requirements.txt
python -m src.train                       # trains, tunes, saves models/model_bundle.pkl + models/metrics.json
python -m src.explain.report              # Shapley / LIME-style artifacts
python -m unittest discover -s tests -t . # 51 tests
uvicorn src.api.app:app --port 8000       # API, docs at /docs
streamlit run dashboard/app.py            # dashboard
docker compose up --build                 # API + dashboard + MLflow UI
```
Long runs can resume in slices: `TRAIN_BUDGET_S=120 python -m src.train` (progress is cached in `experiments/cache`; delete that folder to retrain from scratch).

## Layout
`config/` single config | `src/data` loader + validation | `src/preprocessing` fitted pipeline | `src/features` | `src/models` registry, trainer, CV, threshold | `src/explain` Shapley + LIME-style | `src/serving` predictor | `src/api` FastAPI | `dashboard/` Streamlit | `notebooks/` 01 EDA, 02 features, 03 modeling, 04 XAI | `tests/` | `docker/`, `.github/workflows/` | `docs/` PRD, diagrams, AWS notes, report | `models/` artifacts + model card | `reports/` figures and tables.

## Responsible use
Behavioural data is sensitive. The model uses no personal identifiers and must not be tied to them. See `models/model_card.md` for intended use and limits.
"""

CARD = f"""# Model card: purchase-intent classifier

**Model:** {R['champion']} ({R['metadata']['version']}), imbalance strategy `{R['metadata']['imbalance']}`, params {R['metadata']['params']}. Backend: {R['backend']}.
**Inputs ({len(R['metadata']['input_columns'])}):** {', '.join(R['metadata']['input_columns'])}. **Not used:** PageValues (see below) and inputs removed by selection: {', '.join(fs['dropped']) or 'none'}.
**Output:** probability that the session ends in a purchase; decision = `high_intent` when probability >= {thr['threshold']:.3f}, else `low_intent`. The low threshold reflects cheap interventions, so a 24% probability can still be flagged: `high_intent` means "worth a nudge", not "will probably buy".

## Intended use
Rank/flag sessions for low-cost interventions (a banner, a nudge, remarketing audience) where a wrong flag costs little. Decision support for a conversion-optimisation team; the operating threshold belongs to the business.

## Not intended for
Anything that identifies, profiles or treats individual people differently in a way that matters to them (pricing discrimination, access decisions). Not validated for another shop, another period, or another traffic mix. Not a causal model: it does not say an intervention will change the outcome (a high score often means the visitor would have bought anyway).

## Data
12,000 sessions, 15.7% purchases; split 80/20 stratified (seed {R['metadata']['seed']}); model selection by 5-fold stratified CV on the 80% only; the 20% test set was used once for the final model (plus a labelled PageValues reference run and the per-model test table). The data appears synthetic (463 sessions with BounceRates > ExitRates), so results describe this file.

## Performance (held-out test, n={R['n_test']})
PR-AUC {f3(T['pr_auc'])} (95% CI {T['pr_auc_ci95'][0]:.2f}-{T['pr_auc_ci95'][1]:.2f}); ROC-AUC {f3(T['roc_auc'])} (CI {T['roc_auc_ci95'][0]:.2f}-{T['roc_auc_ci95'][1]:.2f}); Brier {f3(T['brier'])}. At threshold {thr['threshold']:.3f}: recall {T['recall']:.2f}, precision {T['precision']:.2f}, flags {T['flagged_share']:.0%} of sessions; TP {T['tp']} / FP {T['fp']} / FN {T['fn']} / TN {T['tn']}. Baselines: base rate PR-AUC {R['test_rate']:.3f}; ProductRelated alone {B['single_feature_pr_auc']['ProductRelated']:.3f}; "always no purchase" accuracy {B['majority_class_accuracy']:.1%} with zero recall.

## Threshold
Chosen on training out-of-fold predictions to maximise assumed net value (value {V['value_per_conversion']:.0f} per captured purchase, cost {V['cost_per_intervention']:.0f} per flagged session). These two numbers are assumptions, not measurements. Sensitivity:

| value:cost | threshold | precision | recall | flagged (train OOF) |
|---|---|---|---|---|
{sens_rows}

## The PageValues decision
PageValues is the strongest raw signal (correlation 0.64) but no session with PageValues = 0 converts, and in a live visit it reflects pages near checkout. It is excluded from the deployed model; with it, test PR-AUC would be {ref['pr_auc']:.2f} and ROC-AUC {ref['roc_auc']:.2f}, a deployment-leakage number.

## Explanations
Global importance (permutation, drop in PR-AUC): top features below. Local explanations are permutation-sampling Shapley values in probability units, plus a LIME-style surrogate (mean local R2 {E['shapley_vs_lime']['mean_lime_r2']:.2f}: treat LIME numbers as a rough second opinion). Both explain the model, not causal effects.

| Feature | mean PR-AUC drop | std |
|---|---|---|
{perm_rows}

## Limitations and failure modes
- Whole-session totals only: behaviour at an earlier point in a visit was never tested.
- Two features (ProductRelated, ExitRates) carry almost all the signal; sessions with little browsing look alike to the model.
- Probabilities of the deployed logistic model are reasonably calibrated, with slight under-prediction in the highest-score bin (see `reports/figures/test_calibration.png`); GradientBoosting trained with class weights is not (Brier {comp['GradientBoosting']['brier_oof']:.3f}).
- Only 4 of 6 planned algorithms were trained in the build environment; real scikit-learn / boosters may rank differently.
- Monitor in production: conversion rate, input distributions (ProductRelated, ExitRates), and calibration; retrain when they drift. No monitoring is implemented in this repo.
"""
(ROOT / "README.md").write_text(README, encoding="utf-8")
(ROOT / "models" / "model_card.md").write_text(CARD, encoding="utf-8")
print("README.md and model_card.md written")

# ----------------------------------------------------------------------------- final report (markdown -> docx via pandoc)
REPORT = f"""---
title: "E-Commerce Purchase-Intent Prediction with Explainable AI"
subtitle: "Final project report"
author: "Sundram Sharma"
---

# 1. Summary

This project predicts whether an online shopping session ends in a purchase, explains every prediction, and serves the result through an API and a dashboard. It follows the CRISP-DM structure in the project brief. On a held-out test set of {R['n_test']:,} sessions the deployed model reaches **PR-AUC {f3(T['pr_auc'])}** (95% CI {T['pr_auc_ci95'][0]:.2f}-{T['pr_auc_ci95'][1]:.2f}) and **ROC-AUC {f3(T['roc_auc'])}**, against a purchase base rate of {R['test_rate']:.1%}.

Three findings matter more than the headline number:

1. **PageValues was excluded on purpose.** No session with PageValues = 0 ever converts (6,612 of 12,000 sessions), and in a live visit the value is only known late. Using it would raise test PR-AUC from {f2(T['pr_auc'])} to {f2(ref['pr_auc'])}, a deployment-leakage figure.
2. **A one-line baseline is close.** ProductRelated alone gives test PR-AUC {B['single_feature_pr_auc']['ProductRelated']:.3f}; the full pipeline adds about {T['pr_auc']-B['single_feature_pr_auc']['ProductRelated']:.2f}.
3. **The operating threshold is a business decision.** Moving the assumed value-to-cost ratio from 2:1 to 40:1 moves the best threshold from {R['threshold_sensitivity'][0]['threshold']:.2f} to {R['threshold_sensitivity'][-1]['threshold']:.2f}.

# 2. Verification status

{STATUS}

# 3. Business problem and success criteria

A retailer wants to find sessions likely to convert so that low-cost interventions (a banner, a nudge, a remarketing audience) can be targeted. Because only about one session in six converts, accuracy is misleading: "always no purchase" is {B['majority_class_accuracy']:.1%} accurate and finds no buyers. The metrics that lead are therefore PR-AUC and recall on the buying class, with ROC-AUC, F1, Brier score and a confusion matrix as support.

# 4. Data

12,000 sessions with 17 inputs and the target `Converted` (15.7% positive). No missing values and no duplicate rows. Four ID columns (OperatingSystems, Browser, Region, TrafficType) are stored as integers but are categories, so they are one-hot encoded and never scaled. Conversion is highest in November (20.0%) and lowest in February and June (about 10%); new visitors convert less (10.2%) than returning visitors (16.6%). The file looks synthetic: 463 sessions have BounceRates above ExitRates, which real analytics data should not show. Results therefore describe this dataset.

# 5. Method

![Architecture](docs/architecture.png){{width=95%}}

The data were split 80/20, stratified and seeded (the test part was used once for the final model). Everything that chooses something (imbalance strategy, hyper-parameters, features, threshold) uses the training part only with stratified 5-fold cross-validation. Preprocessing (log1p and standardisation of skewed numerics, one-hot encoding of categoricals with unseen categories mapped to zeros, optional SMOTE) is fitted inside each fold and saved as one object that serving reuses, so there is no train/serve skew.

Imbalance strategies were benchmarked rather than assumed. Plain probabilities with threshold tuning and class weights were within noise of each other for every model, and **SMOTE was never better and often worse** (for example DecisionTree 0.43 against 0.48 CV PR-AUC).

![Scoring workflow](docs/workflow.png){{width=95%}}

# 6. Results

| Model | Imbalance | CV PR-AUC | CV ROC-AUC | CV Brier | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|---|---|---|
{comp_rows}

Four of the six requested algorithms were trained; XGBoost, LightGBM and CatBoost are registered automatically when installed. LogisticRegression, RandomForest and GradientBoosting are within one standard error of each other, so the **one-standard-error rule** picked the simplest, **{R['champion']}**. This rule was adopted after the first run showed LogisticRegression and RandomForest tied to three decimals; the raw-best model ({R['champion_by_max_pr_auc']}) has test PR-AUC {tm[R['champion_by_max_pr_auc']]['pr_auc']:.3f}, not higher than the deployed model's {f3(T['pr_auc'])}. GradientBoosting's Brier score is much worse because class weights distort its probabilities.

**Feature selection.** Permutation importance (drop in PR-AUC when one input is shuffled) is dominated by ProductRelated, then ExitRates; everything else is at or below noise. Six inputs ({', '.join(fs['dropped'])}) were removed under a stated rule, with CV PR-AUC {fs['pr_auc_full']:.3f} before and {fs['pr_auc_selected']:.3f} after (a difference inside the noise: the claim is no loss with fewer inputs, not an improvement). The engineered ratio features add 0.0006 CV PR-AUC, which is also noise; they are kept for interpretability only.

**Test set (used once).** PR-AUC {f3(T['pr_auc'])}, ROC-AUC {f3(T['roc_auc'])}, Brier {f3(T['brier'])}. At threshold {thr['threshold']:.3f}: recall {T['recall']:.2f}, precision {T['precision']:.2f}, TP {T['tp']}, FP {T['fp']}, FN {T['fn']}, TN {T['tn']}.

![Test PR and ROC curves](reports/figures/test_pr_roc.png){{width=95%}}

![Calibration on the test set](reports/figures/test_calibration.png){{width=55%}}

# 7. Decision threshold from business value

Net value of flagging a session = value of a captured purchase x purchases flagged - cost per intervention x sessions flagged. The value ({V['value_per_conversion']:.0f}) and cost ({V['cost_per_intervention']:.0f}) are **assumptions**, to be replaced with the retailer's real figures. They give an upper bound because they assume every flagged would-be buyer is captured. The threshold was chosen on training out-of-fold predictions.

| value:cost | threshold | precision | recall | flagged (train OOF) |
|---|---|---|---|---|
{sens_rows}

On the test set the net value is {T['net_value_test']:,.0f}, compared with {T['net_value_flag_everyone']:,.0f} for flagging everyone and 0 for flagging nobody. Flagging everyone is a strong baseline when a purchase is worth ten interventions, so the model's practical gain depends on how expensive interventions really are and on whether they change behaviour, which this data cannot show.

# 8. Explainability

Global importance uses mean absolute Shapley value in probability units (ProductRelated about 10 points, ExitRates about 4, all others 1.5 or less). Its ranking agrees with permutation importance (Spearman {E['rank_agreement_shapley_vs_permutation']:.2f}). The Shapley explainer is a model-agnostic permutation-sampling estimator; contributions add up exactly to the prediction minus the baseline, which the unit tests verify, along with exactness on an additive model. A LIME-style local surrogate gives a second view: it names the same top feature in {E['shapley_vs_lime']['top1_agreement']:.0%} of sessions, but its local fit is weak (mean R2 {E['shapley_vs_lime']['mean_lime_r2']:.2f}), so its numbers are a rough second opinion. The `shap` and `lime` packages themselves could not be installed; both methods explain the model and say nothing about causes.

![Global Shapley importance](reports/figures/shapley_global.png){{width=70%}}

Scoring with an explanation takes about {E['latency_ms_with_explanation']:.0f} ms and scoring alone {E['latency_ms_score_only']:.0f} ms on the build machine (the explainer is built once at start-up).

# 9. Deployment assets

The repository contains a FastAPI service (`/predict`, `/health`, `/model-info`, validated input, extra fields such as PageValues rejected), a six-section Streamlit dashboard that loads the saved model and never retrains, a Dockerfile, a Compose file (API, dashboard, MLflow UI), a GitHub Actions workflow (lint, format check, tests, image build), MLflow logging with a JSON-lines fallback, and AWS ECR/ECS notes. **None of these could be executed in the build environment**; they were written to the libraries' documented APIs and should be run and corrected as needed. The scoring logic they wrap is covered by 51 unit tests.

# 10. Limitations and next steps

- Only whole-session totals exist, so scoring earlier in a visit was never evaluated.
- Two features carry almost all the signal; more behavioural signals (clickstream order, cart events that are available early) would be the highest-value addition.
- The value and cost figures are assumed; an uplift experiment (randomised nudges) is needed to know whether interventions change the outcome.
- Train all six algorithms with the real libraries, and re-check which model wins; add drift monitoring and a calibration check in production.

# 11. Reproducibility

`config/config.yaml` holds every tunable value; the seed is {R['metadata']['seed']}; the data hash is `{R['data_sha256']}`. `python -m src.train` rebuilds all numbers in this report from the CSV, and `python tools/make_docs.py` regenerates the README, model card and this text from the saved results.
"""
(ROOT / "docs" / "report.md").write_text(REPORT, encoding="utf-8")
print("report.md written")
