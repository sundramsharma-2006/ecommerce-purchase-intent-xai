# Model card: purchase-intent classifier

**Model:** LogisticRegression (20261015), imbalance strategy `none`, params {'C': 0.03}. Backend: mini_sklearn (NumPy fallback implementing the scikit-learn API subset).
**Inputs (10):** Administrative, Administrative_Duration, BounceRates, Browser, ExitRates, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, VisitorType. **Not used:** PageValues (see below) and inputs removed by selection: TrafficType, Weekend, Month, SpecialDay, OperatingSystems, Region.
**Output:** probability that the session ends in a purchase; decision = `high_intent` when probability >= 0.125, else `low_intent`. The low threshold reflects cheap interventions, so a 24% probability can still be flagged: `high_intent` means "worth a nudge", not "will probably buy".

## Intended use
Rank/flag sessions for low-cost interventions (a banner, a nudge, remarketing audience) where a wrong flag costs little. Decision support for a conversion-optimisation team; the operating threshold belongs to the business.

## Not intended for
Anything that identifies, profiles or treats individual people differently in a way that matters to them (pricing discrimination, access decisions). Not validated for another shop, another period, or another traffic mix. Not a causal model: it does not say an intervention will change the outcome (a high score often means the visitor would have bought anyway).

## Data
12,000 sessions, 15.7% purchases; split 80/20 stratified (seed 42); model selection by 5-fold stratified CV on the 80% only; the 20% test set was used once for the final model (plus a labelled PageValues reference run and the per-model test table). The data appears synthetic (463 sessions with BounceRates > ExitRates), so results describe this file.

## Performance (held-out test, n=2400)
PR-AUC 0.558 (95% CI 0.51-0.61); ROC-AUC 0.837 (CI 0.81-0.86); Brier 0.098. At threshold 0.125: recall 0.81, precision 0.32, flags 40% of sessions; TP 306 / FP 663 / FN 71 / TN 1360. Baselines: base rate PR-AUC 0.157; ProductRelated alone 0.531; "always no purchase" accuracy 84.3% with zero recall.

## Threshold
Chosen on training out-of-fold predictions to maximise assumed net value (value 40 per captured purchase, cost 4 per flagged session). These two numbers are assumptions, not measurements. Sensitivity:

| value:cost | threshold | precision | recall | flagged (train OOF) |
|---|---|---|---|---|
| 2:1 | 0.470 | 0.65 | 0.32 | 744 |
| 5:1 | 0.255 | 0.47 | 0.63 | 2032 |
| 10:1 | 0.125 | 0.32 | 0.83 | 3846 |
| 20:1 | 0.045 | 0.22 | 0.95 | 6445 |
| 40:1 | 0.020 | 0.19 | 0.99 | 8001 |

## The PageValues decision
PageValues is the strongest raw signal (correlation 0.64) but no session with PageValues = 0 converts, and in a live visit it reflects pages near checkout. It is excluded from the deployed model; with it, test PR-AUC would be 0.85 and ROC-AUC 0.97, a deployment-leakage number.

## Explanations
Global importance (permutation, drop in PR-AUC): top features below. Local explanations are permutation-sampling Shapley values in probability units, plus a LIME-style surrogate (mean local R2 0.22: treat LIME numbers as a rough second opinion). Both explain the model, not causal effects.

| Feature | mean PR-AUC drop | std |
|---|---|---|
| ProductRelated | 0.2755 | 0.0308 |
| ExitRates | 0.0441 | 0.0063 |
| Administrative | 0.0076 | 0.0051 |
| BounceRates | 0.0049 | 0.0018 |
| Administrative_Duration | 0.0018 | 0.0014 |
| ProductRelated_Duration | 0.0014 | 0.0016 |
| Informational_Duration | 0.0013 | 0.0041 |
| VisitorType | 0.0013 | 0.0011 |

## Limitations and failure modes
- Whole-session totals only: behaviour at an earlier point in a visit was never tested.
- Two features (ProductRelated, ExitRates) carry almost all the signal; sessions with little browsing look alike to the model.
- Probabilities of the deployed logistic model are reasonably calibrated, with slight under-prediction in the highest-score bin (see `reports/figures/test_calibration.png`); GradientBoosting trained with class weights is not (Brier 0.159).
- Only 4 of 6 planned algorithms were trained in the build environment; real scikit-learn / boosters may rank differently.
- Monitor in production: conversion rate, input distributions (ProductRelated, ExitRates), and calibration; retrain when they drift. No monitoring is implemented in this repo.
