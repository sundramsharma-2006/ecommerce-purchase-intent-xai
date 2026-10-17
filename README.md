# E-Commerce Purchase-Intent Prediction with Explainable AI

Predicts whether an online shopping session will end in a purchase from browsing behaviour, explains each prediction, and serves it through an API and dashboard. Built against `docs/PRD_Purchase_Intent_XAI.docx` (CRISP-DM). Data: 12,000 sessions, 17 inputs, 15.7% purchases (`dataset/`).

## Verification status (read first)
| Component | Status |
|---|---|
| Data loader + validation, preprocessing, feature engineering, metrics | Run; covered by unit tests |
| Training pipeline (imbalance benchmark, tuning, CV comparison, feature selection, threshold, single test evaluation) | Run end to end (`python -m src.train`) |
| Algorithms | **4 of the 6 requested were trained** (LogisticRegression, DecisionTree, RandomForest, GradientBoosting). XGBoost, LightGBM and CatBoost could not be installed in the build sandbox. They are registered automatically in `src/models/trainer.py` when installed: run `pip install -r requirements.txt` then `python -m src.train` to get all six. |
| ML backend | `mini_sklearn (NumPy fallback implementing the scikit-learn API subset)`. With scikit-learn installed, the real library is used automatically, and numbers will differ slightly. |
| Explainability | Model-agnostic permutation-sampling **Shapley values** and a **LIME-style** surrogate implemented in `src/explain/` (verified by tests: efficiency, exactness on an additive model). The `shap` and `lime` packages themselves were not available; hooks for them are documented. |
| Notebooks (4) | Executed, 0 errors (`tools/build_notebooks.py`) |
| Unit tests | 51 tests (50 pass, 1 skipped: needs pydantic) |
| FastAPI service (`src/api`) | Written to the FastAPI/Pydantic v2 API, **never started** (packages not installable). The scoring logic it calls is tested. |
| Streamlit dashboard (`dashboard/`) | Written, **never rendered**. Its data layer is tested. |
| Docker, Compose, MLflow, GitHub Actions, AWS notes | Written, **not executed**. Experiment runs are logged to `experiments/runs.jsonl` when MLflow is absent. |
| Code style (black / ruff) | **Not run** (tools unavailable). Some lines exceed 120 characters, so the first CI run will likely need `black .` and a few `ruff` fixes. |

## Headline results (and their limits)
- **Deployed model: LogisticRegression** (CV PR-AUC 0.531), chosen by the one-standard-error rule. RandomForest was nominally best by 0.0005 PR-AUC, a difference far inside the fold-to-fold noise. The rule was adopted after seeing that tie, and the raw-best model's test numbers are in the table below.
- **Held-out test set (touched once):** PR-AUC **0.558** (95% CI 0.51-0.61), ROC-AUC **0.837** (CI 0.81-0.86), base rate 0.157. At the chosen threshold 0.125: recall 0.81, precision 0.32.
- **A one-line baseline is close:** using ProductRelated alone gives test PR-AUC 0.531. The full pipeline adds roughly 0.03 PR-AUC. Do not oversell the model.
- **PageValues was excluded on purpose.** In this data no session with PageValues = 0 ever converts (6,612 sessions), and the value is only known late in a visit. Including it lifts test PR-AUC from 0.56 to 0.85 (reported as a labelled upper bound, not a result).
- **Accuracy is not the headline:** "always no purchase" is 84.3% accurate and finds no buyers.
- **Threshold from business value, not 0.5.** With *assumed* figures (value 40 per captured purchase, cost 4 per intervention, edit in `config/config.yaml`) the threshold is 0.125. On the test set the net value is 8,364, versus 5,480 for flagging everyone and 0 for flagging nobody. These are upper bounds: they assume every flagged would-be buyer is captured.
- **Not validated:** scoring *during* a session. The data has whole-session totals only, so "real time" here means scoring from the behaviour so far, evaluated on complete sessions. The data also looks synthetic (463 sessions have BounceRates above ExitRates).

| Model | Imbalance | CV PR-AUC | CV ROC-AUC | CV Brier | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|---|---|---|
| LogisticRegression | none | 0.531 +- 0.028 | 0.832 | 0.100 | 0.555 | 0.836 |
| DecisionTree | none | 0.489 +- 0.020 | 0.802 | 0.105 | 0.479 | 0.797 |
| RandomForest | none | 0.532 +- 0.025 | 0.830 | 0.099 | 0.543 | 0.832 |
| GradientBoosting | class_weight | 0.521 +- 0.021 | 0.830 | 0.159 | 0.553 | 0.835 |

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
