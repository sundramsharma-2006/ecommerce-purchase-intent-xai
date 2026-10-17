
from tools.build_notebooks import md, code

SETUP = """
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
sys.path.insert(0, str(Path.cwd()))
from src.config import load_settings
from src.data.loader import load_sessions, stratified_split
from src.features.engineering import add_engineered_features
from src.utils import metrics as M
S = load_settings()
df = load_sessions(S)
"""

EDA = [
md("""# 01 - Data understanding (EDA)
CRISP-DM phase 2. Every figure ends with a takeaway that feeds a modelling decision. Data: 12,000 sessions, 17 inputs + `Converted`."""),
code(SETUP),
code("""print(df.shape, 'duplicates:', int(df.duplicated().sum()), 'missing cells:', int(df.isna().sum().sum()))
rate = df.Converted.mean()
print(f'conversion rate {rate:.3%} ({int(df.Converted.sum())} purchases)  -> majority-class accuracy {1-rate:.3%}')
df.describe().T.round(3)"""),
md("""**Takeaway:** clean file (no duplicates / missing values). Only ~1 in 6.4 sessions buys, so a model that always says "no purchase" is 84.3% accurate and captures nobody. Accuracy is the wrong headline metric; PR-AUC and recall on buyers are used instead."""),
code("""fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ['ProductRelated', 'ExitRates', 'BounceRates']):
    sns.kdeplot(data=df, x=col, hue='Converted', common_norm=False, fill=True, ax=ax, clip=(0, df[col].quantile(.99)))
    ax.set_title(f'{col}: buyers vs non-buyers')
plt.tight_layout(); plt.show()
print(df.groupby('Converted')[['ProductRelated','ProductRelated_Duration','ExitRates','BounceRates','Administrative']].median().round(3))"""),
md("""**Takeaway:** buyers browse far more product pages and leave from lower-exit pages. These are the behaviours the model should (and does) use. Both are known while the session is still running."""),
code("""fig, ax = plt.subplots(1, 2, figsize=(13, 4))
order = S.cols('months')
m = df.groupby('Month').Converted.agg(['mean','size']).reindex([x for x in order if x in df.Month.unique()])
ax[0].bar(m.index, m['mean'], color='#1F3A5F'); ax[0].axhline(df.Converted.mean(), ls='--', c='grey'); ax[0].set_title('Conversion rate by month')
v = df.groupby('VisitorType').Converted.agg(['mean','size'])
ax[1].bar(v.index, v['mean'], color='#E07A1F'); ax[1].set_title('Conversion rate by visitor type')
plt.tight_layout(); plt.show()
print(m.round(3)); print(v.round(3))"""),
md("""**Takeaway:** conversion is highest in Nov (20.0%) and May/Dec, lowest in Feb/June (about 10%); new visitors convert less (10.2%) than returning ones (16.6%). Month sizes are very uneven (Feb: 377 sessions, May: 3,554), so month effects for small months are noisy. Later we test whether Month adds anything once behaviour is known."""),
code("""num = ['Administrative','Informational','ProductRelated','Administrative_Duration','Informational_Duration','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay','Converted']
c = df[num].corr()
plt.figure(figsize=(9, 7)); sns.heatmap(c, annot=True, fmt='.2f', cmap='RdBu_r', center=0, annot_kws={'size': 7}); plt.title('Correlation matrix'); plt.show()
c['Converted'].drop('Converted').sort_values()"""),
md("""**Takeaway:** PageValues (0.64) and ProductRelated (0.48) are the strongest linear signals; ExitRates and BounceRates are negative and strongly correlated with each other (redundancy for linear models)."""),
md("""## The PageValues check (leakage decision)
The brief warns PageValues is partly a checkout-adjacent signal. Test it directly."""),
code("""pv0 = df[df.PageValues == 0]; pv1 = df[df.PageValues > 0]
print(f'PageValues == 0: {len(pv0)} sessions ({len(pv0)/len(df):.1%}), purchases: {int(pv0.Converted.sum())}')
print(f'PageValues  > 0: {len(pv1)} sessions, conversion rate {pv1.Converted.mean():.1%}')"""),
md("""**Takeaway / decision:** *no session with PageValues = 0 ever converts* (0 of 6,612), i.e. PageValues is, in this data, a near-deterministic proxy for the outcome and in a live shop it is only known late (it reflects pages near checkout). A model using it would look excellent and be useless for early intervention. **Decision: the deployed model excludes PageValues**; a "with PageValues" run is reported only as a labelled upper bound (notebook 03)."""),
code("""rows = []
for col in S.cols('int_coded_categorical'):
    g = df.groupby(col).Converted.agg(['mean','size'])
    g = g[g['size'] >= 100]
    rows.append({'column': col, 'levels(n>=100)': len(g), 'min rate': g['mean'].min(), 'max rate': g['mean'].max()})
pd.DataFrame(rows).round(3)"""),
md("""**Takeaway:** the four int-coded ID columns are *categories* (Region 3 is not "more" than Region 1) and show nearly flat conversion across levels, so they are one-hot encoded, never scaled, and are candidates for removal in feature selection."""),
code("""d = add_engineered_features(df)
print('sessions where BounceRates > ExitRates:', int((df.BounceRates > df.ExitRates).sum()),
      '(in Google Analytics a bounce is also an exit, so this is odd; kept as-is, flagged as a data-quality note)')
print('sessions with 0 pages of any type:', int((d.total_pages == 0).sum()))
print('Informational_Duration == 0 share:', round(float((df.Informational_Duration == 0).mean()), 3))"""),
md("""**Takeaway:** 463 sessions have BounceRates above ExitRates, which should not happen in real GA data; the dataset is evidently synthetic (the brief says "realistic"), so conclusions describe this dataset, not a real shop. Durations are heavy-tailed with many zeros, hence log1p scaling in the pipeline."""),
code("""# interactive chart (Plotly) - skipped automatically when plotly is not installed
try:
    import plotly.express as px
    fig = px.scatter(df.sample(3000, random_state=1), x='ProductRelated', y='ExitRates', color=df.sample(3000, random_state=1).Converted.astype(str),
                     opacity=.5, title='Product pages vs exit rate (interactive)', labels={'color': 'Converted'})
    fig.write_html(str(S.path('figures_dir') / 'eda_interactive.html'))
    print('saved reports/figures/eda_interactive.html')
except ImportError:
    print('plotly is not installed in this environment: interactive chart skipped (install plotly and re-run).')"""),
]

FEAT = [
md("""# 02 - Data preparation, feature engineering and selection
CRISP-DM phase 3. Preprocessing lives in `src/preprocessing/pipeline.py` (fit per CV fold, never on all data); this notebook shows the reasoning and the evidence."""),
code(SETUP),
code("""from src.preprocessing.pipeline import build_preprocessor
from src.models.trainer import build_registry, SessionClassifier
from src.models.evaluation import cross_validate
train, test = stratified_split(df, S)
y = train.Converted.to_numpy()
pre = build_preprocessor(S).fit(train)
print('input columns the model needs:', pre.input_columns)
print('expanded model features:', len(pre.feature_names_))
print('PageValues used:', 'PageValues' in pre.input_columns)"""),
md("""**Takeaway:** 16 raw inputs; PageValues deliberately absent. Categoricals (including OperatingSystems / Browser / Region / TrafficType) become one-hot columns; unseen categories at serving time map to all zeros."""),
code("""e = add_engineered_features(train)
eng = ['total_pages','total_duration','avg_time_per_page','product_page_share','time_per_product_page']
print(e[eng + ['Converted']].corr()['Converted'].drop('Converted').round(3))"""),
md("""**Takeaway:** the engineered ratios are easy to explain to a stakeholder (how deep, how long, how product-focused). Their value is judged by an ablation below, not assumed."""),
code("""spec = build_registry()['LogisticRegression']
import src.preprocessing.pipeline as P
base = P.LOG_COLS_BASE.copy(), P.PLAIN_COLS_BASE.copy()
def cv_with(logc, plainc):
    P.LOG_COLS_BASE[:] = logc; P.PLAIN_COLS_BASE[:] = plainc
    try:
        r = cross_validate(spec, S, train, y, 'none', {'C': 0.03})
    finally:
        P.LOG_COLS_BASE[:] = base[0]; P.PLAIN_COLS_BASE[:] = base[1]
    return r
raw_log = [c for c in base[0] if c not in eng]; raw_plain = [c for c in base[1] if c not in eng]
a = cv_with(base[0], base[1]); b = cv_with(raw_log, raw_plain)
pd.DataFrame({'features': ['raw + engineered', 'raw only'], 'CV PR-AUC': [a['pr_auc_mean'], b['pr_auc_mean']], 'std': [a['pr_auc_std'], b['pr_auc_std']],
              'CV ROC-AUC': [a['roc_auc_mean'], b['roc_auc_mean']]}).round(4)"""),
md("""**Takeaway:** the engineered ratios add 0.0006 CV PR-AUC (0.5313 vs 0.5307), far inside one fold-to-fold standard deviation (0.028). They are **not shown to improve prediction**; a logistic model on log-counts can already form most of these combinations. They are kept because they are easy to explain to a stakeholder, not because they win."""),
code("""pi = pd.DataFrame(json.load(open(S.path('model_dir') / 'metrics.json'))['permutation_importance'])
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(pi.feature[::-1], pi.mean_drop[::-1], xerr=pi.std_drop[::-1], color='#1F3A5F'); ax.set_xlabel('drop in PR-AUC when shuffled (champion, 5-fold)'); ax.set_title('Permutation importance')
plt.tight_layout(); plt.show()
pi.round(4)"""),
md("""**Takeaway:** ProductRelated (0.28) dwarfs everything; ExitRates (0.04) is second; everything else is at or below noise. Lens 2 (Shapley) is in notebook 04 and agrees on the ranking (Spearman 0.89)."""),
code("""fs = json.load(open(S.path('model_dir') / 'metrics.json'))['feature_selection']
print(json.dumps(fs, indent=1))"""),
md("""**Selection result:** inputs that do not earn their place (shuffling costs < 0.001 PR-AUC and hurts in at most half the folds) were removed, and the reduced set was kept only because CV PR-AUC did not drop. The change in CV PR-AUC is within noise: the claim is "no loss with fewer inputs", not "improvement". Note that Month is among the removed inputs even though raw conversion differs by month: once browsing behaviour is known, month adds nothing measurable."""),
]

MODEL = [
md("""# 03 - Modeling and evaluation
CRISP-DM phases 4-5. Training itself is `python -m src.train` (stratified 5-fold CV on the 80% training split, test split untouched until the end). This notebook reads the saved results and checks them."""),
code(SETUP),
code("""R = json.load(open(S.path('model_dir') / 'metrics.json'))
print('backend:', R['backend']); print('models trained:', R['models_available'])
print('champion:', R['champion'], '| rule:', R['champion_rule']['rule'])
print('best by raw mean PR-AUC would have been:', R['champion_by_max_pr_auc'])"""),
md("""**Important honesty notes.** (1) This build environment could not install XGBoost, LightGBM or CatBoost, so only **four** of the six required algorithms were trained here (LogisticRegression, DecisionTree, RandomForest, GradientBoosting). The trainer registers the other three automatically when the packages are installed: run `pip install -r requirements.txt` and `python -m src.train` to get all six. (2) The first run showed LogisticRegression and RandomForest tied to three decimals, so the one-standard-error rule (simplest model within 1 SE of the best) was adopted *after* seeing that; the raw-best alternative is reported above and its test numbers are in the table below."""),
code("""pd.DataFrame(R['imbalance_benchmark']).drop(columns=['seconds']).round(4)"""),
md("""**Takeaway:** the imbalance strategy matters less than expected. `none` / `class_weight` are within noise for every model, and **SMOTE is never better and often worse** (e.g. DecisionTree 0.429 vs 0.482). Plain threshold tuning on un-weighted probabilities is therefore used where it ties, and it keeps probabilities calibrated."""),
code("""comp = pd.DataFrame(R['comparison']).drop(columns=['params'])
comp['test_pr_auc'] = [R['test_all_models'][m]['pr_auc'] for m in comp.model]
comp['test_roc_auc'] = [R['test_all_models'][m]['roc_auc'] for m in comp.model]
comp.round(4)"""),
md("""**Takeaway:** all non-tree-stump models land within roughly one standard error of each other (CV PR-AUC 0.52-0.53, std 0.02-0.03); the single DecisionTree is clearly worse. GradientBoosting's Brier score (0.159) is much worse than the others (about 0.10) because it was trained with balanced class weights, which distorts its probabilities; do not use its raw probabilities for the value-based threshold."""),
code("""oof = pd.read_csv(S.path('reports_dir') / 'oof_predictions.csv')
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for c in [c for c in oof.columns if c.startswith('oof_') and c != 'oof_champion_selected']:
    pr, rc, _ = M.pr_curve(oof.y, oof[c]); ax[0].plot(rc, pr, label=c[4:])
    o = np.argsort(-oof[c].values); ys = oof.y.values[o]; ax[1].plot(np.cumsum(1-ys)/(1-ys).sum(), np.cumsum(ys)/ys.sum(), label=c[4:])
ax[0].axhline(oof.y.mean(), ls='--', c='grey'); ax[0].set(xlabel='recall', ylabel='precision', title='Out-of-fold PR curves (training data)'); ax[0].legend()
ax[1].plot([0,1],[0,1],'--',c='grey'); ax[1].set(xlabel='FPR', ylabel='TPR', title='Out-of-fold ROC'); plt.tight_layout(); plt.show()"""),
md("""## Decision threshold from business value (not 0.5)
Net value of flagging a session = value of a captured purchase x purchases flagged - cost per intervention x sessions flagged. **The value (40) and cost (4) are assumptions in `config/config.yaml`, not measured numbers**; replace them with the retailer's figures. The threshold is chosen on training out-of-fold predictions only."""),
code("""vc = pd.read_csv(S.path('reports_dir') / 'value_curve_oof.csv')
t = R['threshold']
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(vc.threshold, vc.net_value, c='#1F3A5F'); ax[0].axvline(t['threshold'], c='#E07A1F', ls='--'); ax[0].set(xlabel='threshold', ylabel='net value (training OOF)', title='Value-optimal threshold')
ax[1].plot(vc.threshold, vc.precision, label='precision'); ax[1].plot(vc.threshold, vc.recall, label='recall'); ax[1].axvline(t['threshold'], c='#E07A1F', ls='--'); ax[1].legend(); ax[1].set(xlabel='threshold')
plt.tight_layout(); plt.show()
print('chosen threshold', t['threshold'], '| precision %.3f recall %.3f flagged %d' % (t['precision'], t['recall'], t['flagged']))
pd.DataFrame(R['threshold_sensitivity']).round(3)"""),
md("""**Takeaway:** with value:cost = 10:1 the best threshold is about 0.125 (flag ~40% of sessions, recall 0.82, precision 0.32). The threshold moves from 0.47 to 0.02 as the value:cost ratio goes from 2 to 40 - the model is only half the answer; the business's numbers decide the operating point."""),
md("""## One evaluation on the untouched test set"""),
code("""T = R['test']
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in T.items() if k not in ('roc_auc_ci95', 'pr_auc_ci95')})
print('ROC-AUC 95% CI', np.round(T['roc_auc_ci95'], 3), '| PR-AUC 95% CI', np.round(T['pr_auc_ci95'], 3))
print('baselines:', json.dumps(R['test_baselines'], indent=1))"""),
code("""from IPython.display import Image
import matplotlib.image as mpimg
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].imshow(mpimg.imread(S.path('figures_dir') / 'test_pr_roc.png')); ax[0].axis('off')
ax[1].imshow(mpimg.imread(S.path('figures_dir') / 'test_calibration.png')); ax[1].axis('off')
plt.tight_layout(); plt.show()""") if False else code("""fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
import matplotlib.image as mpimg
ax[0].imshow(mpimg.imread(S.path('figures_dir') / 'test_pr_roc.png')); ax[0].axis('off')
ax[1].imshow(mpimg.imread(S.path('figures_dir') / 'test_calibration.png')); ax[1].axis('off')
plt.tight_layout(); plt.show()"""),
md("""**Takeaway (read with the baselines above):** test PR-AUC 0.558 (CI 0.51-0.61) and ROC-AUC 0.837. The model is clearly better than chance (base rate 0.157) and than "always no purchase", but **a single feature, ProductRelated, already gives a test PR-AUC of 0.531**: the full pipeline adds only about 0.03 PR-AUC over a one-line rule. Be careful not to oversell it.

At the chosen threshold the test confusion matrix is TP 306 / FP 663 / FN 71 / TN 1360. Net value on the test set is 8,364 versus 5,480 for flagging everyone and 0 for flagging nobody (under the assumed value and cost); flagging everyone is a strong baseline here because purchases are valuable relative to the intervention cost."""),
code("""ref = R['test_with_page_values_reference']
print('WITH PageValues (reference only, same recipe):', {k: round(v, 3) for k, v in ref.items() if isinstance(v, float)})
print('WITHOUT PageValues (deployed):                 ROC-AUC %.3f  PR-AUC %.3f' % (T['roc_auc'], T['pr_auc']))"""),
md("""**Takeaway:** adding PageValues lifts test PR-AUC from 0.56 to 0.85 and ROC-AUC from 0.84 to 0.97. That jump is exactly why it must not be used for early-session intervention (see notebook 01); the 0.85 would be a deployment-leakage number. It is shown to quantify the trap, not as a result.

**Limits to keep in mind:** (a) the data holds only whole-session totals, so "score during the session" is not validated here, only "score using final session behaviour"; (b) the value/cost figures are assumed; (c) the data looks synthetic (bounce > exit in 463 rows), so the numbers describe this dataset."""),
]

XAI = [
md("""# 04 - Explainability (SHAP-style global + local, LIME-style local)
CRISP-DM phases 5-6. **What these methods are:** the `shap` and `lime` packages could not be installed in the build environment, so `src/explain/` contains (a) a model-agnostic **permutation-sampling Shapley** explainer (the estimator behind SHAP for any model; contributions sum exactly to f(x) minus the baseline, verified in tests) and (b) a **LIME-style** local surrogate whose interpretable representation is "feature keeps its value / is replaced by a background value". If `shap`/`lime` are installed, `src/explain/shapley.py::shap_tree_values` gives an independent check. Explanations describe the model, not causes: a high contribution does not mean changing that behaviour will change the purchase."""),
code(SETUP),
code("""from src.serving.predictor import PurchasePredictor
E = json.load(open(S.path('reports_dir') / 'explain_results.json'))
pred = PurchasePredictor.load(settings=S)
print('model:', pred.meta['model'], '| features explained:', len(pred.features))
g = pd.DataFrame(E['global']); g.round(4).head(10)"""),
code("""import matplotlib.image as mpimg
plt.figure(figsize=(8, 5)); plt.imshow(mpimg.imread(S.path('figures_dir') / 'shapley_global.png')); plt.axis('off'); plt.show()"""),
md("""**Takeaway:** ProductRelated (mean |effect| about 10 percentage points of purchase probability) and ExitRates (about 4 points) account for almost all of the model's behaviour; everything else is 1-1.5 points or less. Ranking agreement with permutation importance (notebook 02): Spearman 0.89."""),
code("""rows = []
for name, v in E['local'].items():
    print(f"--- {name}: model P(purchase) = {v['probability']:.3f}, baseline {v['base']:.3f}, actually converted: {v['actual_converted']}, Shapley efficiency gap {v['efficiency_gap']:.1e}")
    print('  Shapley :', {k: round(x, 3) for k, x in v['shapley'].items()})
    print('  LIME    :', {k: round(x, 3) for k, x in list(v['lime'].items())[:5]}, f"(local fit R2 {v['lime_r2']:.2f})")"""),
md("""**Takeaway:** for the highest-scored session ProductRelated alone adds about +57 points; for the lowest it subtracts about 11 points together with ExitRates (-6). The two methods name the same top feature, but check LIME's local R2: when it is low (0.2-0.3 for two of the three examples) the linear surrogate fits the model poorly and the LIME numbers should be treated as a rough second opinion, not a measurement."""),
code("""print('Shapley vs LIME-style over', E['shapley_vs_lime']['n'], 'sessions:', {k: round(v, 3) for k, v in E['shapley_vs_lime'].items() if k != 'n'})
print('serving latency (ms): with explanation %.1f | score only %.1f' % (E['latency_ms_with_explanation'], E['latency_ms_score_only']))"""),
md("""**Takeaway:** the two lenses agree on the single most important feature in 92% of sessions but only moderately on the full ordering (mean rank correlation 0.66), because most of the 10 features the final model uses have near-zero effects whose order is noise. Scoring with an explanation costs about 16 ms on this machine (the Shapley explainer is built once at startup, not per request)."""),
]

ALL = {"01_eda": EDA, "02_features": FEAT, "03_modeling": MODEL, "04_xai": XAI}
