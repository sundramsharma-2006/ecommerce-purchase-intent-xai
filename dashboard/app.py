"""Streamlit dashboard:  streamlit run dashboard/app.py

Six sections: overview, live scoring, funnel/cohorts, model comparison, explainability, system.
Loads the saved model (never retrains). Written to the Streamlit + Plotly APIs; it could not be rendered in the
build sandbox (neither package installable there) - the data layer in dashboard/services.py is tested.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import services as S  # noqa: E402
from src.config import load_settings  # noqa: E402

st.set_page_config(page_title="Purchase-Intent Dashboard", layout="wide")
settings = load_settings()


@st.cache_data
def get_sessions():
    return S.sessions(settings)


@st.cache_resource
def get_predictor():
    return S.load_predictor()


@st.cache_data
def get_metrics():
    return S.metrics_summary()


df = get_sessions()
section = st.sidebar.radio("Section", ["Overview", "Live session scoring", "Funnel & cohorts", "Model comparison",
                                       "Explainability", "System & metrics"])
st.sidebar.caption("Behavioural data is sensitive: this model must not be tied to personal identifiers.")

if section == "Overview":
    k = S.kpis(df)
    st.title("E-commerce purchase-intent prediction")
    c = st.columns(4)
    c[0].metric("Sessions", f"{k['sessions']:,}")
    c[1].metric("Purchases", f"{k['purchases']:,}")
    c[2].metric("Conversion rate", f"{k['conversion_rate']:.1%}")
    c[3].metric("Avg product pages", f"{k['avg_product_pages']:.1f}")
    st.plotly_chart(px.bar(S.month_order(df), x="Month", y="conversion_rate", title="Conversion rate by month"), use_container_width=True)

elif section == "Live session scoring":
    st.header("Score a session")
    pred = get_predictor()
    with st.form("session"):
        a, b, c3 = st.columns(3)
        adm = a.number_input("Administrative pages", 0, 200, 2)
        adm_d = a.number_input("Administrative seconds", 0.0, 5000.0, 40.0)
        inf = b.number_input("Informational pages", 0, 200, 0)
        inf_d = b.number_input("Informational seconds", 0.0, 5000.0, 0.0)
        prod = c3.number_input("Product pages", 0, 2000, 25)
        prod_d = c3.number_input("Product seconds", 0.0, 20000.0, 900.0)
        br = a.slider("BounceRates", 0.0, 0.2, 0.01)
        er = b.slider("ExitRates", 0.0, 0.3, 0.03)
        sd = c3.select_slider("SpecialDay", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        month = a.selectbox("Month", settings.cols("months"), index=10)
        vt = b.selectbox("VisitorType", settings.cols("visitor_types"))
        wk = c3.checkbox("Weekend")
        osy = a.number_input("OperatingSystems (ID)", 1, 20, 2)
        brw = b.number_input("Browser (ID)", 1, 30, 2)
        reg = c3.number_input("Region (ID)", 1, 20, 1)
        tt = a.number_input("TrafficType (ID)", 1, 30, 2)
        go = st.form_submit_button("Score")
    if go:
        res = pred.predict({"Administrative": adm, "Administrative_Duration": adm_d, "Informational": inf,
                            "Informational_Duration": inf_d, "ProductRelated": prod, "ProductRelated_Duration": prod_d,
                            "BounceRates": br, "ExitRates": er, "SpecialDay": sd, "Month": month, "OperatingSystems": osy,
                            "Browser": brw, "Region": reg, "TrafficType": tt, "VisitorType": vt, "Weekend": wk})
        st.metric("Purchase probability", f"{res['probability']:.1%}", help=f"Decision threshold {res['threshold']:.2f}")
        st.write(f"Decision: **{res['decision']}** | confidence: **{res['confidence']}** (distance from the threshold)")
        contrib = pd.DataFrame(res["top_contributors"])
        st.plotly_chart(px.bar(contrib, x="effect", y="feature", orientation="h", color="direction",
                               title=f"What moved this score (baseline {res['baseline_probability']:.1%})"), use_container_width=True)

elif section == "Funnel & cohorts":
    st.header("Funnel and cohorts")
    st.dataframe(S.funnel(df))
    by = st.selectbox("Cohort by", ["VisitorType", "Month", "Weekend", "TrafficType", "Region", "Browser"])
    st.plotly_chart(px.bar(S.cohort(df, by), x=by, y="conversion_rate", hover_data=["sessions"]), use_container_width=True)

elif section == "Model comparison":
    m = get_metrics()
    st.header("Model comparison (5-fold CV on training data)")
    comp = pd.DataFrame(m["comparison"]).drop(columns=["params"])
    st.dataframe(comp)
    st.plotly_chart(px.bar(comp, x="model", y="pr_auc", error_y="pr_auc_std", title="CV PR-AUC (higher is better)"), use_container_width=True)
    st.caption(f"Champion: {m['champion']} | backend: {m['backend']}")

elif section == "Explainability":
    st.header("Global explanations")
    m = get_metrics()
    pi = pd.DataFrame(m["permutation_importance"])
    st.plotly_chart(px.bar(pi, x="mean_drop", y="feature", orientation="h", title="Permutation importance (drop in PR-AUC)"), use_container_width=True)
    fig = Path(settings.path("figures_dir")) / "shapley_global.png"
    if fig.exists():
        st.image(str(fig), caption="Mean |Shapley| per feature (probability units)")

else:
    m = get_metrics()
    st.header("System & metrics")
    t = m["test"]
    c = st.columns(4)
    c[0].metric("Test PR-AUC", f"{t['pr_auc']:.3f}")
    c[1].metric("Test ROC-AUC", f"{t['roc_auc']:.3f}")
    c[2].metric("Recall @ threshold", f"{t['recall']:.2f}")
    c[3].metric("Precision @ threshold", f"{t['precision']:.2f}")
    st.dataframe(pd.DataFrame(m["threshold_sensitivity"]))
    st.json(m["metadata"])
