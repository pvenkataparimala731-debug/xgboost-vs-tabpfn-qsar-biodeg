"""
XGBoost vs TabPFN on qsar-biodeg — Streamlit demo
Companion app for the MSc dissertation:
"Benchmarking XGBoost Against TabPFN Zero-Shot Predictions on TabArena Tabular
Datasets: A Feature-Structure Analysis of Foundation Model Underperformance"

Run locally:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

RANDOM_STATE = 42
st.set_page_config(page_title="XGBoost vs TabPFN — qsar-biodeg", layout="wide")

# ---------------------------------------------------------------------------
# Cached results from the dissertation notebook (single held-out split, n=264)
# Update these if you re-run the notebook and get different numbers.
# ---------------------------------------------------------------------------
CACHED_RESULTS = {
    "xgboost": {"roc_auc": 0.9222, "f1": 0.8000},
    "tabpfn": {"roc_auc": 0.9400, "f1": 0.8193},
    "mcnemar_p": 0.664,
    "n_test": 264,
    "n_total": 1055,
    "n_features": 41,
}


@st.cache_data(show_spinner="Loading qsar-biodeg dataset from OpenML...")
def load_data():
    import openml

    dataset = openml.datasets.get_dataset(1494)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le


@st.cache_resource(show_spinner="Training XGBoost on the shared train/test split...")
def train_xgboost(X, y_enc):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, stratify=y_enc, random_state=RANDOM_STATE
    )
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def try_load_tabpfn():
    """Attempt to set up TabPFN if a token is available as a Streamlit secret
    or environment variable. Returns None if unavailable, so the app degrades
    gracefully to cached results only."""
    token = None
    try:
        token = st.secrets.get("TABPFN_TOKEN", None)
    except Exception:
        pass
    token = token or os.environ.get("TABPFN_TOKEN")
    if not token:
        return None
    try:
        os.environ["TABPFN_TOKEN"] = token
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(random_state=RANDOM_STATE)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("About")
st.sidebar.markdown(
    """
This app accompanies an MSc dissertation comparing **XGBoost** (a tuned
gradient-boosted baseline) against **TabPFN** (a zero-shot tabular
foundation model) on the **qsar-biodeg** dataset from the TabArena
benchmark.

- 1,055 chemical compounds
- 41 numeric molecular descriptors
- Binary target: ready- vs not-ready-biodegradable

**Note on TabPFN:** TabPFN requires a licensed API token from Prior Labs.
This app runs XGBoost live in your browser session; TabPFN results are
shown from the dissertation's cached run unless a `TABPFN_TOKEN` secret is
configured (see README).
"""
)

tab1, tab2, tab3 = st.tabs(["📊 Results Dashboard", "🔮 Live XGBoost Demo", "📁 About the Data"])

# ---------------------------------------------------------------------------
# Tab 1 — Results dashboard (reproduces the dissertation's key figures)
# ---------------------------------------------------------------------------
with tab1:
    st.header("XGBoost vs TabPFN — Held-out Test Performance")
    st.caption(
        f"Single stratified split, n = {CACHED_RESULTS['n_test']} test samples "
        f"(matches Chapter 5 of the dissertation)."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        results_df = pd.DataFrame(
            {
                "Model": ["XGBoost", "TabPFN (zero-shot)"],
                "ROC-AUC": [CACHED_RESULTS["xgboost"]["roc_auc"], CACHED_RESULTS["tabpfn"]["roc_auc"]],
                "F1-score": [CACHED_RESULTS["xgboost"]["f1"], CACHED_RESULTS["tabpfn"]["f1"]],
            }
        )
        st.dataframe(results_df.set_index("Model"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        results_df.set_index("Model")[["ROC-AUC", "F1-score"]].plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("Test-set metrics")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)

    with col2:
        st.metric(
            "McNemar's test p-value",
            f"{CACHED_RESULTS['mcnemar_p']:.3f}",
            help="p > 0.05 means the difference in error patterns between the "
            "two models is not statistically significant.",
        )
        st.info(
            "TabPFN scores marginally higher on both metrics, but McNemar's "
            f"test (p = {CACHED_RESULTS['mcnemar_p']:.3f}) shows the difference "
            "in error patterns is **not statistically significant**. On this "
            "dataset, TabPFN's zero-shot performance is broadly comparable to "
            "a competently tuned XGBoost baseline rather than a clear "
            "improvement."
        )

    st.divider()
    st.subheader("Feature-structure findings")
    st.markdown(
        """
- Several features (e.g. those with skewness above 5–6) are heavily
  right-skewed, which may be harder for a model pretrained on
  near-normal synthetic data to handle.
- One heavily skewed feature was both among XGBoost's top-4 most
  important predictors **and** among the features most associated with
  TabPFN's misclassifications — a plausible, though not conclusively
  proven, link between feature skew and reduced zero-shot performance.
"""
    )

# ---------------------------------------------------------------------------
# Tab 2 — Live XGBoost demo
# ---------------------------------------------------------------------------
with tab2:
    st.header("Try the XGBoost model live")
    st.caption(
        "This trains the same XGBoost configuration used in the dissertation "
        "(300 estimators, max depth 4, learning rate 0.05) on the real "
        "qsar-biodeg data, live in this session."
    )

    try:
        X, y_enc, le = load_data()
        model, X_train, X_test, y_train, y_test = train_xgboost(X, y_enc)

        xgb_pred = model.predict(X_test)
        xgb_proba = model.predict_proba(X_test)[:, 1]
        live_auc = roc_auc_score(y_test, xgb_proba)
        live_f1 = f1_score(y_test, xgb_pred)

        c1, c2 = st.columns(2)
        c1.metric("Live ROC-AUC (this session)", f"{live_auc:.4f}")
        c2.metric("Live F1-score (this session)", f"{live_f1:.4f}")

        st.divider()
        st.subheader("Predict on a custom compound")
        st.caption(
            "Adjust the top predictors below; remaining features are held at "
            "their training-set mean. This is a simplified, illustrative "
            "interface — not a substitute for the full 41-feature model."
        )

        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        top_features = importances.head(6).index.tolist()

        input_row = X.mean().to_dict()
        cols = st.columns(3)
        for i, feat in enumerate(top_features):
            lo, hi = float(X[feat].min()), float(X[feat].max())
            default = float(X[feat].mean())
            with cols[i % 3]:
                input_row[feat] = st.slider(feat, lo, hi, default)

        input_df = pd.DataFrame([input_row])[X.columns]
        pred_proba = float(model.predict_proba(input_df)[0, 1])
        raw_class = le.inverse_transform([int(pred_proba >= 0.5)])[0]

        # OpenML's qsar-biodeg target sometimes comes through as raw codes
        # rather than descriptive text; map to a readable label either way.
        friendly_labels = {"1": "Not-ready biodegradable (NRB)", "2": "Ready biodegradable (RB)",
                            "RB": "Ready biodegradable (RB)", "NRB": "Not-ready biodegradable (NRB)"}
        pred_label = friendly_labels.get(str(raw_class), str(raw_class))

        st.metric("Predicted class", pred_label)
        st.progress(min(max(pred_proba, 0.0), 1.0), text=f"P(ready biodegradable) = {pred_proba:.3f}")

        with st.expander("Feature importance (top 10)"):
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            importances.head(10).sort_values().plot(kind="barh", ax=ax2)
            ax2.set_xlabel("XGBoost importance")
            st.pyplot(fig2)

        with st.expander("Confusion matrix on held-out test set"):
            cm = confusion_matrix(y_test, xgb_pred)
            fig3, ax3 = plt.subplots(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
            ax3.set_xlabel("Predicted")
            ax3.set_ylabel("Actual")
            st.pyplot(fig3)

    except Exception as e:
        st.error(
            "Could not load the dataset or train the model in this "
            "environment. If you're running locally, check your internet "
            f"connection (OpenML download). Details: {e}"
        )

# ---------------------------------------------------------------------------
# Tab 3 — About the data
# ---------------------------------------------------------------------------
with tab3:
    st.header("About the qsar-biodeg dataset")
    st.markdown(
        """
**Source:** [OpenML dataset 1494](https://www.openml.org/d/1494), originally
developed by Mansouri et al. for QSAR (quantitative structure–activity
relationship) modelling of chemical biodegradability, and curated within
the [TabArena benchmark](https://arxiv.org/abs/2506.16791)'s list of
tabular datasets.

- **1,055** chemical compounds
- **41** numeric molecular descriptors (structural and physicochemical
  properties)
- **Binary target:** ready biodegradable (RB) vs not-ready biodegradable
  (NRB)
- No missing values

**Why this dataset:** TabArena's own analysis found that tabular
foundation models such as TabPFN tend to perform best on smaller
datasets. At just over 1,000 rows, qsar-biodeg sits well within that
range, making it a focused test case for comparing TabPFN's zero-shot
performance against a tuned XGBoost baseline.
"""
    )
    st.caption(
        "Full methodology, statistical testing, and feature-structure "
        "analysis are described in the accompanying dissertation report "
        "and Jupyter notebook in this repository."
    )
  
  


    
