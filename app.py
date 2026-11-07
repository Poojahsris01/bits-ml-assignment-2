import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Credit Default Risk Lab", layout="wide")

TARGET = "default_next_month"
FEATURES = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
            + [f"PAY_{i}" for i in range(1, 7)]
            + [f"BILL_AMT{i}" for i in range(1, 7)]
            + [f"PAY_AMT{i}" for i in range(1, 7)])

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(model, X, y):
    pred = model.predict(X)
    auc = roc_auc_score(y, get_scores(model, X)) if y.nunique() > 1 else float("nan")
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


st.title("Credit Card Default Risk Lab")
st.caption("Compare five classifiers on the UCI Taiwan credit default data. "
           "Models are pre-trained and are not retrained on your upload.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    st.markdown("The CSV must contain all 23 feature columns plus "
                f"`{TARGET}`.")

if uploaded is None:
    st.info("Upload `test_data.csv` from the repository to begin.")
    st.stop()

data = pd.read_csv(uploaded)
missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

X, y = data[FEATURES], data[TARGET]
st.write(f"Loaded **{len(data):,}** rows. "
         f"Default rate in this file: **{y.mean():.1%}**")

tab1, tab2 = st.tabs(["Selected model", "All models compared"])

with tab1:
    model = load_model(MODEL_FILES[model_name])
    pred, metrics = compute_metrics(model, X, y)

    st.subheader(model_name)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No default", "Default"],
                    yticklabels=["No default", "Default"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3))

with tab2:
    rows = {}
    for name, path in MODEL_FILES.items():
        _, m = compute_metrics(load_model(path), X, y)
        rows[name] = m
    st.dataframe(pd.DataFrame(rows).T.round(4), use_container_width=True)