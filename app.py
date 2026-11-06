"""Streamlit app: an educational demo of a cervical cancer risk-factor model.

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from model import FEATURES, train_model

ROOT = Path(__file__).parent


@st.cache_resource
def get_model():
    # The dataset is small, so training takes about a second. Training here (once per
    # server process) avoids version mismatches from committing a pickled model.
    return train_model()


def load_metrics() -> dict | None:
    path = ROOT / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def header() -> None:
    st.title("Cervical Cancer Risk Factors: A Machine Learning Demo")
    st.write(
        "This app uses a decision tree trained on the UCI *Cervical Cancer (Risk Factors)* "
        "dataset to estimate whether a patient's answers resemble those of patients with a "
        "positive biopsy."
    )
    st.warning(
        "**This is a student project, not a medical tool.** The model is weak (see the "
        "results below) and must not be used to make health decisions. If you have any "
        "concerns, speak to a doctor about cervical screening."
    )


def user_input() -> pd.DataFrame:
    st.header("Your answers")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=13, max_value=100, value=30)
        partners = st.number_input("Number of sexual partners", min_value=0, max_value=50, value=2)
        first = st.number_input("Age at first sexual intercourse", min_value=10, max_value=60, value=17)
        pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=0)
    with col2:
        smoke_years = st.number_input("Years smoking (0 if never)", min_value=0.0, max_value=80.0, value=0.0, step=1.0)
        hc_years = st.number_input("Years using hormonal contraceptives (0 if never)", min_value=0.0, max_value=40.0, value=0.0, step=1.0)
        iud_years = st.number_input("Years using an IUD (0 if never)", min_value=0.0, max_value=40.0, value=0.0, step=1.0)
        stds = st.number_input("Number of STDs ever diagnosed", min_value=0, max_value=10, value=0)

    values = [age, partners, first, pregnancies, smoke_years, hc_years, iud_years, stds]
    # Build a DataFrame with the training column names, so features can never be
    # passed to the model in the wrong order.
    return pd.DataFrame([values], columns=FEATURES)


def show_prediction(model, features: pd.DataFrame) -> None:
    st.header("Model output")
    if model.predict(features)[0] == 1:
        st.info(
            "Your answers are **more similar** to patients in the dataset who had a positive "
            "biopsy. This is not a diagnosis."
        )
    else:
        st.info(
            "Your answers are **less similar** to patients in the dataset who had a positive "
            "biopsy. This is not a diagnosis, and regular screening is still recommended."
        )


def show_results(metrics: dict | None) -> None:
    st.header("How good is the model?")
    if metrics is None:
        st.write("Run `python train.py` to generate evaluation metrics.")
        return
    tree, base = metrics["decision_tree"], metrics["baseline_stratified_random"]
    st.write(
        f"Evaluated with 5-fold stratified cross-validation on {metrics['n_samples']} "
        f"patients, of whom {metrics['n_positive']} had a positive biopsy."
    )
    rows = {
        "ROC-AUC": "roc_auc",
        "Recall (positives found)": "recall",
        "Precision": "precision",
        "Balanced accuracy": "balanced_accuracy",
    }
    st.table(
        pd.DataFrame(
            {
                "Decision tree": [f"{tree[k]['mean']:.2f} ± {tree[k]['std']:.2f}" for k in rows.values()],
                "Random-guess baseline": [f"{base[k]['mean']:.2f}" for k in rows.values()],
            },
            index=list(rows.keys()),
        )
    )
    st.write(
        "Self-reported risk factors alone carry only a weak signal: the model does better than "
        "random guessing, but misses most positive cases."
    )
    tree_png = ROOT / "figures" / "tree.png"
    if tree_png.exists():
        with st.expander("See the decision tree"):
            st.image(str(tree_png))


def main() -> None:
    header()
    model = get_model()
    features = user_input()
    show_prediction(model, features)
    show_results(load_metrics())


main()
