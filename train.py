"""Evaluate the model with stratified cross-validation, then train on all data.

Usage:
    python train.py

Outputs:
    model.joblib          trained pipeline (used by app.py)
    metrics.json          cross-validated metrics for the model and a baseline
    figures/tree.png      plot of the fitted decision tree
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import plot_tree

from model import FEATURES, build_model, load_data

ROOT = Path(__file__).parent
SCORING = ["roc_auc", "recall", "precision", "balanced_accuracy", "accuracy"]


def evaluate(estimator, X, y, cv) -> dict:
    scores = cross_validate(estimator, X, y, cv=cv, scoring=SCORING)
    return {
        m: {
            "mean": round(float(scores[f"test_{m}"].mean()), 3),
            "std": round(float(scores[f"test_{m}"].std()), 3),
        }
        for m in SCORING
    }


def main() -> None:
    X, y = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "features": FEATURES,
        "decision_tree": evaluate(build_model(), X, y, cv),
        # Baseline that guesses classes at random in proportion to their frequency.
        "baseline_stratified_random": evaluate(
            DummyClassifier(strategy="stratified", random_state=42), X, y, cv
        ),
    }

    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    model = build_model().fit(X, y)
    dump(model, ROOT / "model.joblib")

    (ROOT / "figures").mkdir(exist_ok=True)
    plt.figure(figsize=(20, 10))
    plot_tree(
        model.named_steps["tree"],
        feature_names=FEATURES,
        class_names=["Negative biopsy", "Positive biopsy"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "tree.png", dpi=120)
    print("Saved model.joblib, metrics.json and figures/tree.png")


if __name__ == "__main__":
    main()
