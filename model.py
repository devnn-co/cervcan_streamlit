"""Shared data loading and model definition for the cervical cancer risk project."""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).parent / "kag_risk_factors_cervical_cancer.csv"

# Target: a confirmed biopsy result is the closest thing to ground truth in the dataset.
TARGET = "Biopsy"

# Only self-reported risk factors are used as features. Columns that are themselves
# diagnoses or screening results (Dx:*, Hinselmann, Schiller, Citology) are excluded,
# because they leak the answer into the model.
FEATURES = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes (years)",
    "Hormonal Contraceptives (years)",
    "IUD (years)",
    "STDs (number)",
]


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset, treating '?' as missing, and return features and target."""
    df = pd.read_csv(path, na_values="?")
    return df[FEATURES], df[TARGET].astype(int)


def build_model(random_state: int = 42) -> Pipeline:
    """Median imputation followed by a shallow, class-balanced decision tree.

    Imputation lives inside the pipeline so it is fitted on training data only.
    class_weight="balanced" compensates for the heavy class imbalance (~6% positive).
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "tree",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model() -> Pipeline:
    """Fit the model on the full dataset."""
    X, y = load_data()
    return build_model().fit(X, y)
