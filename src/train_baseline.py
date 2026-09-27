"""Train the initial lexical domain-risk baseline model."""

from pathlib import Path

import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CANDIDATE_MODEL_PATH = MODELS_DIR / "scaled_logistic_v0_1.joblib"

TRAIN_PATH = PROCESSED_DATA_DIR / "domain_risk_train_v0_1.csv"

RANDOM_SEED = 42
VALIDATION_FOLDS = 5

CANDIDATE_MODEL_NAME = "scaled_logistic_v0_1"

FEATURE_COLUMNS = [
    "hostname_length",
    "digit_count",
    "digit_ratio",
    "hyphen_count",
    "hostname_entropy",
    "alphabetic_ratio",
]

CANDIDATE_FEATURE_COLUMNS = FEATURE_COLUMNS.copy()

REDUCED_FEATURE_COLUMNS = [
    "hostname_length",
    "digit_ratio",
    "hyphen_count",
    "hostname_entropy",
]


def load_training_data(
    train_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the training partition and separate features from the target."""

    df = pd.read_csv(train_path)

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    return X, y


def build_baseline_model() -> LogisticRegression:
    """Return the initial interpretable baseline classifier."""

    return LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=1000,
    )


def build_scaled_logistic_model() -> Pipeline:
    """Return a standardized logistic-regression comparison model."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=RANDOM_SEED,
                    max_iter=1000,
                ),
            ),
        ]
    )


def save_candidate_model(
    model,
    model_path: Path = CANDIDATE_MODEL_PATH,
) -> None:
    """Persist the fitted frozen candidate pipeline."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def build_random_forest_model() -> RandomForestClassifier:
    """Return the initial Random Forest comparison classifier."""

    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def train_baseline_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> LogisticRegression:
    """Fit the baseline classifier using the approved training features."""

    model = build_baseline_model()
    model.fit(X, y)

    return model


def get_feature_coefficients(
    model: LogisticRegression,
) -> pd.DataFrame:
    """Return model coefficients alongside their feature names."""

    return pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": model.coef_[0],
        }
    ).sort_values(
        "coefficient",
        ascending=False,
    )


def create_validation_splitter() -> StratifiedGroupKFold:
    """Return the group-aware cross-validation splitter."""

    return StratifiedGroupKFold(
        n_splits=VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )


def cross_validate_baseline(
    df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    model_builder=build_baseline_model,
) -> pd.DataFrame:
    """Evaluate the baseline using group-aware cross-validation."""

    X = df[feature_columns]
    y = df["label"]
    groups = df["registrable_domain"]

    splitter = create_validation_splitter()

    results = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]

        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]

        model = model_builder()
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        probabilities = model.predict_proba(X_validation)[:, 1]

        results.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y_validation, predictions),
                "precision": precision_score(y_validation, predictions),
                "recall": recall_score(y_validation, predictions),
                "f1": f1_score(y_validation, predictions),
                "roc_auc": roc_auc_score(y_validation, probabilities),
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    X_train, y_train = load_training_data(TRAIN_PATH)

    model = train_baseline_model(
        X_train,
        y_train,
    )

    coefficients = get_feature_coefficients(model)

    print("Baseline logistic regression trained")
    print(f"Training observations: {len(X_train):,}")
    print(f"Predictive features: {len(FEATURE_COLUMNS)}")
    print()
    print("Feature coefficients:")
    print(coefficients.to_string(index=False))


    training_df = pd.read_csv(TRAIN_PATH)
    validation_results = cross_validate_baseline(training_df)

    print()
    print("5-fold group-aware cross-validation:")
    print(validation_results.to_string(index=False))

    print()
    print("Mean validation metrics:")
    print(
        validation_results[
            ["accuracy", "precision", "recall", "f1", "roc_auc"]
        ].mean().to_string()
    )
