"""Evaluate the frozen candidate model against the held-out test dataset."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.train_baseline import (
    CANDIDATE_FEATURE_COLUMNS,
    CANDIDATE_MODEL_NAME,
)

from src.predict_domain import load_candidate_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TEST_PATH = PROCESSED_DATA_DIR / "domain_risk_test_v0_1.csv"

def evaluate_candidate() -> dict:
    """Evaluate the frozen candidate artifact on the completed held-out test data."""

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df[CANDIDATE_FEATURE_COLUMNS]
    y_test = test_df["label"]

    model = load_candidate_model()

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "model": CANDIDATE_MODEL_NAME,
        "test_observations": len(test_df),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


if __name__ == "__main__":
    results = evaluate_candidate()

    print("Frozen candidate held-out evaluation")
    print(f"Model: {results['model']}")
    print(f"Test observations: {results['test_observations']:,}")
    print()
    print(f"Accuracy:  {results['accuracy']:.6f}")
    print(f"Precision: {results['precision']:.6f}")
    print(f"Recall:    {results['recall']:.6f}")
    print(f"F1:        {results['f1']:.6f}")
    print(f"ROC-AUC:   {results['roc_auc']:.6f}")