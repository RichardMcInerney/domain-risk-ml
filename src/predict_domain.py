"""Run lexical domain-risk inference using the frozen candidate model."""

from pathlib import Path
import ipaddress
import re

import joblib
import pandas as pd



from src.train_baseline import CANDIDATE_FEATURE_COLUMNS
from src.build_features import build_feature_record

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "scaled_logistic_v0_1.joblib"

HOSTNAME_LABEL_PATTERN = re.compile(
    r"^(?!-)[a-z0-9-]{1,63}(?<!-)$"
)

def load_candidate_model(model_path: Path = MODEL_PATH):
    """Load the frozen candidate model pipeline."""

    return joblib.load(model_path)


def validate_hostname(hostname: str) -> str:
    """Validate and normalize a hostname supplied for inference."""

    if not isinstance(hostname, str):
        raise TypeError("hostname must be a string")

    hostname = hostname.strip().lower()

    if not hostname:
        raise ValueError("hostname must not be empty")

    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ValueError("IP addresses are outside the model input boundary")

    labels = hostname.split(".")

    if any(not label for label in labels):
        raise ValueError("hostname contains an empty label")

    if any(
        HOSTNAME_LABEL_PATTERN.fullmatch(label) is None
        for label in labels
    ):
        raise ValueError("hostname contains an invalid label")

    if len(hostname) > 253:
        raise ValueError("hostname exceeds 253 characters")

    return hostname


def build_inference_features(hostname: str) -> pd.DataFrame:
    """Build the ordered model feature row for one hostname."""

    hostname = validate_hostname(hostname)

    feature_record = build_feature_record(hostname)

    return pd.DataFrame(
        [feature_record],
        columns=CANDIDATE_FEATURE_COLUMNS,
    )


def predict_domain_probability(hostname: str) -> float:
    """Return the model probability associated with the URLhaus class."""

    model = load_candidate_model()
    features = build_inference_features(hostname)

    probability = model.predict_proba(features)[0, 1]

    return float(probability)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m src.predict_domain <hostname>"
        )

    hostname = sys.argv[1]
    probability = predict_domain_probability(hostname)

    print(f"Hostname: {validate_hostname(hostname)}")
    print(f"URLhaus-class probability: {probability:.6f}")