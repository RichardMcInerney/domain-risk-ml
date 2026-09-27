"""
Exploratory data analysis for Zaxonite Domain Risk ML.

Produces reproducible descriptive analysis and visualisations of the
prepared domain-risk dataset. EDA outputs describe the dataset and are
not used to tune the frozen v0.1 candidate model.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
)

from src.train_baseline import (
    CANDIDATE_FEATURE_COLUMNS,
    TRAIN_PATH,
    build_baseline_model,
    build_random_forest_model,
    build_scaled_logistic_model,
    cross_validate_baseline,
)

from src.evaluate_candidate import TEST_PATH, evaluate_candidate
from src.predict_domain import load_candidate_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORT_DATA_DIR = REPORTS_DIR / "data"

FEATURE_DATA_PATH = (
    PROCESSED_DATA_DIR / "domain_risk_features_v0_1.csv"
)

MODEL_FEATURES = [
    "hostname_length",
    "digit_count",
    "digit_ratio",
    "hyphen_count",
    "hostname_entropy",
    "alphabetic_ratio",
]


def ensure_output_directories() -> None:
    """Create directories used for generated EDA outputs."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_data(
    data_path: Path = FEATURE_DATA_PATH,
) -> pd.DataFrame:
    """Load the prepared feature dataset for exploratory analysis."""

    return pd.read_csv(data_path)


def summarise_features_by_source(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return mean model-feature values for each dataset source."""

    return (
        df.groupby("source")[MODEL_FEATURES]
        .mean()
        .round(4)
    )


def describe_features_by_source(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return descriptive statistics for model features by source."""

    return (
        df.groupby("source")[MODEL_FEATURES]
        .describe()
        .round(4)
    )


def plot_hostname_length_distribution(
    df: pd.DataFrame,
) -> None:
    """Plot hostname-length distributions for Tranco and URLhaus."""

    ensure_output_directories()

    figure, axis = plt.subplots(figsize=(10, 6))

    for source in ["tranco", "urlhaus"]:
        values = df.loc[
            df["source"] == source,
            "hostname_length",
        ]

        axis.hist(
            values,
            bins=range(5, 53, 2),
            alpha=0.6,
            label=source.title(),
        )

    axis.set_title("Hostname Length Distribution by Dataset Source")
    axis.set_xlabel("Hostname length (characters)")
    axis.set_ylabel("Number of observations")
    axis.legend()

    figure.tight_layout()

    output_path = FIGURES_DIR / "hostname_length_distribution.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)



def plot_hostname_entropy_distribution(
    df: pd.DataFrame,
) -> None:
    """Plot hostname-entropy distributions for Tranco and URLhaus."""

    ensure_output_directories()

    figure, axis = plt.subplots(figsize=(10, 6))

    for source in ["tranco", "urlhaus"]:
        values = df.loc[
            df["source"] == source,
            "hostname_entropy",
        ]

        axis.hist(
            values,
            bins=30,
            alpha=0.6,
            label=source.title(),
        )

    axis.set_title("Hostname Entropy Distribution by Dataset Source")
    axis.set_xlabel("Hostname entropy")
    axis.set_ylabel("Number of observations")
    axis.legend()

    figure.tight_layout()

    output_path = FIGURES_DIR / "hostname_entropy_distribution.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_digit_presence_by_source(
    df: pd.DataFrame,
) -> None:
    """Plot the percentage of hostnames containing at least one digit."""

    ensure_output_directories()

    digit_presence = (
        df.assign(has_digit=df["digit_count"] > 0)
        .groupby("source")["has_digit"]
        .mean()
        .mul(100)
        .reindex(["tranco", "urlhaus"])
    )

    figure, axis = plt.subplots(figsize=(8, 6))

    bars = axis.bar(
        ["Tranco", "URLhaus"],
        digit_presence.values,
    )

    axis.set_title("Hostnames Containing Digits by Dataset Source")
    axis.set_ylabel("Hostnames containing at least one digit (%)")
    axis.set_ylim(0, 35)

    for bar, value in zip(bars, digit_presence.values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.1f}%",
            ha="center",
        )

    figure.tight_layout()

    output_path = FIGURES_DIR / "digit_presence_by_source.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_subdomain_presence_by_source(
    df: pd.DataFrame,
) -> None:
    """Plot subdomain presence to illustrate dataset-source confounding."""

    ensure_output_directories()

    subdomain_presence = (
        df.assign(has_subdomain=df["subdomain_depth"] > 0)
        .groupby("source")["has_subdomain"]
        .mean()
        .mul(100)
        .reindex(["tranco", "urlhaus"])
    )

    figure, axis = plt.subplots(figsize=(8, 6))

    bars = axis.bar(
        ["Tranco", "URLhaus"],
        subdomain_presence.values,
    )

    axis.set_title("Subdomain Presence by Dataset Source")
    axis.set_ylabel("Hostnames with subdomains (%)")
    axis.set_ylim(0, 70)

    for bar, value in zip(bars, subdomain_presence.values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
        )

    figure.tight_layout()

    output_path = FIGURES_DIR / "subdomain_presence_by_source.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_feature_correlation_matrix(
    df: pd.DataFrame,
) -> None:
    """Plot correlations between the frozen model features."""

    ensure_output_directories()

    correlation = df[MODEL_FEATURES].corr()

    figure, axis = plt.subplots(figsize=(10, 8))

    image = axis.imshow(
        correlation,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    axis.set_xticks(range(len(MODEL_FEATURES)))
    axis.set_yticks(range(len(MODEL_FEATURES)))

    axis.set_xticklabels(
        MODEL_FEATURES,
        rotation=45,
        ha="right",
    )
    axis.set_yticklabels(MODEL_FEATURES)

    for row in range(len(MODEL_FEATURES)):
        for column in range(len(MODEL_FEATURES)):
            axis.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
            )

    axis.set_title("Correlation Matrix â€” Frozen Model Features")

    figure.colorbar(
        image,
        ax=axis,
        label="Pearson correlation",
    )

    figure.tight_layout()

    output_path = FIGURES_DIR / "feature_correlation_matrix.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def compare_validation_models() -> pd.DataFrame:
    """Return mean group-aware CV metrics for the three comparison models."""

    training_df = pd.read_csv(TRAIN_PATH)

    model_builders = {
        "Logistic Regression": build_baseline_model,
        "Random Forest": build_random_forest_model,
        "Scaled Logistic": build_scaled_logistic_model,
    }

    results = []

    for model_name, model_builder in model_builders.items():
        fold_results = cross_validate_baseline(
            training_df,
            model_builder=model_builder,
        )

        mean_metrics = (
            fold_results[
                [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                ]
            ]
            .mean()
            .to_dict()
        )

        results.append(
            {
                "model": model_name,
                **mean_metrics,
            }
        )

    return pd.DataFrame(results)


def export_validation_model_comparison() -> None:
    """Export cross-validation model comparison metrics for reporting."""

    ensure_output_directories()

    comparison = compare_validation_models().round(6)

    output_path = REPORT_DATA_DIR / "validation_model_comparison.csv"

    comparison.to_csv(output_path, index=False)


def plot_validation_model_comparison() -> None:
    """Plot mean group-aware CV metrics for the comparison models."""

    ensure_output_directories()

    comparison = compare_validation_models()

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    chart_data = comparison.set_index("model")[metrics].T

    figure, axis = plt.subplots(figsize=(11, 6))

    chart_data.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title("Model Comparison â€” Mean 5-Fold Group-Aware CV")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Mean validation score")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Model")

    figure.tight_layout()

    output_path = FIGURES_DIR / "validation_model_comparison.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_candidate_roc_curve() -> None:
    """Plot the ROC curve from the completed held-out v0.1 evaluation."""

    ensure_output_directories()

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df[CANDIDATE_FEATURE_COLUMNS]
    y_test = test_df["label"]

    model = load_candidate_model()

    probabilities = model.predict_proba(X_test)[:, 1]

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probabilities,
    )

    figure, axis = plt.subplots(figsize=(8, 7))

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label="Scaled Logistic (ROC-AUC = 0.888)",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Chance reference",
    )

    axis.set_title("Frozen Candidate â€” Held-Out ROC Curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()

    figure.tight_layout()

    output_path = FIGURES_DIR / "candidate_held_out_roc_curve.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_candidate_confusion_matrix() -> None:
    """Plot the confusion matrix from the completed held-out v0.1 evaluation."""

    ensure_output_directories()

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df[CANDIDATE_FEATURE_COLUMNS]
    y_test = test_df["label"]

    model = load_candidate_model()

    predictions = model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    image = axis.imshow(matrix, cmap="Blues")

    axis.set_title("Frozen Candidate â€” Held-Out Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")

    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["Tranco (0)", "URLhaus (1)"])
    axis.set_yticklabels(["Tranco (0)", "URLhaus (1)"])

    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
        label="Observations",
    )

    figure.tight_layout()

    output_path = FIGURES_DIR / "candidate_held_out_confusion_matrix.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def get_candidate_confusion_matrix() -> pd.DataFrame:
    """Return the completed held-out candidate confusion matrix in tidy form."""

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df[CANDIDATE_FEATURE_COLUMNS]
    y_test = test_df["label"]

    model = load_candidate_model()

    predictions = model.predict(X_test)

    matrix = confusion_matrix(y_test, predictions)

    return pd.DataFrame(
        [
            {
                "actual_class": "Tranco (0)",
                "predicted_class": "Tranco (0)",
                "count": int(matrix[0, 0]),
            },
            {
                "actual_class": "Tranco (0)",
                "predicted_class": "URLhaus (1)",
                "count": int(matrix[0, 1]),
            },
            {
                "actual_class": "URLhaus (1)",
                "predicted_class": "Tranco (0)",
                "count": int(matrix[1, 0]),
            },
            {
                "actual_class": "URLhaus (1)",
                "predicted_class": "URLhaus (1)",
                "count": int(matrix[1, 1]),
            },
        ]
    )


def export_candidate_confusion_matrix() -> None:
    """Export the completed held-out confusion matrix for reporting."""

    ensure_output_directories()

    matrix = get_candidate_confusion_matrix()
    matrix.insert(0, "evaluation_status", "used_held_out_test")

    output_path = (
        REPORT_DATA_DIR / "candidate_confusion_matrix.csv"
    )

    matrix.to_csv(output_path, index=False)


def get_candidate_standardized_coefficients() -> pd.DataFrame:
    """Return standardized coefficients from the frozen candidate model."""

    model = load_candidate_model()

    coefficients = model.named_steps["classifier"].coef_[0]

    result = pd.DataFrame(
        {
            "feature": CANDIDATE_FEATURE_COLUMNS,
            "standardized_coefficient": coefficients,
        }
    )

    result["absolute_coefficient"] = (
        result["standardized_coefficient"].abs()
    )

    return result.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def export_candidate_coefficients() -> None:
    """Export frozen candidate standardized coefficients for reporting."""

    ensure_output_directories()

    coefficients = get_candidate_standardized_coefficients().round(
        {
            "standardized_coefficient": 6,
            "absolute_coefficient": 6,
        }
    )

    output_path = REPORT_DATA_DIR / "candidate_coefficients.csv"

    coefficients.to_csv(output_path, index=False)


def plot_candidate_standardized_coefficients() -> None:
    """Plot standardized coefficients from the frozen candidate model."""

    ensure_output_directories()

    coefficients = get_candidate_standardized_coefficients()

    plot_data = coefficients.sort_values(
        "standardized_coefficient"
    )

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.barh(
        plot_data["feature"],
        plot_data["standardized_coefficient"],
    )

    axis.axvline(
        0,
        linewidth=1,
    )

    axis.set_title("Frozen Candidate â€” Standardized Coefficients")
    axis.set_xlabel(
        "Standardized logistic-regression coefficient"
    )
    axis.set_ylabel("Feature")

    figure.tight_layout()

    output_path = (
        FIGURES_DIR / "candidate_standardized_coefficients.png"
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def build_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Build tidy pairwise correlations for the frozen model features."""

    correlation_matrix = df[MODEL_FEATURES].corr()

    rows = []

    for feature_x in MODEL_FEATURES:
        for feature_y in MODEL_FEATURES:
            rows.append(
                {
                    "feature_x": feature_x,
                    "feature_y": feature_y,
                    "correlation": round(
                        correlation_matrix.loc[feature_x, feature_y],
                        4,
                    ),
                }
            )

    return pd.DataFrame(rows)


def export_feature_correlations(df: pd.DataFrame) -> None:
    """Export tidy feature correlations for reporting."""

    ensure_output_directories()

    correlations = build_feature_correlations(df)

    output_path = REPORT_DATA_DIR / "feature_correlations.csv"

    correlations.to_csv(output_path, index=False)


def build_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Build tidy descriptive statistics by source and model feature."""

    rows = []

    for source, source_df in df.groupby("source"):
        for feature in MODEL_FEATURES:
            values = source_df[feature]

            rows.append(
                {
                    "source": source,
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": round(values.mean(), 4),
                    "std": round(values.std(), 4),
                    "min": round(values.min(), 4),
                    "q25": round(values.quantile(0.25), 4),
                    "median": round(values.median(), 4),
                    "q75": round(values.quantile(0.75), 4),
                    "max": round(values.max(), 4),
                }
            )

    return pd.DataFrame(rows)


def export_feature_statistics(df: pd.DataFrame) -> None:
    """Export tidy feature descriptive statistics for reporting."""

    ensure_output_directories()

    statistics = build_feature_statistics(df)

    output_path = REPORT_DATA_DIR / "feature_statistics.csv"

    statistics.to_csv(output_path, index=False)


def build_lexical_presence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build source-level presence rates for selected lexical characteristics."""

    rows = []

    for source, source_df in df.groupby("source"):
        characteristics = {
            "Contains digit": source_df["digit_count"] > 0,
            "Contains hyphen": source_df["hyphen_count"] > 0,
            "Contains subdomain": source_df["subdomain_depth"] > 0,
        }

        for characteristic, mask in characteristics.items():
            rows.append(
                {
                    "source": source,
                    "characteristic": characteristic,
                    "count": int(mask.sum()),
                    "percentage": round(mask.mean() * 100, 2),
                }
            )

    return pd.DataFrame(rows)


def export_lexical_presence_summary(df: pd.DataFrame) -> None:
    """Export lexical-characteristic presence rates for reporting."""

    ensure_output_directories()

    summary = build_lexical_presence_summary(df)

    output_path = REPORT_DATA_DIR / "lexical_presence_summary.csv"

    summary.to_csv(output_path, index=False)


def build_source_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build source-level summary statistics for reporting."""

    summary = (
        df.groupby("source")
        .agg(
            observations=("hostname", "size"),
            unique_registrable_domains=(
                "registrable_domain",
                "nunique",
            ),
            mean_hostname_length=("hostname_length", "mean"),
            mean_digit_ratio=("digit_ratio", "mean"),
            mean_hyphen_count=("hyphen_count", "mean"),
            mean_hostname_entropy=("hostname_entropy", "mean"),
            mean_alphabetic_ratio=("alphabetic_ratio", "mean"),
        )
        .reset_index()
    )

    return summary.round(
    {
        "mean_hostname_length": 4,
        "mean_digit_ratio": 4,
        "mean_hyphen_count": 4,
        "mean_hostname_entropy": 4,
        "mean_alphabetic_ratio": 4,
    }
)


def export_source_summary(df: pd.DataFrame) -> None:
    """Export source-level summary statistics for reporting."""

    ensure_output_directories()

    summary = build_source_summary(df)

    output_path = REPORT_DATA_DIR / "source_summary.csv"

    summary.to_csv(output_path, index=False)


def export_candidate_evaluation_metrics() -> None:
    """Export the completed held-out candidate evaluation for reporting."""

    ensure_output_directories()

    results = evaluate_candidate()

    metrics = pd.DataFrame(
        [
            {
                "model": results["model"],
                "evaluation_status": "used_held_out_test",
                "test_observations": results["test_observations"],
                "accuracy": round(results["accuracy"], 6),
                "precision": round(results["precision"], 6),
                "recall": round(results["recall"], 6),
                "f1": round(results["f1"], 6),
                "roc_auc": round(results["roc_auc"], 6),
            }
        ]
    )

    output_path = REPORT_DATA_DIR / "candidate_evaluation_metrics.csv"

    metrics.to_csv(output_path, index=False)


def generate_all_figures() -> None:
    """Generate the complete reproducible EDA and evaluation figure set."""

    df = load_feature_data()

    plot_hostname_length_distribution(df)
    plot_hostname_entropy_distribution(df)
    plot_digit_presence_by_source(df)
    plot_subdomain_presence_by_source(df)
    plot_feature_correlation_matrix(df)

    plot_validation_model_comparison()
    plot_candidate_roc_curve()
    plot_candidate_confusion_matrix()
    plot_candidate_standardized_coefficients()



    export_source_summary(df)
    export_lexical_presence_summary(df)
    export_feature_statistics(df)
    export_feature_correlations(df)
    export_validation_model_comparison()
    export_candidate_coefficients()
    export_candidate_evaluation_metrics()
    export_candidate_confusion_matrix()


if __name__ == "__main__":
    generate_all_figures()
