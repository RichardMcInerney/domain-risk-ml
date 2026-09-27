"""
Feature dataset builder for Zaxonite Domain Risk ML.

Reads the prepared baseline hostname dataset and applies the tested
lexical feature functions from src/features.py.

The resulting dataset preserves hostname, label and source so that
individual model observations remain traceable during development
and evaluation.
"""

from pathlib import Path

import csv
import tldextract

from src.features import (
    alphabetic_ratio,
    digit_count,
    digit_ratio,
    hostname_entropy,
    hostname_length,
    hyphen_count,
    subdomain_depth,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = PROCESSED_DATA_DIR / "domain_risk_baseline_v0_1.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "domain_risk_features_v0_1.csv"

TLD_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=()
)


def get_registrable_domain(hostname: str) -> str:
    """Return the registrable domain used for group-aware splitting."""

    extracted = TLD_EXTRACTOR(hostname)

    if not extracted.domain or not extracted.suffix:
        return ""

    return f"{extracted.domain}.{extracted.suffix}"


def build_feature_record(hostname: str) -> dict:
    """Return the lexical feature values for one hostname."""

    return {
        "hostname_length": hostname_length(hostname),
        "digit_count": digit_count(hostname),
        "digit_ratio": digit_ratio(hostname),
        "hyphen_count": hyphen_count(hostname),
        "subdomain_depth": subdomain_depth(hostname),
        "hostname_entropy": hostname_entropy(hostname),
        "alphabetic_ratio": alphabetic_ratio(hostname),
    }


def build_feature_dataset(
    input_path: Path,
    output_path: Path,
) -> None:
    """Build the ML-ready feature dataset from the prepared baseline CSV."""

    fieldnames = [
        "hostname",
        "label",
        "source",
        "hostname_length",
        "registrable_domain",
        "digit_count",
        "digit_ratio",
        "hyphen_count",
        "subdomain_depth",
        "hostname_entropy",
        "alphabetic_ratio",
    ]

    rows_written = 0

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for row in reader:
                hostname = row["hostname"]
                features = build_feature_record(hostname)

                writer.writerow(
                    {
                        "hostname": hostname,
                        "registrable_domain": get_registrable_domain(hostname),
                        "label": row["label"],
                        "source": row["source"],
                        **features,
                    }
                )

                rows_written += 1

    print("Feature dataset created")
    print(f"Rows written: {rows_written:,}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    build_feature_dataset(
        INPUT_PATH,
        OUTPUT_PATH,
    )
