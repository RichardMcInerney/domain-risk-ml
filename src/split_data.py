"""Create leakage-resistant train/test splits for the domain risk model."""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit



PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = PROCESSED_DATA_DIR / "domain_risk_features_v0_1.csv"

TRAIN_OUTPUT_PATH = PROCESSED_DATA_DIR / "domain_risk_train_v0_1.csv"
TEST_OUTPUT_PATH = PROCESSED_DATA_DIR / "domain_risk_test_v0_1.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.20

def create_group_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset while keeping registrable-domain groups intact."""

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    train_indices, test_indices = next(
        splitter.split(
            df,
            y=df["label"],
            groups=df["registrable_domain"],
        )
    )

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    return train_df, test_df


def validate_group_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Verify that no registrable-domain group appears in both partitions."""

    train_groups = set(train_df["registrable_domain"])
    test_groups = set(test_df["registrable_domain"])

    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise ValueError(
            f"Group leakage detected: {len(overlap)} groups overlap."
        )


def write_group_split(
    input_path: Path,
    train_output_path: Path,
    test_output_path: Path,
) -> None:
    """Create, validate, and save the group-aware train/test datasets."""

    df = pd.read_csv(input_path)

    train_df, test_df = create_group_split(df)

    validate_group_split(train_df, test_df)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print("Group-aware split created")
    print(f"Training rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Training data: {train_output_path}")
    print(f"Test data: {test_output_path}")


if __name__ == "__main__":
    write_group_split(
        INPUT_PATH,
        TRAIN_OUTPUT_PATH,
        TEST_OUTPUT_PATH,
    )
