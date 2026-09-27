import pandas as pd
import joblib

from src.predict_domain import (
    build_inference_features,
    predict_domain_probability,
    validate_hostname,
)

from src.features import (
    alphabetic_ratio,
    digit_count,
    digit_ratio,
    hostname_entropy,
    hostname_length,
    hyphen_count,
    subdomain_depth,
)

from src.build_features import (
    build_feature_record,
    get_registrable_domain,
)

from src.split_data import (
    create_group_split,
    validate_group_split,
)

from src.train_baseline import (
    FEATURE_COLUMNS,
    TRAIN_PATH,
    build_baseline_model,
    load_training_data,
    train_baseline_model,
    create_validation_splitter,
    cross_validate_baseline,
    build_random_forest_model,
    build_scaled_logistic_model,
    CANDIDATE_MODEL_NAME,
    CANDIDATE_FEATURE_COLUMNS,
    save_candidate_model,
)


def test_alphabetic_ratio():
    assert alphabetic_ratio("") == 0.0
    assert alphabetic_ratio("abcd") == 1.0
    assert alphabetic_ratio("1234") == 0.0
    assert alphabetic_ratio("ab12") == 0.5
    assert alphabetic_ratio("abc.com") == 6 / 7


def test_hostname_length():
    assert hostname_length("google.com") == 10
    assert hostname_length("example.org") == 11
    assert hostname_length("a.co") == 4
    assert hostname_length("") == 0


def test_digit_count():
    assert digit_count("google.com") == 0
    assert digit_count("abc123.com") == 3
    assert digit_count("login2026.net") == 4
    assert digit_count("12345.org") == 5
    assert digit_count("") == 0


def test_digit_ratio():
    assert digit_ratio("google.com") == 0.0
    assert digit_ratio("abc123.com") == 0.3
    assert digit_ratio("1234.com") == 0.5
    assert digit_ratio("") == 0.0


def test_hyphen_count():
    assert hyphen_count("google.com") == 0
    assert hyphen_count("secure-login.com") == 1
    assert hyphen_count("account-security-check.net") == 2
    assert hyphen_count("") == 0


def test_subdomain_depth():
    assert subdomain_depth("example.com") == 0
    assert subdomain_depth("www.example.com") == 1
    assert subdomain_depth("login.secure.example.com") == 2
    assert subdomain_depth("example.co.uk") == 0
    assert subdomain_depth("www.example.co.uk") == 1
    assert subdomain_depth("login.secure.example.co.uk") == 2
    assert subdomain_depth("") == 0


def test_hostname_entropy():
    assert hostname_entropy("") == 0.0
    assert hostname_entropy("aaaa") == 0.0
    assert hostname_entropy("ab") == 1.0
    assert hostname_entropy("abcd") == 2.0


def test_build_feature_record():
    record = build_feature_record("example123-test.com")

    assert set(record) == {
        "hostname_length",
        "digit_count",
        "digit_ratio",
        "hyphen_count",
        "subdomain_depth",
        "hostname_entropy",
        "alphabetic_ratio",
    }

    assert record["hostname_length"] == hostname_length("example123-test.com")
    assert record["digit_count"] == digit_count("example123-test.com")
    assert record["digit_ratio"] == digit_ratio("example123-test.com")
    assert record["hyphen_count"] == hyphen_count("example123-test.com")
    assert record["subdomain_depth"] == subdomain_depth("example123-test.com")
    assert record["hostname_entropy"] == hostname_entropy("example123-test.com")
    assert record["alphabetic_ratio"] == alphabetic_ratio("example123-test.com")


def test_get_registrable_domain():
    assert get_registrable_domain("example.com") == "example.com"
    assert get_registrable_domain("www.example.com") == "example.com"
    assert get_registrable_domain("login.secure.example.com") == "example.com"
    assert get_registrable_domain("example.co.uk") == "example.co.uk"
    assert get_registrable_domain("www.example.co.uk") == "example.co.uk"
    assert get_registrable_domain("") == ""


def test_create_group_split_has_no_group_overlap():
    df = pd.DataFrame(
        {
            "hostname": [
                "example.com",
                "www.example.com",
                "test.com",
                "www.test.com",
                "other.com",
                "sample.com",
            ],
            "registrable_domain": [
                "example.com",
                "example.com",
                "test.com",
                "test.com",
                "other.com",
                "sample.com",
            ],
            "label": [0, 0, 1, 1, 0, 1],
        }
    )

    train_df, test_df = create_group_split(df)

    validate_group_split(train_df, test_df)

    assert set(train_df["registrable_domain"]).isdisjoint(
        set(test_df["registrable_domain"])
    )


def test_training_feature_boundary():
    X, y = load_training_data(TRAIN_PATH)

    assert list(X.columns) == FEATURE_COLUMNS
    assert "subdomain_depth" not in X.columns
    assert "registrable_domain" not in X.columns
    assert "source" not in X.columns
    assert "hostname" not in X.columns
    assert len(X) == len(y)


def test_train_baseline_model():
    X, y = load_training_data(TRAIN_PATH)

    model = train_baseline_model(X, y)

    assert hasattr(model, "coef_")
    assert model.coef_.shape == (1, len(FEATURE_COLUMNS))
    assert model.classes_.tolist() == [0, 1]


def test_build_random_forest_model():
    """Random Forest comparison model should use the approved configuration."""

    model = build_random_forest_model()

    assert model.n_estimators == 200
    assert model.random_state == 42
    assert model.n_jobs == -1


def test_build_scaled_logistic_model():
    """Scaled logistic model should contain scaling followed by classification."""

    model = build_scaled_logistic_model()

    assert list(model.named_steps) == ["scaler", "classifier"]
    assert model.named_steps["classifier"].random_state == 42
    assert model.named_steps["classifier"].max_iter == 1000


def test_validation_folds_have_no_group_overlap():
    df = pd.read_csv(TRAIN_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["label"]
    groups = df["registrable_domain"]

    splitter = create_validation_splitter()

    for train_indices, validation_indices in splitter.split(
        X,
        y,
        groups,
    ):
        train_groups = set(groups.iloc[train_indices])
        validation_groups = set(groups.iloc[validation_indices])

        assert train_groups.isdisjoint(validation_groups)


def test_cross_validate_baseline():
    df = pd.read_csv(TRAIN_PATH)

    results = cross_validate_baseline(df)

    assert len(results) == 5
    assert list(results.columns) == [
        "fold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    assert results[metric_columns].notna().all().all()
    assert (
        (results[metric_columns] >= 0)
        & (results[metric_columns] <= 1)
    ).all().all()


def test_candidate_model_configuration():
    """Candidate v0.1 should retain its frozen model identity and features."""

    assert CANDIDATE_MODEL_NAME == "scaled_logistic_v0_1"
    assert CANDIDATE_FEATURE_COLUMNS == FEATURE_COLUMNS
    assert "subdomain_depth" not in CANDIDATE_FEATURE_COLUMNS


def test_save_candidate_model(tmp_path):
    """Saved candidate pipeline should reload successfully."""

    model = build_scaled_logistic_model()

    output_path = tmp_path / "candidate.joblib"

    save_candidate_model(
        model,
        output_path,
    )

    assert output_path.exists()

    loaded_model = joblib.load(output_path)

    assert list(loaded_model.named_steps) == ["scaler", "classifier"]


def test_build_inference_features():
    """Inference should produce exactly the frozen candidate feature set."""

    features = build_inference_features("example.com")

    assert features.shape == (1, len(CANDIDATE_FEATURE_COLUMNS))
    assert list(features.columns) == CANDIDATE_FEATURE_COLUMNS
    assert "subdomain_depth" not in features.columns


def test_predict_domain_probability():
    """Frozen candidate inference should return a valid probability."""

    probability = predict_domain_probability("example.com")

    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


def test_validate_hostname_normalizes_input():
    """Hostname validation should strip whitespace and normalize case."""

    hostname = validate_hostname("  Example.COM  ")

    assert hostname == "example.com"


def test_validate_hostname_rejects_empty_input():
    """Hostname validation should reject empty input."""

    try:
        validate_hostname("   ")
    except ValueError as error:
        assert str(error) == "hostname must not be empty"
    else:
        raise AssertionError("Expected ValueError for empty hostname")


def test_validate_hostname_rejects_non_string_input():
    """Hostname validation should reject non-string input."""

    try:
        validate_hostname(123)
    except TypeError as error:
        assert str(error) == "hostname must be a string"
    else:
        raise AssertionError("Expected TypeError for non-string hostname")


def test_validate_hostname_rejects_invalid_labels():
    """Hostname validation should reject malformed DNS labels."""

    invalid_hostnames = [
        "-example.com",
        "example-.com",
        "exam ple.com",
        "example..com",
    ]

    for hostname in invalid_hostnames:
        try:
            validate_hostname(hostname)
        except ValueError as error:
            assert str(error) in {
                "hostname contains an invalid label",
                "hostname contains an empty label",
            }
        else:
            raise AssertionError(
                f"Expected ValueError for invalid hostname: {hostname}"
            )


def test_validate_hostname_rejects_overlong_hostname():
    """Hostname validation should reject names exceeding 253 characters."""

    hostname = ("a" * 63 + ".") * 3 + ("a" * 63)

    try:
        validate_hostname(hostname)
    except ValueError as error:
        assert str(error) == "hostname exceeds 253 characters"
    else:
        raise AssertionError("Expected ValueError for overlong hostname")


def test_predict_domain_probability_is_deterministic():
    """Repeated inference should produce the same frozen-model result."""

    first = predict_domain_probability("example.com")
    second = predict_domain_probability("example.com")

    assert first == second


def test_validate_hostname_accepts_punycode():
    """Hostname validation should accept syntactically valid ASCII Punycode labels."""

    hostname = validate_hostname("xn--bcher-kva.example")

    assert hostname == "xn--bcher-kva.example"


def test_validate_hostname_rejects_trailing_dot():
    """Hostname validation should reject trailing-dot DNS notation."""

    try:
        validate_hostname("example.com.")
    except ValueError as error:
        assert str(error) == "hostname contains an empty label"
    else:
        raise AssertionError(
            "Expected ValueError for trailing-dot hostname"
        )


def test_validate_hostname_rejects_ip_address():
    """Inference should reject raw IP addresses outside the model population."""

    try:
        validate_hostname("192.168.1.1")
    except ValueError as error:
        assert str(error) == "IP addresses are outside the model input boundary"
    else:
        raise AssertionError(
            "Expected ValueError for raw IP-address input"
        )
