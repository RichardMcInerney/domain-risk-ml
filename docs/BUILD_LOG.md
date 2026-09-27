# Domain Risk ML â€” Build Log

This document is a lightweight chronological record of significant development
milestones, technical decisions, tests and changes made during development of
the Domain Risk ML model.

It is intended as an engineering record rather than a formal governance document.

---

## 26 September 2026 â€” Initial Development Baseline

### Project objective

Develop a machine-learning model that analyses lexical characteristics of
domains/hostnames associated with malware activity and compares them with
established domains.

The model is intended to provide supporting domain-risk evidence for future
Zaxonite intelligence capabilities. Its output must not be treated as proof
that a domain is malicious or legitimate.

### Dataset design

Initial dataset sources:

- URLhaus â€” malware-associated URL/domain observations.
- Tranco â€” established/popular domains used as the comparison population.

Current prepared baseline:

- 6,150 total observations.
- 3,075 URLhaus observations.
- 3,075 Tranco observations.
- Known overlapping domain families excluded.
- Dataset designed for group-aware separation by registrable domain to reduce
  leakage between training and evaluation data.

Dataset provenance and scope are documented separately in:

- `docs/DATASET_PROVENANCE.md`
- `docs/MODEL_SCOPE.md`

### Initial lexical feature engineering

The first feature module is implemented in:

`src/features.py`

Current feature set:

- Hostname length
- Digit count
- Digit ratio
- Hyphen count
- Subdomain depth
- Hostname entropy
- Alphabetic-character ratio

The initial design deliberately concentrates on observable lexical properties
of the hostname rather than attempting to infer ownership, intent or
maliciousness directly.

### Testing

Feature-level unit tests are maintained in:

`tests/test_features.py`

Tests have been added incrementally alongside each feature implementation.

### Feature test checkpoint â€” 26 September 2026

The initial lexical feature test suite was executed successfully.

Result:

- 7 tests collected.
- 7 tests passed.
- 0 failures.
- Runtime: 0.39 seconds.

Validated features:

- Alphabetic-character ratio
- Hostname length
- Digit count
- Digit ratio
- Hyphen count
- Subdomain depth
- Hostname entropy

This establishes the first tested feature-engineering baseline before
dataset-wide feature extraction and model development.

### Development principles

- Build and test features individually before model training.
- Avoid train/test leakage between related domains.
- Preserve dataset provenance.
- Treat model output as supporting risk evidence rather than a verdict.
- Prefer interpretable baseline modelling before increasing model complexity.
- Record significant development milestones in this build log.

### Current development state

Feature engineering is in progress.

The newest feature added is `alphabetic_ratio`. Its unit test has been written
and is awaiting execution as the first task of the next development stage.

### Feature builder checkpoint

A dedicated feature-building module was introduced in `src/build_features.py`
to keep feature extraction separate from raw dataset preparation.

The module currently converts a single hostname into a structured feature
record containing all seven validated lexical features.

A unit test was added to verify that the feature record contains the expected
fields and correctly maps each feature function to its output.

Test result:

- 8 tests collected.
- 8 tests passed.
- 0 failures.

The feature extraction layer is now ready to be extended to the complete
6,150-observation baseline dataset.

### Dataset-source confounding identified â€” subdomain depth

Exploratory feature QA identified a significant source-related difference in
the `subdomain_depth` feature.

Observed distribution:

- Tranco observations with a subdomain: 0 / 3,075 (0.00%)
- URLhaus observations with a subdomain: 1,886 / 3,075 (61.33%)
- URLhaus observations without a subdomain: 1,189 / 3,075 (38.67%)

This difference is strongly influenced by source construction. The Tranco
dataset supplies listed domains, whereas URLhaus observations are extracted
from URLs and may therefore contain deeper hostnames.

Consequently, `subdomain_depth` could allow a classifier to learn dataset
provenance rather than a generalisable malware-associated lexical signal.

Decision:

- Retain `subdomain_depth` in the generated feature dataset for analysis and
  traceability.
- Exclude `subdomain_depth` from the initial baseline model training features.
- Reassess the feature in a later dataset version containing appropriate
  legitimate subdomain examples.

Other initial lexical features remain candidates for baseline modelling,
subject to further QA.

### Candidate feature correlation review

Correlation analysis was performed on the six lexical features being considered
for initial baseline modelling after excluding `subdomain_depth`.

Notable correlations included:

- `digit_count` / `digit_ratio`: 0.904
- `digit_ratio` / `alphabetic_ratio`: -0.898
- `digit_count` / `alphabetic_ratio`: -0.793
- `hostname_length` / `hostname_entropy`: 0.789

These relationships are plausible consequences of the underlying hostname
composition and do not by themselves indicate a feature-generation error.

Decision:

- Retain all six candidate features for the first baseline experiment.
- Record the substantial feature correlations when interpreting model
  coefficients and feature importance.
- Do not interpret correlated feature importance independently without
  considering redundancy.
- Compare against a reduced-feature configuration in a later experiment if
  appropriate.
- Continue excluding `subdomain_depth` from baseline model training because
  of the previously identified dataset-source confounding.

### Registrable-domain grouping checkpoint

Registrable-domain metadata was added to the generated feature dataset to
support leakage-resistant model evaluation.

The registrable domain is retained as evaluation metadata and is not intended
to be used as a predictive model feature.

Dataset QA after regeneration:

- Total observations: 6,150
- Total columns: 11
- Missing registrable domains: 0
- Blank registrable domains: 0
- Unique registrable-domain groups: 5,329
- Mean observations per group: 1.154
- Maximum observations in a single group: 19

Most registrable-domain groups contain one hostname, but some contain multiple
related hostnames. A row-level random train/test split could therefore place
related hostnames on both sides of the evaluation boundary.

Decision:

- Train/test separation will be performed at registrable-domain group level.
- All hostnames belonging to the same registrable domain must remain within
  the same partition.
- `registrable_domain` will not be supplied to the classifier as a predictive
  feature.


### Group-aware train/test split checkpoint

A deterministic group-aware train/test split was implemented using
`registrable_domain` as the grouping boundary.

Configuration:

- Random seed: 42
- Requested test proportion: 20%
- Training observations: 4,890
- Test observations: 1,260
- Total observations preserved: 6,150

Class distribution:

- Training: 2,462 label 0 / 2,428 label 1
- Test: 613 label 0 / 647 label 1

Post-write QA confirmed:

- Registrable-domain overlap between train and test: 0
- Missing values in training dataset: 0
- Missing values in test dataset: 0

The held-out test dataset will not be used for model fitting or routine model
selection. Predictive model development will use the training partition.

`registrable_domain` is retained as evaluation metadata and will not be
supplied to the classifier as a predictive feature.


### Initial baseline training checkpoint

The first baseline logistic-regression classifier was successfully fitted
using the 4,890-observation training partition.

Predictive inputs were restricted to the six approved lexical features:

- hostname_length
- digit_count
- digit_ratio
- hyphen_count
- hostname_entropy
- alphabetic_ratio

The held-out test partition was not used during this training step.

Initial fitted coefficients:

- hostname_entropy: +1.289582
- hostname_length: +0.193572
- digit_count: +0.009527
- hyphen_count: -0.219064
- digit_ratio: -4.334788
- alphabetic_ratio: -7.457649

Positive coefficients are associated with movement toward label 1
(URLhaus-associated) and negative coefficients toward label 0 (Tranco),
conditional on the other model inputs.

Raw coefficient magnitude will not be treated as direct feature importance.
Several candidate features are strongly correlated and operate on different
numeric scales, so coefficient interpretation requires caution.

The held-out test dataset remains untouched.

### Baseline cross-validation checkpoint

The six-feature logistic-regression baseline was evaluated using 5-fold
stratified group-aware cross-validation on the 4,890-observation training
partition.

Registrable-domain groups remained isolated between training and validation
within every fold. The separate 1,260-observation held-out test partition was
not used.

Mean validation metrics:

- Accuracy: 0.756237
- Precision: 0.764543
- Recall: 0.736410
- F1: 0.750011
- ROC-AUC: 0.830428

Observed fold ranges:

- Accuracy: 0.745399 to 0.777096
- ROC-AUC: 0.819274 to 0.851310

The relatively consistent fold results indicate that the initial lexical
feature set contains useful class-separation signal across unseen
registrable-domain groups.

These results are treated as a baseline rather than final model performance.
The held-out test dataset remains reserved for later final evaluation.

`subdomain_depth` remains excluded because of the previously identified
dataset-source confounding.

### Reduced-feature comparison

A four-feature logistic-regression candidate was evaluated to test whether
strongly correlated lexical variables were adding unnecessary redundancy.

Reduced feature set:

- hostname_length
- digit_ratio
- hyphen_count
- hostname_entropy

`digit_count` and `alphabetic_ratio` were removed for this experiment because
of their strong correlations with `digit_ratio`.

The same 5-fold stratified group-aware validation procedure was used.

Mean reduced-model metrics:

- Accuracy: 0.747444
- Precision: 0.747913
- Recall: 0.741347
- F1: 0.744577
- ROC-AUC: 0.816703

Compared with the six-feature baseline, the reduced model produced slightly
higher recall but lower accuracy, precision, F1 and ROC-AUC.

Decision:

The original six-feature logistic-regression model remains the baseline.
Correlation alone will not be treated as sufficient reason to remove a
feature when controlled validation shows a loss of discriminatory performance.

The held-out test dataset remains untouched.

### Random Forest baseline comparison

A Random Forest classifier was evaluated against the logistic-regression
baseline using the same six lexical features and the same 5-fold stratified
group-aware cross-validation procedure.

Random Forest configuration:

- 200 estimators
- random_state = 42
- n_jobs = -1
- no hyperparameter tuning

Mean validation metrics:

- Accuracy: 0.787117
- Precision: 0.824405
- Recall: 0.726533
- F1: 0.772158
- ROC-AUC: 0.837046

Compared with logistic regression, Random Forest improved accuracy,
precision, F1 and ROC-AUC, while recall decreased slightly.

The improvement in ROC-AUC was modest, indicating that nonlinear modelling
extracts some additional signal from the current lexical feature set but
does not radically change discrimination performance.

No Random Forest hyperparameter optimisation has yet been performed.

The held-out test dataset remains untouched.

### Standardized logistic-regression comparison

A second logistic-regression model was evaluated using StandardScaler
followed by LogisticRegression in a scikit-learn Pipeline.

Scaling was fitted independently within each training fold, preventing
validation data from influencing preprocessing.

The same six lexical features and 5-fold stratified group-aware
cross-validation procedure were used.

Mean validation metrics:

- Accuracy: 0.777914
- Precision: 0.792926
- Recall: 0.749186
- F1: 0.770104
- ROC-AUC: 0.846411

Compared with the original unscaled logistic-regression baseline, the
standardized model improved all five reported validation metrics.

Compared with the initial Random Forest, the standardized logistic model
produced higher recall and ROC-AUC, while the Random Forest produced higher
accuracy, precision and slightly higher F1.

Decision:

Standardized logistic regression is retained as a serious candidate
alongside Random Forest. No final model has yet been selected.

The held-out test dataset remains untouched.

### Standardized coefficient stability

The standardized logistic-regression coefficients were examined across all
five stratified group-aware validation folds.

Mean coefficient and standard deviation across folds:

- digit_ratio: -2.229340 (SD 0.109538)
- alphabetic_ratio: -1.888628 (SD 0.034611)
- hostname_length: +1.339800 (SD 0.033022)
- digit_count: +0.816674 (SD 0.110542)
- hostname_entropy: +0.604730 (SD 0.041327)
- hyphen_count: -0.497712 (SD 0.023507)

All six features retained the same coefficient direction across all five
validation folds.

This indicates that the fitted relationships are reasonably stable across
the group-separated training partitions. Coefficient magnitude and direction
are interpreted as properties of the fitted model, not as causal effects or
standalone domain-risk rules.

Correlated predictors remain subject to conditional-effect interpretation.

The held-out test dataset remains untouched.

### Frozen candidate held-out evaluation

The frozen `scaled_logistic_v0_1` candidate was evaluated once against the
previously untouched held-out test partition.

Candidate configuration:

- Training observations: 4,890
- Held-out test observations: 1,260
- Six frozen lexical features
- StandardScaler
- LogisticRegression
- subdomain_depth excluded
- Model and feature boundary frozen before test evaluation

Held-out results:

- Accuracy: 0.812698
- Precision: 0.823622
- Recall: 0.808346
- F1: 0.815913
- ROC-AUC: 0.888246

For comparison, mean 5-fold group-aware validation ROC-AUC during development
was 0.846411.

All reported held-out metrics exceeded the corresponding cross-validation
means. This does not imply that the held-out figures are expected future
production performance; the particular held-out partition may be easier than
the average validation fold.

The held-out dataset has now been used for final candidate evaluation and
must no longer be treated as an untouched test set.

Future model development must not tune against this test partition.
A new independent and preferably temporally separated evaluation dataset
should be used for subsequent model generations.

## Frozen Model Artifact and Inference

- Frozen candidate: `scaled_logistic_v0_1`
- Saved fitted pipeline: `models/scaled_logistic_v0_1.joblib`
- Metadata sidecar: `models/scaled_logistic_v0_1_metadata.json`
- Artifact integrity recorded using SHA-256.
- Production-style inference implemented in `src/predict_domain.py`.
- Inference reuses the existing tested lexical feature-generation logic.
- Inference is restricted to the frozen six-feature candidate boundary.
- `subdomain_depth` remains excluded from model inference.
- Model output is treated as URLhaus-class probability within the training classification framework, not as a literal probability that a domain is malicious.
- Automated tests added for inference feature construction and probability output.
- Full regression suite: **20 passed**.

## Inference Input Validation

The frozen model inference boundary now performs basic hostname validation
before feature extraction.

Validation currently:

- Requires a string input.
- Strips surrounding whitespace.
- Normalizes hostname input to lowercase.
- Rejects empty input.
- Rejects empty DNS labels.
- Rejects labels containing invalid characters.
- Rejects labels longer than 63 characters.
- Rejects hostnames longer than 253 characters.

The validator is intentionally syntactic. It does not perform DNS resolution,
registration checks, reputation checks, or maliciousness determination.

Automated regression coverage was added for normalization, empty input,
non-string input, malformed labels, and overlong hostnames.

Full test suite: **25 passed**.

## CLI Inference Interface

A command-line inference entry point has been added to
`src/predict_domain.py`.

Usage:

    python -m src.predict_domain <hostname>

The CLI:

- validates and normalizes the supplied hostname;
- builds the frozen six-feature inference vector;
- loads `scaled_logistic_v0_1.joblib`;
- returns the URLhaus-class probability to six decimal places.

Example:

    python -m src.predict_domain example.com

Result:

    Hostname: example.com
    URLhaus-class probability: 0.147038

Repeated inference for identical input is deterministic.

Full regression suite: **26 passed**.

### EDA and Model Evaluation Visualisation Checkpoint

- Reproducible EDA workflow added in `analysis/eda.py`.
- Dataset-source feature distributions reproduced from the prepared 6,150-row feature dataset.
- Source-associated lexical characteristics examined, including digit and hyphen presence.
- `subdomain_depth` source confounding reproduced: 0.00% of Tranco observations versus 61.33% of URLhaus observations contained subdomains. The feature remains excluded from model training.
- Frozen-feature correlation structure reproduced.
- Three-model group-aware cross-validation comparison reproduced from the 4,890-row development partition.
- Held-out v0.1 ROC curve and confusion matrix produced for reporting of the already-completed final evaluation only; the used test partition was not used for further model selection or tuning.
- Frozen candidate standardized coefficients reproduced from the development partition.
- Existing regression suite remains green: 26 tests passed.
- Project test command standardized as `python -m pytest -q` to ensure consistent project import resolution.
