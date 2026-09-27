# Zaxonite Domain Risk ML — Model Scope

## Version

Frozen v0.1 candidate

## Objective

Develop a machine-learning classifier that identifies lexical characteristics associated with malware-linked URLhaus hostnames compared with established Tranco domains.

The model is designed to provide a supporting domain-risk intelligence signal rather than a standalone determination of maliciousness.

## Classes

### 0 — Tranco / Established Domain

Domain sampled from the Tranco established-domain reference dataset.

This label does not guarantee that a domain is safe, benign or trustworthy.

### 1 — URLhaus / Malware-Associated Hostname

Domain or hostname derived from URLhaus threat-intelligence observations containing malware-associated URLs.

This label represents association with the URLhaus source dataset. It does not establish that the domain owner, organisation, or every resource hosted on the domain is malicious.

## Frozen v0.1 Feature Boundary

The frozen v0.1 candidate uses six features derived exclusively from the hostname string:

- `hostname_length`
- `digit_count`
- `digit_ratio`
- `hyphen_count`
- `hostname_entropy`
- `alphabetic_ratio`

`subdomain_depth` was investigated during exploratory analysis but deliberately excluded from model training because of strong source-specific confounding. In the prepared dataset, subdomains occur in 61.33% of URLhaus observations and 0.00% of Tranco observations.

DNS intelligence, RDAP data, registration history, certificate information, website content, network infrastructure, reputation services and behavioural telemetry are outside the v0.1 model boundary.

## Model Boundary

The frozen v0.1 candidate is a scaled logistic-regression classifier.

Its output represents a **URLhaus-class probability within the v0.1 dataset and modelling framework**.

The output must not be interpreted as a calibrated probability that an arbitrary domain is malicious.

## Intended Use

The model is intended for:

- security research and analysis;
- domain and URL risk triage;
- prioritisation for further investigation;
- use as one supporting signal within a wider evidence-based assessment;
- demonstration of reproducible threat-intelligence machine learning.

## Non-Intended Use

The model must not be treated as proof that a domain, organisation or person is fraudulent, malicious, safe or trustworthy.

The model must not be used as a standalone blocking, allow-listing, fraud, identity or security verdict.

The model must not autonomously make consequential decisions about individuals.

## Evaluation Boundary

The v0.1 candidate has completed its held-out evaluation.

The 1,260-observation held-out test partition has therefore been consumed and is no longer considered untouched. It must not be used for further model selection or tuning.

Future model development should use new independent evaluation data, preferably including a later temporal dataset or separately collected operational population.

## Design Principle

Machine-learning output is supporting evidence, not a verdict.

Any future Zaxonite IQ integration should preserve deterministic evidence, explainability and human review alongside the ML-derived signal.