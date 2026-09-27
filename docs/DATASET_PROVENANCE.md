# Zaxonite Domain Risk ML — Dataset Provenance

## Purpose

This document records the origin, meaning, handling and limitations of datasets
used to develop and evaluate the Zaxonite Domain Risk ML model.

Dataset inclusion does not imply that every observation has been independently
verified by Zaxonite Systems Ltd.

---

## Dataset 1 — Malware-Associated Domains

### Source
URLhaus

### Provider
abuse.ch

### Intended Role
Positive class (label = 1)

### Source Meaning
URLhaus collects URLs associated with malware distribution.

For this project, domain names or hostnames were extracted from qualifying
URL records.

### Important Limitation
Presence in the source dataset indicates association with a reported
malware-distribution URL. It must not be interpreted as proof that every
resource on the domain is malicious or that the domain owner is responsible
for malicious activity.

### Processing
The original URLhaus full database dump is preserved unchanged in `data/raw/`.

Initial validation confirmed:

- ZIP archive integrity passed.
- Archive contained `csv.txt`.
- Source structure was `id,dateadded,url,url_status,last_online,threat,tags,urlhaus_link,reporter`.
- 58,554 URL observations were present.
- 42,690 observations used raw IP addresses as hosts.
- 15,864 observations used domain hostnames.
- 3,611 unique domain hostnames were identified.
- 0 observations were invalid or unusable during URL parsing.
- Raw IP-address hosts are excluded from the lexical domain-classification population.
- Duplicate hostname observations were collapsed during baseline dataset construction.
- Cross-label overlap with the Tranco comparison dataset was analysed before training-data construction.
- 28 exact hostname overlaps were identified.
- URLhaus contained 2,448 unique registrable domains.
- Tranco contained 999,989 unique registrable domains.
- 194 registrable-domain overlaps were identified.
- Registrable-domain parsing used `tldextract` with its packaged Public Suffix List snapshot and network fetching disabled.
- Overlapping registrable-domain families were excluded from the malware-associated training population to reduce contradictory and ambiguous labels.
- No extraction, filtering, relabelling or transformation was performed on the preserved raw archive.

### Acquisition Date
23 September 2026 (UTC)

### Local Raw File
`data/raw/urlhaus_full_20260923.csv.zip`

### Source Version / Reference
URLhaus full database dump acquired 23 September 2026 (UTC).

---

## Dataset 2 — Established Domains

### Source
Tranco

### Intended Role
Comparison class (label = 0)

### Source Meaning
Tranco provides a research-oriented ranking of popular domains.

### Important Limitation
Inclusion in a popular-domain ranking does not prove that a domain is safe,
benign or trustworthy.

The label represents membership in the comparison dataset rather than a
guaranteed absence of malicious activity.

The original Tranco ZIP archive is preserved unchanged in `data/raw/`.

Initial validation confirmed:

- ZIP archive integrity passed.
- Archive contained `top-1m.csv`.
- CSV contained 1,000,000 ranked domain records.
- Source structure was `rank,domain`.
- No extraction, filtering, relabelling or transformation was performed on the raw archive.

### Acquisition Date
23 September 2026

### Local Raw File
`data/raw/tranco_top_1m_20260923.csv.zip`

### Source Version / Reference
Latest standard Tranco top-one-million list acquired on 23 September 2026.

---

## Label Definition

0 = established-domain comparison sample

1 = malware-associated sample

These labels describe dataset membership.

They must not be represented as:

0 = definitely safe

1 = definitely fraudulent

---

## Data Handling Principles

- Preserve original source data separately from processed data.
- Do not manually alter labels to improve model performance.
- Record filtering and deduplication operations.
- Investigate conflicting labels.
- Prevent train/test leakage.
- Record dataset versions or acquisition dates.
- Do not commit large raw datasets to the public Git repository unless their
  redistribution terms have been reviewed.

---

## Processed Baseline Dataset

### Version

v0.1

### Local File

`data/processed/domain_risk_baseline_v0_1.csv`

### Construction

The baseline dataset was constructed from the preserved URLhaus and Tranco
source archives.

URLhaus observations using raw IP addresses were excluded because the v0.1
model operates on domain hostnames.

URLhaus hostnames whose registrable domain appeared in the Tranco comparison
dataset were excluded to reduce contradictory and ambiguous class labels.

The remaining 3,075 URLhaus hostnames were assigned label `1`
(malware-associated).

A reproducible sample of 3,075 Tranco domains was selected using random seed
`42` and assigned label `0` (established-domain comparison sample).

The resulting dataset contains 6,150 observations with balanced classes.

### Validation

Post-construction validation confirmed:

- 6,150 total observations.
- 3,075 label `0` observations.
- 3,075 label `1` observations.
- 3,075 Tranco observations.
- 3,075 URLhaus observations.
- 0 null values.
- 0 duplicate hostnames.
- 0 invalid class labels.
- 0 hostnames occurring across both class labels.

### Interpretation

Label `1` represents association with the URLhaus malware-distribution
dataset and must not be interpreted as proof that a domain, domain owner,
organisation or individual is malicious.

Label `0` represents membership in the sampled Tranco comparison population
and must not be interpreted as a guarantee that a domain is safe or
trustworthy.