# Spec Delta

## Purpose

Defines a per-field confidence score for models that emit none, and calibrates it on labeled fields. Q3 is answered with measured accuracy per confidence band, auto-accept thresholds and the resulting human review rate.

## ADDED Requirements

### Requirement: Field confidence features
For each extracted field, the system SHALL compute these features:

- `mean_logprob` and `min_logprob` over the completion tokens that produced the field value. If the value cannot be located in the completion, the whole-completion statistics SHALL be used and `span_found=false` recorded.
- `agree`: 1 if the other model's normalized value matches, 0 if it differs, null if the other model has no value.
- `arith`: 1 if the field is part of a balance/total equation that holds, 0 if it breaks, null if not applicable.

#### Scenario: Span not found
- **WHEN** a normalized field value does not appear in the raw completion
- **THEN** the features use whole-completion logprob statistics and `span_found=false`

### Requirement: Fitted confidence proxy
The system SHALL fit `conf(f) = σ(w0 + w1·mean_logprob + w2·min_logprob + w3·agree + w4·arith)` by logistic regression on `field_exact`. The training data is labeled fields: ThaiOCRBench KIE, digital statement pages and the manual set. Cross-validation SHALL be 5-fold, grouped by `sample_id`, so fields of one sample never span train and test.

A `logprob_only` variant SHALL be fitted alongside, for single-model production. Null features SHALL be handled by explicit indicator encoding, not by dropping fields. A model whose endpoint returns no token logprobs (PaddleOCR-VL through its pipeline server) SHALL keep its logprob features null with the indicator set; its `full` variant is fitted on the remaining features and its `logprob_only` variant SHALL be recorded as "not available — no logprobs", not fitted.

#### Scenario: Grouped folds
- **WHEN** calibration is fitted
- **THEN** no `sample_id` appears in both the training and held-out fold of any split

#### Scenario: Two variants
- **WHEN** `calibrate` completes
- **THEN** out-of-fold confidences exist for both the `full` and `logprob_only` variants, per model that returns logprobs

#### Scenario: Model without logprobs
- **WHEN** `calibrate` runs on PaddleOCR-VL predictions, which carry no token logprobs
- **THEN** its `logprob_only` variant is reported as "not available — no logprobs" and its `full` variant uses only the agreement and arithmetic features

### Requirement: Calibration bands and ECE
The system SHALL write `calibration.jsonl` and a report table with five bands: ≥0.98, 0.90–0.98, 0.75–0.90, 0.50–0.75 and <0.50. Each band SHALL show field count, accuracy and a 95% CI. The system SHALL also report ECE over 10 equal-width bins, a reliability plot, and the same band table restricted to critical fields.

#### Scenario: Band table
- **WHEN** calibration output is rendered
- **THEN** there is one row per band per (model, variant), with n, accuracy and CI, plus a critical-only table

### Requirement: Threshold search and review rate
`calibrate` SHALL search `t_accept` and `t_reject` for each target auto-accept accuracy given with `--target-acc` (default 0.99 and 0.995). For each target, it SHALL report:

- the auto-accept, review and reject shares
- the accuracy inside each bucket

The review share SHALL be the reported Q3 review rate. Thresholds and fitted weights SHALL be saved to `thresholds.json` so the operating point is reproducible.

#### Scenario: Target not reachable
- **WHEN** no threshold achieves the target accuracy on auto-accepted fields
- **THEN** the result for that target states "not reachable", with auto-accept share 0, rather than reporting a threshold that misses the target

#### Scenario: Reproducible operating point
- **WHEN** `thresholds.json` is applied to the same out-of-fold confidences
- **THEN** it reproduces the reported bucket shares exactly
