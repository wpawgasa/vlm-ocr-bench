# Spec Delta

## ADDED Requirements

### Requirement: Variant detection
For a document whose class declares variants, the system SHALL record the detected variant, drawn from that class's variant list, or `unknown`. For a class without variants, the artifact SHALL record no variant. The classifier SHALL be offered a class's variant ids and aliases only under that class. A reply naming a variant that is not in the chosen class's list SHALL be recorded as `unknown` with the raw reply kept. Conditional required fields SHALL use this value.

This supersedes "Bank detection for statements" of `add-docparse-system`; a statement's variant is its bank.

#### Scenario: Bank as variant
- **WHEN** a KBank statement is classified
- **THEN** the artifact records `class_id: bank_statement` and `variant: kbank`

#### Scenario: Receipt variant of an invoice
- **WHEN** a Thai receipt titled "ใบเสร็จรับเงิน" is classified
- **THEN** the artifact records `class_id: invoice` and `variant: receipt`

#### Scenario: Class without variants
- **WHEN** a government letter is classified as `government_form`
- **THEN** the artifact records no variant, and no variant-conditional field is evaluated against one

### Requirement: Benchmark label map for classification scoring
The system SHALL keep a committed map from ThaiOCRBench Document classification labels to registry class ids or `unknown`, so that docparse's class ids can be scored by the benchmark's classification metric. A benchmark label absent from the map SHALL map to `unknown` and SHALL be counted in the evaluation output.

#### Scenario: Mapped label scored
- **WHEN** a benchmark sample is labelled "Bank statement" and docparse answers `bank_statement`
- **THEN** the classification score for that sample is 1

#### Scenario: Unmapped label counted
- **WHEN** a benchmark label has no entry in the map
- **THEN** the sample's expected class is `unknown`, and the evaluation output counts it under `unmapped_labels`
