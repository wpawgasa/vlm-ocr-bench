# Spec Delta

## Purpose

Assembles all run outputs into a client-facing scorecard that answers each client question line by line. It includes supporting tables, plots and anonymized output samples that follow the normalized output contract.

## ADDED Requirements

### Requirement: Report artifacts
`ocrbench report <run_id>` SHALL write the following to `runs/<run_id>/report/`:

- `scorecard.md`
- `scorecard.xlsx`, with one sheet per table
- `plots/`, containing at least the reliability curve, latency curves and CER by condition
- `samples/`, containing anonymized Q4 examples

#### Scenario: Report from complete run
- **WHEN** `report` runs after score, calibrate, probe and bench-latency
- **THEN** all four artifact kinds exist, and every table in the markdown has a matching xlsx sheet

### Requirement: Question-mapped scorecard
The scorecard SHALL contain one row per client question line: Q1 Thai text (CER by condition), Q1 handwriting, Q1 tables, Q1 bank statements, Q2 field accuracy (with a critical-only row), Q2 FA/FR, Q3 confidence, Q4 output, Q5 duplicates, Q5 reconciliation and Q8 throughput. Each row SHALL have the columns Metric, TeleOCR, dots.ocr, n, Data and Note. A row whose inputs are missing SHALL be marked "not benchmarked", with the reason, and SHALL NOT be omitted.

#### Scenario: Missing stage
- **WHEN** `bench-latency` was not run for the run id
- **THEN** the Q8 row reads "not benchmarked — no latency.jsonl" and the other rows are still filled

#### Scenario: Proxy labeling
- **WHEN** the scorecard is rendered
- **THEN** ThaiOCRBench-derived rows are labeled as capability proxy, statement-derived rows as domain proxy, and the two are never pooled into one number

### Requirement: Supporting tables
The report SHALL include these tables, each cell with n and 95% CI:

- task × model (headline)
- condition × model
- critical vs non-critical × model
- bank × doc_type × model
- the capability probe matrix
- calibration bands and thresholds
- statement checks A/B/C
- duplicate recall/FPR by threshold
- latency by (model, gpu, concurrency)

It SHALL also count `missing_required` and `page_gap` flags per file, for sheet 02 items 3.3 and 3.4.

#### Scenario: Flag counts
- **WHEN** statements have pages with `page_gap`
- **THEN** the report lists per-file counts of `missing_required` and `page_gap`

### Requirement: Normalized output contract
Every field in client-facing samples SHALL follow this shape: `field`, `value`, `raw`, `type`, `unit`, `definition`, `source` {file, page, bbox}, `normalization` (list of applied steps), `confidence`, `error_flags`, `model`, `prompt_version` and `run_id`. `error_flags` SHALL be drawn from {`balance_mismatch`, `total_mismatch`, `model_disagreement`, `low_confidence`, `missing_required`, `page_gap`}.

#### Scenario: Contract validation
- **WHEN** a sample field is written
- **THEN** it validates against the output contract schema, and an unknown error flag is rejected

#### Scenario: Low-confidence flag
- **WHEN** a field's confidence is below the stored `t_accept` for the 99% target
- **THEN** its `error_flags` includes `low_confidence`

### Requirement: Anonymization of client-facing samples
Before writing `samples/`, the report SHALL make these replacements:

- person and account names become `[NAME]`
- account numbers keep only the last 4 digits
- bounding boxes of redacted spans are blanked

Raw client data outside `samples/` SHALL NOT be copied into the report.

#### Scenario: Account masking
- **WHEN** a sample contains account number "1234567890"
- **THEN** the written sample shows only "7890" with the rest masked, and that field's bbox is null
