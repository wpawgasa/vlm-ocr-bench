# Spec Delta

## Purpose

Scores every prediction against its ground truth with task-appropriate metrics (text, table, field and classification). Aggregates any slice with sample counts and bootstrap confidence intervals, answering Q1/Q2 and giving numbers comparable to published ThaiOCRBench results.

## ADDED Requirements

### Requirement: Per-sample and per-field score rows
`ocrbench score` SHALL write one row to `scores.jsonl` per `(sample_id, condition, model, metric)`. For field tasks, it SHALL also write one row to `fields.jsonl` per field. Every score row SHALL carry the slice keys `model`, `task`, `condition`, `domain`, `bank`, `doc_type` and `is_critical`, so that any aggregation is a group-by without re-scoring.

#### Scenario: Slice keys present
- **WHEN** a KIE sample is scored
- **THEN** each field row carries `is_critical` set from the manifest's `critical_fields`, plus all other slice keys

### Requirement: Text metrics
The system SHALL compute these text metrics:

- `cer`: character-level Levenshtein distance over NFC codepoints divided by ground-truth length.
- `wer`: Levenshtein distance over Thai word tokens (PyThaiNLP `newmm`) divided by the ground-truth token count.
- `bmfl`: ThaiOCRBench's official text score, (BLEU + METEOR + F-measure + (1 − edit distance)) / 4 over PyThaiNLP `newmm` tokens.

#### Scenario: CER on identical text
- **WHEN** prediction and ground truth are identical after normalization
- **THEN** `cer` = 0

### Requirement: Official ThaiOCRBench task score
For every kept ThaiOCRBench task, the system SHALL also compute `tob_score`, the benchmark's official per-sample score, so results sit beside published numbers:

- BMFL for Full-page OCR, Fine-grained text recognition and Handwritten content extraction.
- The VQA-style score for Text recognition and Document classification.
- TEDS for Table parsing.
- The document-parsing score for Document parsing.
- Key-value F1 for Key information extraction and mapping.

It SHALL be applied to the model's answer in its final documented output form, as the official evaluator receives it: the model's own official post-processing applied (for example OTSL→HTML, or layout JSON rendered as reading-order Markdown with HTML tables), and for benchmark-question requests the reply text itself. It SHALL NOT be applied to our canonicalized text. The implementation SHALL reproduce the published per-sample scores of a reference model (`res_folder` of the ThaiOCRBench repository) within 0.01, on a fixture of at least 20 samples per kept task, before its numbers are reported as comparable.

#### Scenario: Parity fixture gate
- **WHEN** a task's parity fixture test fails (any sample off by more than 0.01)
- **THEN** the report marks that task's `tob_score` "not comparable" instead of presenting it beside published numbers

### Requirement: Structure metrics
The system SHALL compute tree edit distance similarity (`ted`) for Table parsing and Document parsing. The score is normalized to [0,1], where 1 means identical trees. It SHALL compute ANLS with threshold 0.5 for Document classification.

#### Scenario: Identical table
- **WHEN** the predicted HTML table equals the ground-truth table
- **THEN** `ted` = 1

#### Scenario: ANLS threshold
- **WHEN** the normalized Levenshtein similarity between the predicted and true label is below 0.5
- **THEN** the sample's ANLS is 0

### Requirement: Field metrics with FA/FR
For KIE and statement fields, the system SHALL compute these metrics:

- `field_exact`: 1 if the normalized values are equal. `null` equal to `null` counts as correct.
- `field_fuzzy`: 1 if field CER ≤ 0.1.
- `kie_f1`: micro precision, recall and F1 over (field, value) pairs.

A non-null prediction where the ground truth is null SHALL count as a false accept. A null prediction where the ground truth is non-null SHALL count as a false reject. FA and FR rates SHALL be reported.

#### Scenario: False accept
- **WHEN** the ground truth for `id_number` is null and the model outputs "1234"
- **THEN** the field row has `false_accept=true` and `field_exact=0`

#### Scenario: False reject
- **WHEN** the ground truth for `total_amount` is "500.00" and the model outputs null
- **THEN** the field row has `false_reject=true`

### Requirement: Statement row F1
For statements with row ground truth, rows SHALL be matched on normalized (date, amount) with zero tolerance. Row precision, recall and F1 SHALL then be reported.

#### Scenario: Row matched
- **WHEN** a predicted row and a ground-truth row share the same date and the same debit/credit amount
- **THEN** they count as one true-positive match

### Requirement: Aggregation with n and confidence intervals
Aggregates SHALL be produced at least for:

- task × model
- condition × model
- critical vs non-critical × model
- bank × doc_type × model
- Government/Finance domain × model, when available

Every aggregate SHALL report `n` and a 95% bootstrap confidence interval from 1,000 resamples over samples, with a fixed seed.

#### Scenario: Aggregate row content
- **WHEN** the task × model table is produced
- **THEN** each cell carries the mean, n, CI low and CI high

#### Scenario: Errors included
- **WHEN** a prediction has `error` set
- **THEN** it contributes worst-case values to all aggregates it belongs to
