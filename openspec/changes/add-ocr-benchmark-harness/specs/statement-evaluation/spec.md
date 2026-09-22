# Spec Delta

## Purpose

Scores bank-statement extraction without a labeled corpus. It uses PDF text-layer ground truth, arithmetic self-consistency and cross-model agreement, anchored by a small manual label set, and measures duplicate detection for Q5.

## ADDED Requirements

### Requirement: Check A — text-layer ground truth
For `digital` pages, the system SHALL score the following against the text-layer ground truth, with both sides passed through the same normalizer:

- Page text with `cer`.
- Header fields (account number, account name, period, opening/closing balance) with `field_exact`.

It SHALL parse ground-truth rows from the text layer using the bank's configured column layout, and score rows with `row_f1`.

#### Scenario: Bank without a layout
- **WHEN** a digital page's bank has no configured row layout
- **THEN** page CER and header fields are still scored, `row_f1` is marked not applicable for that page, and the report counts such pages

### Requirement: Check B — arithmetic self-consistency
For every extracted statement file, the system SHALL evaluate the following equations:

- Every row satisfies `balance_i = balance_{i-1} + credit_i − debit_i`.
- The sum of credits equals the summary credit total, and the sum of debits equals the summary debit total, when the totals are present.
- The last balance equals the closing balance.

Per model, it SHALL report:

- `row_consistency_rate`: the share of rows where the balance equation holds.
- `file_reconciles`: all equations hold for the file.
- A histogram of the first break row.

Decimal arithmetic SHALL be exact.

#### Scenario: Consistent file
- **WHEN** every row satisfies the balance equation and the last balance equals the closing balance
- **THEN** `file_reconciles=true` for that file

#### Scenario: Broken row flagged
- **WHEN** row 7's balance does not equal row 6's balance plus credit minus debit
- **THEN** row 7 gets `balance_mismatch`, `first_break_row=7`, and `file_reconciles=false`

#### Scenario: Cross-page continuity
- **WHEN** the first balance basis on page i+1 does not match the last balance on page i
- **THEN** the file gets a `page_gap` flag

### Requirement: Check C — cross-model agreement
For each field and row that appears in both models' outputs for the same page and condition, the system SHALL record `agree` when the normalized values match. It SHALL report the agreement rate per model pair as its own column.

#### Scenario: Missing in one model
- **WHEN** a field is present in TeleOCR's output but absent in dots.ocr's
- **THEN** the field's agreement is null (not counted as agree or disagree)

### Requirement: Review queue and manual anchor set
The system SHALL export `review/queue.csv` with one row per field. It SHALL contain every cross-model disagreement plus a random 10% of agreements (fixed seed), with columns for `sample_id`, page, field, both model values and an empty label column.

The system SHALL load a labeled CSV of 30–50 pages. These labels cover header fields and every row's (date, debit, credit, balance). The loaded labels SHALL be treated as ground truth for statement accuracy. They SHALL also be used to report the precision of Check B and Check C flags as predictors of true errors.

#### Scenario: Label round trip
- **WHEN** the exported queue is filled in and loaded
- **THEN** labeled pages get `gt_kind=json` statement ground truth, and statement accuracy and flag precision are computed on them

#### Scenario: Invalid label row
- **WHEN** a label CSV row has an unparseable amount or date
- **THEN** the loader rejects the file with the offending line number instead of skipping the row

### Requirement: Duplicate detection evaluation
The system SHALL build a duplicate test set. It contains 20 original statements, each with 3 variants:

- a re-scan (`scan_low`)
- 180° rotation followed by `photo`
- a 5% margin crop

It also contains 20 distinct statements from the same accounts as hard negatives.

The detector SHALL combine three signals:

- exact SHA-256 of the file bytes
- Jaccard similarity over the set of (date, amount) row tuples
- a perceptual hash of the page image

The system SHALL report recall on variants and false-positive rate on hard negatives, for Jaccard thresholds 0.6, 0.7, 0.8 and 0.9, per model.

#### Scenario: Exact duplicate
- **WHEN** two files have identical bytes
- **THEN** they are flagged duplicate regardless of extraction output

#### Scenario: Threshold sweep
- **WHEN** duplicate evaluation completes
- **THEN** one recall/FPR row exists for each (model, threshold ∈ {0.6, 0.7, 0.8, 0.9})
