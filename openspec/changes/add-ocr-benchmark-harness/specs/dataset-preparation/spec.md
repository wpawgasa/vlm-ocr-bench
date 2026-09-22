# Spec Delta

## Purpose

Builds the evaluation sample set: a labeled ThaiOCRBench subset (the capability proxy) and ingested Thai bank statements (the domain proxy). Each sample is rendered under three image-quality conditions and recorded in one manifest.

## ADDED Requirements

### Requirement: ThaiOCRBench subset selection
The system SHALL load `typhoon-ai/ThaiOCRBench` and keep exactly these 9 tasks: Full-page OCR, Text recognition, Fine-grained text recognition, Handwritten content extraction, Table parsing, Document parsing, Key information extraction, Key information mapping and Document classification. Chart parsing, Diagram VQA, Cognition VQA and Infographics VQA SHALL be excluded.

Full-page OCR, Text recognition, Fine-grained text recognition and Document parsing SHALL be capped at 200 samples each. The other kept tasks SHALL include all samples. Capped selection SHALL be random with seed 42 and stratified by domain when a domain field exists.

#### Scenario: Task name verification
- **WHEN** the loaded dataset's metadata lacks one of the 9 expected task names
- **THEN** `prepare` exits non-zero and lists the missing name(s) and the task names actually present

#### Scenario: Caps and seed
- **WHEN** `prepare` runs on the full dataset
- **THEN** each capped task contributes at most 200 samples
- **AND** repeated runs select the same sample ids

#### Scenario: Domain slice tagging
- **WHEN** the dataset has a domain field
- **THEN** each manifest row carries its domain, and Government and Finance samples are identifiable for a separate slice
- **WHEN** the dataset has no domain field
- **THEN** `domain` is empty and `prepare` logs that the Government/Finance slice is unavailable

### Requirement: Bank-statement ingestion and routing
The system SHALL ingest a directory of bank-statement PDFs and images. It SHALL classify each file as `digital` (PDF with embedded fonts), `scanned` (PDF with no fonts) or `photo` (image file). It SHALL record the bank for each file from a configured set (kbank, scb, bbl, ktb, krungsri, ttb, gsb). Every page SHALL be rasterized to a 200 dpi PNG and SHALL become one sample, carrying `page_no` and `n_pages`.

#### Scenario: Digital PDF routing
- **WHEN** a PDF whose font listing is non-empty is ingested
- **THEN** its pages get `doc_type=digital`

#### Scenario: Scanned PDF routing
- **WHEN** a PDF with no embedded fonts is ingested
- **THEN** its pages get `doc_type=scanned` and `gt_kind=none` (unless a manual label exists)

#### Scenario: Unknown bank
- **WHEN** a file's bank cannot be determined from the configured mapping
- **THEN** `prepare` exits non-zero, naming the file, and does not silently assign a bank

### Requirement: Text-layer ground truth for digital pages
For every `digital` page, the system SHALL extract the layout-preserving text layer as `gt_text` and per-word bounding boxes as `gt_words`. These SHALL be stored in the page's ground-truth file with `gt_kind=text_layer`. No redaction SHALL happen at ingestion.

#### Scenario: Text-layer GT stored
- **WHEN** a digital page is ingested
- **THEN** its ground-truth file contains non-empty `gt_text` and a `gt_words` list, where each word has text and a bbox in page pixel coordinates at 200 dpi

### Requirement: Statement corpus coverage report
`prepare` SHALL report the statement corpus composition: pages by `doc_type`, distinct banks, and the count of multi-page files. It SHALL warn when the corpus falls below the target mix: at least 40 digital pages, at least 40 scanned/photo pages, at least 4 banks and at least 10 multi-page statements.

#### Scenario: Under-target corpus
- **WHEN** fewer than 40 digital pages are ingested
- **THEN** `prepare` completes but prints a warning that Check A is weakened, and records it in `runs/<run_id>/prepare_summary.json`

### Requirement: Deterministic degradation conditions
Every sample SHALL be materialized under three conditions:
- `clean`: the original image.
- `scan_low`: resampled to a 150 dpi equivalent, JPEG quality 40, gaussian blur σ=0.6.
- `photo`: rotation within ±3°, perspective warp ≤2%, brightness ±15%, JPEG quality 55.

Random parameters SHALL be derived from a seed determined by `(sample_id, condition)`. Ground truth SHALL be unchanged, except that for `photo` any ground-truth bounding boxes SHALL be transformed by the same homography.

#### Scenario: Determinism per sample and condition
- **WHEN** the same sample is degraded twice under `photo`
- **THEN** the outputs are pixel-identical

#### Scenario: Box transform under photo
- **WHEN** a sample with ground-truth word boxes is degraded under `photo`
- **THEN** the stored boxes for that condition equal the original boxes mapped through the applied homography

### Requirement: Manifest schema
The system SHALL write `manifest.jsonl` with one row per `(sample, condition)`. The fields are `sample_id`, `source`, `task`, `domain`, `bank`, `doc_type`, `page_no`, `n_pages`, `condition`, `image_path`, `gt_kind`, `gt_path`, `critical_fields`.

- `gt_kind` SHALL be one of `json`, `text`, `html`, `text_layer`, `none`.
- `critical_fields` SHALL list the field names that count as critical (IDs, amounts, dates, names, account numbers). It SHALL be empty for tasks without fields.

#### Scenario: Row count
- **WHEN** `prepare` completes with N samples
- **THEN** `manifest.jsonl` contains exactly 3·N rows, one per sample and condition

#### Scenario: Schema validation
- **WHEN** a manifest row is read back
- **THEN** it validates against the manifest schema, and a row with an unknown `gt_kind` is rejected
