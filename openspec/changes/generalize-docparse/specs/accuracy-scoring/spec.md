# Spec Delta

## ADDED Requirements

### Requirement: Text-layer scoring for non-statement pages
For a page with text-layer ground truth whose task is full-page OCR or document parsing, the system SHALL compute `cer` and `wer` of the prediction's canonical text against the text layer's canonical text in reading order. Statement pages SHALL keep statement scoring. A page whose text layer is marked `structure_only` SHALL get its text metrics written with a `no_verdict` flag.

#### Scenario: Government page CER
- **WHEN** a `thaipdf` government page with `gt_kind=text_layer` is scored for `ocr_fullpage`
- **THEN** `cer` and `wer` rows are written for every model, computed against the text layer

#### Scenario: Structure-only text
- **WHEN** a DocLayNet page's text ground truth is scored
- **THEN** its `cer` row carries `no_verdict=true`

### Requirement: Layout block F1
The system SHALL compute `layout_f1` for pages with layout ground truth: predicted and ground-truth blocks are matched greedily by IoU ≥ 0.5 within the same category, and micro precision, recall and F1 are reported over the page. Categories SHALL be the harness block types; a ground-truth category with no mapping SHALL be ignored and counted.

#### Scenario: Perfect layout
- **WHEN** every predicted block matches a ground-truth block of the same category at IoU ≥ 0.5 and nothing is left over
- **THEN** `layout_f1` = 1

#### Scenario: Wrong category
- **WHEN** a predicted `text` block overlaps a ground-truth `table` block at IoU 0.9
- **THEN** it is not a match, and both count against precision and recall

### Requirement: Reading-order metric
For pages with reading-order ground truth, the system SHALL compute `reading_order`: over blocks matched as for `layout_f1`, the normalized Kendall τ between the predicted and ground-truth orders, mapped to [0, 1] where 1 is identical order. A page with fewer than two matched blocks SHALL get no value.

#### Scenario: Reversed order
- **WHEN** docparse emits four matched blocks in exactly reverse ground-truth order
- **THEN** `reading_order` = 0

#### Scenario: One block
- **WHEN** a page has one matched block
- **THEN** no `reading_order` row is written for it

### Requirement: Class and source slices
Aggregates SHALL also be produced for `doc_class × source × condition × model` and for `doc_class × variant × model`, each with n and a 95% bootstrap CI. Rows with `source=synthetic` SHALL appear only in slices keyed by source and SHALL be excluded from every slice that pools sources.

#### Scenario: Class × source table
- **WHEN** scores exist for `government_form` from `thaipdf` and `thaiocrbench`
- **THEN** the `doc_class_source_condition_model` table has rows for each source and condition with n and CI

#### Scenario: Synthetic excluded from pooled slices
- **WHEN** synthetic invoice scores exist
- **THEN** the `task_model` table excludes them, and the `doc_class_source_condition_model` table shows them under `source=synthetic`
