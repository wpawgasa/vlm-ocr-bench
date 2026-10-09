# Spec Delta

## ADDED Requirements

### Requirement: Born-digital PDF corpus ingestion
The system SHALL ingest a dataset of kind `pdfcorpus`: a directory tree `<class>/[<variant>/]<file>.pdf` of PDFs and images. Each file SHALL get its document class from the directory, its optional variant from the sub-directory, and its routing (`digital`, `scanned`, `photo`) as for statements. Digital pages SHALL have their ToUnicode maps repaired before extraction, and SHALL get text-layer ground truth with word boxes when the class-neutral usability check passes, else `gt_kind=none`. Every page SHALL be materialized under the three conditions with the existing deterministic degradations. A directory whose name is not a registry class SHALL make `prepare` exit non-zero naming it.

#### Scenario: Government PDF ingested
- **WHEN** `data/thaipdf/government_form/letter-0012.pdf` is a 3-page digital PDF with readable text
- **THEN** three samples are written with `doc_class=government_form`, `source=thaipdf`, `gt_kind=text_layer`, each under `clean`, `scan_low` and `photo`

#### Scenario: Unknown class directory
- **WHEN** the tree contains a directory `memo/`
- **THEN** `prepare` exits non-zero naming `memo` and the class directory

### Requirement: Class-neutral text-layer usability
The text-layer usability check SHALL NOT depend on statement keywords. It SHALL reject a page whose text contains private-use codepoints after repair, whose dictionary-word share of alphabetic tokens is below a configured threshold, or whose Thai-or-Latin letter share is below a configured threshold. On the statement corpus it SHALL reproduce the existing usable/unusable split exactly. The statement keyword test SHALL remain as an additional check for `bank_statement` only.

#### Scenario: Statement split preserved
- **WHEN** the check runs on the 130 digital statement pages of run `2026-09-22-a`
- **THEN** exactly the 97 pages usable today pass and the 33 others fail

#### Scenario: Glyph-code text rejected without keywords
- **WHEN** a government PDF's text layer extracts as a substitution cipher of ASCII letters
- **THEN** the page gets `gt_kind=none` and is counted in `text_layer_unusable`

### Requirement: Table ground truth from text layers
For a digital page with a usable text layer, the system SHALL attempt ruled-table extraction and SHALL store the result as table ground truth only when the table is rectangular with a non-empty header row and the corpus's table spot-check has passed. Until that check passes, such pages SHALL score text metrics only, and the prepare summary SHALL count `text_layer_tables_pending`.

#### Scenario: Spot-check not yet passed
- **WHEN** the `thaipdf` corpus has no recorded table spot-check
- **THEN** no table ground truth is written for its pages, and `prepare_summary.json` reports the pending count

#### Scenario: Spot-check passed
- **WHEN** the recorded spot-check for `thaipdf` shows ≥ 95% cell agreement
- **THEN** rectangular ruled tables on its pages are written as table ground truth with `gt_kind=html`

### Requirement: Open structure sets
The system SHALL ingest a dataset of kind `structure_set` from a public host, capped per set with a seeded sample, evaluation splits only. The ground truth SHALL be limited to the structure the set provides: layout blocks with categories mapped to the harness's block types, reading order where present, table HTML, and form fields mapped to a registry class's fields where a mapping is configured. Text ground truth from these sets SHALL be stored but marked `structure_only`, so that it is reported without a verdict. A set SHALL NOT be loaded unless its entry in the source record (below) exists and lists `evaluation` among its permitted uses.

#### Scenario: DocLayNet pages prepared
- **WHEN** `structure_sets.yaml` lists DocLayNet with cap 200
- **THEN** 200 seeded pages are written with `gt_kind=layout`, categories mapped to `text, title, table, figure, header, footer, other`, and `source=doclaynet`

#### Scenario: Missing licence entry
- **WHEN** a set is listed in the dataset config but absent from the source record
- **THEN** `prepare` exits non-zero naming the set and the source record file

### Requirement: Source licence record
The repository SHALL keep a committed source record listing, for every dataset source: name, URL, licence, permitted uses (`evaluation`, `training`, `redistribution`), retrieval date and the person who confirmed it. `prepare` SHALL refuse a source with no entry, and the training-set builder SHALL refuse a source whose permitted uses lack `training`. The record SHALL hold no document content.

#### Scenario: Non-commercial set in training
- **WHEN** the TrOCR training builder is given crops from a source whose permitted uses are `[evaluation]`
- **THEN** the build fails naming the source and its permitted uses

### Requirement: Document class in the manifest
The manifest SHALL carry `doc_class` (a registry class id or null) and SHALL accept `source` values for every configured dataset kind. Existing manifests without `doc_class` SHALL still validate. Statement rows SHALL carry `doc_class=bank_statement`.

#### Scenario: Old manifest still valid
- **WHEN** a manifest row from run `2026-09-22-a` without `doc_class` is read back
- **THEN** it validates with `doc_class=null`

#### Scenario: Synthetic dataset rows
- **WHEN** a `synthetic` dataset is prepared
- **THEN** its rows carry `source=synthetic`, `doc_class` from the template's class, and `gt_kind=text_layer` or `json` as the template provides
