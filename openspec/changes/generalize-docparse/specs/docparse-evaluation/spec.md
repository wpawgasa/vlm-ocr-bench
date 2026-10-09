# Spec Delta

## ADDED Requirements

### Requirement: Frozen eval list per class
The system SHALL keep one committed, versioned eval list per document class (`configs/docparse/eval/<class>.v<N>.yaml`), containing file ids, their source, an optional harness id, an optional variant and a content hash, and no document content. The statement list `configs/docparse/eval_files.v1.yaml` SHALL be read through the same loader without being modified. A class's list SHALL be frozen before any tuning of docparse on that class; a change SHALL require a new version. Every evaluation result SHALL record the version of every list it used. One loader call SHALL return the union of all current lists' file ids for leak checks.

This supersedes "Frozen eval file list" of `add-docparse-system`, which covered statements only.

#### Scenario: Per-class lists loaded
- **WHEN** the eval directory holds `invoice.v1.yaml`, `government_form.v1.yaml`, `tabular_report.v1.yaml` and the statement v1 file
- **THEN** the loader exposes four class lists with their versions, and the union of ids has no duplicates

#### Scenario: Statement list untouched
- **WHEN** the per-class loader reads `configs/docparse/eval_files.v1.yaml`
- **THEN** it returns the same six files as the statement loader, and the file's bytes are unchanged

#### Scenario: Content in a list refused
- **WHEN** an eval list entry carries a `text` or `fields` key
- **THEN** loading fails naming the list and the entry

### Requirement: Success criteria per class
For each class and condition, the evaluation SHALL compare docparse with the best single VLM on the class's eval samples, with paired cluster-bootstrap 95% CIs (clusters = files) of the difference on the class's metrics:
- `bank_statement`: as specified in `add-docparse-system` (statement field F1, CER, Check B, Q/A);
- `invoice` and `government_form`: KIE field F1 over the class's fields (primary), page CER, Q/A exact match, provenance and abstention; declared better only where the field-F1 lower bound is above 0 and the CER upper bound is below 0;
- `tabular_report`: table TEDS (primary), page CER, layout block F1 and reading order; declared better only where the TEDS lower bound is above 0 and the CER upper bound is at or below 0;
- `unknown` pages: page CER (primary), document-structure similarity, TEDS, layout block F1 and reading order; declared better only where the CER upper bound is below 0 and the TEDS lower bound is at or above 0.

The best single VLM SHALL be chosen per class by its mean on the class's primary metric over the same samples. A class with fewer than 10 eval files SHALL be reported with its n and no verdict.

#### Scenario: Invoice verdict
- **WHEN** on `clean` the invoice field-F1 difference CI is [0.03, 0.09] and the CER difference CI is [-0.020, -0.004]
- **THEN** the report states docparse is better on `invoice`/`clean`

#### Scenario: Too few files
- **WHEN** the invoice list has 7 files
- **THEN** the report shows the invoice metrics with n = 7 and the verdict "n too small"

### Requirement: Generic cross-class criterion
The evaluation SHALL compute, over all real eval pages of all classes, the paired cluster-bootstrap 95% CI of docparse minus the per-class best VLM on page CER and, over pages with a table, on TEDS. docparse SHALL be declared generally better only where the CER difference's upper bound is below 0 and the TEDS difference's lower bound is at or above 0. The generic verdict SHALL be shown beside every per-class verdict, and SHALL never replace a failed per-class verdict.

#### Scenario: Generic verdict with a failing class
- **WHEN** the generic CIs meet the rule, and `tabular_report` fails its own rule on `photo`
- **THEN** the report shows "generally better" and, in the same table, "not shown better" for `tabular_report`/`photo`

### Requirement: Reporting per class, source and condition
Every reported metric SHALL be broken down by class × source × condition, with n and a 95% CI, and by variant where the class has variants. Headline tables SHALL include real sources only. Synthetic results SHALL appear in a separate table labelled synthetic and SHALL NOT enter any aggregate with real sources. Every number SHALL trace to a committed or DVC-tracked artifact.

#### Scenario: Synthetic kept apart
- **WHEN** the evaluation includes synthetic invoice pages
- **THEN** the headline invoice table excludes them, and a table titled "synthetic" reports them with their template ids

#### Scenario: Per-source rows
- **WHEN** `government_form` has eval pages from `thaipdf` and `thaiocrbench`
- **THEN** the report shows the class metrics for each source separately and for both together

### Requirement: Per-class Q/A sets
For every class with fields or tables, the system SHALL generate a templated Q/A set from the class's eval ground truth, covering each query type the class declares, with at least 10% unanswerable items, each item recording its gold answer and source fields or rows. Scoring SHALL be exact match after normalization, provenance correctness, and abstention precision and recall.

#### Scenario: Invoice Q/A coverage
- **WHEN** the invoice Q/A set is generated
- **THEN** it contains `field`, `table_filter_aggregate` and `cross_check` items, and at least 10% ask for an absent optional field or a non-existent line item

### Requirement: docparse scored on every harness task
The benchmark harness SHALL be able to score docparse on every task it defines, by mapping the parsed document to the task's normalized output: full text in reading order for full-page OCR, table HTML for table parsing, markdown with headings for document parsing, one extraction per key for key information tasks, the mapped class label for classification, the statement mapper for statements, and blocks with boxes and categories for layout. The same scorers, slices and Check B SHALL apply unchanged.

#### Scenario: Scored on ThaiOCRBench KIE
- **WHEN** docparse is run on the ThaiOCRBench KIE samples of run `2026-09-22-a`
- **THEN** `ocrbench score` writes `kie_f1` rows for docparse next to the baselines, one extraction per benchmark key

#### Scenario: Scored on layout
- **WHEN** docparse is run on DocLayNet pages
- **THEN** `ocrbench score` writes `layout_f1` rows for docparse and for the layout backend alone
