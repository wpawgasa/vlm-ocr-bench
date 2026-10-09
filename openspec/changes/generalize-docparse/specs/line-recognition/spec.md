# Spec Delta

## ADDED Requirements

### Requirement: Gate breakdown per class
The promotion gate SHALL report CER per reader per document class, on held-out text-layer lines of that class, beside the pooled numbers. The pass or fail decision SHALL stay global. The gate artifact SHALL name every class with fewer than 500 held-out lines as `insufficient` instead of reporting a CER for it.

#### Scenario: Per-class rows in the gate artifact
- **WHEN** the gate runs with statement, government and report lines available
- **THEN** `gate.json` contains one row per reader per class, each with the line count and CER per condition

#### Scenario: Class with too few lines
- **WHEN** the invoice class has 120 held-out lines
- **THEN** its gate row says `insufficient` and no CER is reported for it

### Requirement: Per-class reader override
A class file MAY set `line_reader` to `paddle_crop` or `trocr`. When set, that reader SHALL read the documents of that class, the readings artifact SHALL record the reader and the override source, and segmentation SHALL use that reader's aspect limit.

#### Scenario: Class override applied
- **WHEN** the pipeline default is `trocr` and `tabular_report.yaml` sets `line_reader: paddle_crop`
- **THEN** a tabular report's readings come from `paddle_crop`, with `max_aspect` 10, and the artifact records `reader_override: class`
