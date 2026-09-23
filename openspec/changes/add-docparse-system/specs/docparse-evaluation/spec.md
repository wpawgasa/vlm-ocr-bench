# Spec Delta

## Purpose

Measures whether docparse beats the best single VLM on the same data, using a frozen, leak-free eval set and a falsifiable success criterion.

## ADDED Requirements

### Requirement: Frozen eval file list
The system SHALL keep a committed eval file list of file ids only. The list SHALL contain:
- every statement file with at least one usable text-layer page;
- the files in the manual anchor set.

It SHALL be frozen before any docparse tuning. Changing it SHALL require a new list version. The version SHALL be recorded in every evaluation result.

#### Scenario: Eval result records list version
- **WHEN** an evaluation runs
- **THEN** its output records the eval list version and the file count

### Requirement: docparse scored by the benchmark harness
The benchmark harness SHALL be able to run docparse as a model over an existing prepared run. The harness SHALL write docparse's predictions in the same schema as the other models, so that the same scorers, slices and Check B apply unchanged.

#### Scenario: Score alongside baselines
- **WHEN** docparse is run on run `2026-09-22-a`, and `ocrbench score` is executed
- **THEN** the score outputs contain docparse rows next to dots.ocr, typhoon-ocr1.5 and PaddleOCR-VL rows for the same samples

### Requirement: Statement Q/A set
The system SHALL generate a templated Q/A set from eval statements with ground truth. The set SHALL cover every extraction query type. At least 10% of its queries SHALL be unanswerable. Each item SHALL record the gold answer and its source rows or fields. Extraction SHALL be scored on:
- exact match after normalization;
- provenance correctness;
- abstention precision and recall.

#### Scenario: Unanswerable item scored
- **WHEN** the extractor answers an unanswerable item with a value
- **THEN** the item counts as a false answer in abstention precision

### Requirement: Success criterion
The evaluation SHALL compare docparse with the best single VLM on the same eval samples. For each condition (`clean`, `scan_low`, `photo`), it SHALL report the paired bootstrap 95% CI of the difference in statement field F1 and in CER, reported per bank.

docparse SHALL be declared better only where the lower CI bound of the field-F1 difference is above 0 **and** the upper CI bound of the CER difference is below 0. The report SHALL also include the Check B row-consistency rate over all statement pages.

#### Scenario: Inconclusive result
- **WHEN** the field-F1 difference CI on `photo` is [-0.01, 0.04]
- **THEN** the report states "not shown better" for `photo`
