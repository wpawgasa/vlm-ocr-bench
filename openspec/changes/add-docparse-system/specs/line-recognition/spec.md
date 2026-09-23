# Spec Delta

## Purpose

Reads each line segment with a configurable line reader and decides, by a measured gate, whether the Thai TrOCR reader replaces the baseline reader.

## ADDED Requirements

### Requirement: Line reader contract
Every line reader SHALL accept a batch of segment crops. For each crop, it SHALL return the text (NFC), per-character confidences, or none if the backend has none, and the reader name and version. A reader failure on one segment SHALL produce an error result for that segment only.

#### Scenario: Batch read
- **WHEN** 300 segments of one page are read
- **THEN** 300 results are returned in input order, each with text, confidences or none, and the reader's name and version

#### Scenario: Single segment failure
- **WHEN** one crop is empty after clipping
- **THEN** that segment's result is an error, and the other results are unaffected

### Requirement: Baseline and TrOCR readers
The system SHALL provide two readers, `paddle_crop` (the PaddleOCR-VL crop "OCR:" read) and `trocr` (the Thai TrOCR). The reader SHALL be chosen in configuration. The default SHALL be `paddle_crop` until the TrOCR gate passes.

#### Scenario: Reader selection
- **WHEN** the configuration sets `line_reader: trocr` and a TrOCR checkpoint is configured
- **THEN** segments are read by TrOCR, and the artifact records the checkpoint id

### Requirement: TrOCR promotion gate
The system SHALL provide a gate evaluation on held-out lines from non-training files. It SHALL report CER for each reader, split into clean and degraded conditions. TrOCR SHALL pass only if all of these hold:
- its clean CER is ≤ 2%;
- its degraded CER is ≤ 6%;
- its CER on the same lines is ≤ `paddle_crop`'s in each condition.

The thresholds SHALL be configurable. The gate result SHALL be written as an artifact.

#### Scenario: TrOCR beats absolute targets but loses to the baseline
- **WHEN** TrOCR's CER is 1.8% clean and 5.5% degraded, and `paddle_crop` scores 1.5% and 5.0%
- **THEN** the gate fails, and the default reader stays `paddle_crop`
