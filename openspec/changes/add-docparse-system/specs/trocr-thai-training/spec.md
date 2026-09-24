# Spec Delta

## Purpose

Fine-tunes the Thai TrOCR checkpoint on statement-shaped lines only when it has failed the promotion gate, without leaking the evaluation files into training.

## ADDED Requirements

### Requirement: Fine-tuning only after a failed gate
The system SHALL start a fine-tuning run only when given a gate result that shows the configured `trocr` checkpoint failed the promotion gate. Without such a result, it SHALL refuse and SHALL name the missing or passing gate artifact.

#### Scenario: Gate passed
- **WHEN** a fine-tune is requested, and the latest gate result shows muocr passed
- **THEN** the run is refused with a message citing that gate result, and no training data is built

#### Scenario: Gate failed
- **WHEN** the latest gate result shows muocr failed the relative condition
- **THEN** the fine-tune may proceed

### Requirement: Start from the base checkpoint
A fine-tune SHALL initialize from the configured base checkpoint (initially `muocr-base-26m-stage2-finetuned-20240820-v1`). It SHALL keep that checkpoint's tokenizer and input geometry. Training targets that the tokenizer cannot encode without unknown tokens SHALL be rejected and counted in the build report.

#### Scenario: Unencodable target
- **WHEN** a text-layer label contains a private-use glyph that normalization cannot map, and that the tokenizer encodes as unknown
- **THEN** the sample is rejected, and the build report counts it

### Requirement: Training data sources
The training set SHALL combine two sources:
- synthetic statement-shaped Thai lines rendered with the installed Thai fonts, with degradations matching the benchmark's `scan_low` and `photo` conditions;
- real line crops labelled from a usable PDF text layer, or pseudo-labelled where two independent readers agree exactly after normalization.

Every sample SHALL record its source, font or file id, and degradation.

#### Scenario: Pseudo-label agreement
- **WHEN** two readers disagree on a real crop from a scanned page
- **THEN** that crop is excluded from training

### Requirement: No eval leakage
The system SHALL refuse to build a training set containing any crop from a file in the frozen eval file list. It SHALL report the offending file ids.

#### Scenario: Eval file present in real crops
- **WHEN** the real-crop source includes a file listed in the frozen eval set
- **THEN** the build fails with that file id, and no training set is written

### Requirement: Reproducible training outputs
A fine-tuning run SHALL write:
- the checkpoint;
- the base checkpoint id;
- the resolved config;
- the training-set manifest hash;
- held-out CER per condition.

Checkpoints SHALL be stored outside git. Given the same config and seed, a run SHALL produce the same training-set manifest. The fine-tuned checkpoint SHALL become the `trocr` reader only by passing the same promotion gate.

#### Scenario: Rebuild determinism
- **WHEN** the training set is built twice with the same config and seed
- **THEN** the two manifests are identical
