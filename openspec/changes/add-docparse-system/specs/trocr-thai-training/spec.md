# Spec Delta

## Purpose

Produces a Thai line-recognition model from synthetic and real statement lines without leaking the evaluation files into training.

## ADDED Requirements

### Requirement: Training data sources
The training set SHALL combine two sources:
- synthetic Thai lines rendered with the installed Thai fonts, with degradations matching the benchmark's `scan_low` and `photo` conditions;
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

### Requirement: Character-level Thai vocabulary
The recognizer SHALL output characters from a fixed vocabulary. The vocabulary SHALL cover the Thai block including Thai digits, ASCII, and currency and punctuation symbols. Targets SHALL be NFC-normalized. A training target containing a character outside the vocabulary SHALL be rejected and counted in the build report.

#### Scenario: Out-of-vocabulary character
- **WHEN** a text-layer label contains a private-use glyph that normalization cannot map
- **THEN** the sample is rejected, and the build report counts it

### Requirement: Reproducible training outputs
A training run SHALL write:
- the checkpoint;
- the vocabulary;
- the resolved config;
- the training-set manifest hash;
- held-out CER per condition.

Checkpoints SHALL be stored outside git. Given the same config and seed, a run SHALL produce the same training-set manifest.

#### Scenario: Rebuild determinism
- **WHEN** the training set is built twice with the same config and seed
- **THEN** the two manifests are identical
