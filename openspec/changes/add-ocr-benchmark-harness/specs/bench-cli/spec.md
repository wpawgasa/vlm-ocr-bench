# Spec Delta

## Purpose

Provides the `ocrbench` command-line entry point, the run configuration files and the `runs/<run_id>/` artifact layout, so every benchmark stage can be run, re-run and reproduced on its own.

## ADDED Requirements

### Requirement: Command set
The system SHALL provide an `ocrbench` command with the subcommands `prepare`, `infer`, `probe`, `score`, `calibrate`, `bench-latency` and `report`. Each subcommand SHALL be listed with a one-line description by `ocrbench --help`.

#### Scenario: Help lists all stages
- **WHEN** a user runs `ocrbench --help`
- **THEN** the output lists `prepare`, `infer`, `probe`, `score`, `calibrate`, `bench-latency` and `report`
- **AND** the command exits with status 0

### Requirement: Run directory as the stage contract
Every stage SHALL read its inputs from and write its outputs to `runs/<run_id>/` under the data directory (`$OCRBENCH_DATA_DIR`, defaulting to `./data`). Outputs SHALL be JSON Lines files (`manifest.jsonl`, `predictions.jsonl`, `scores.jsonl`, `fields.jsonl`, `calibration.jsonl`, `latency.jsonl`) plus `thresholds.json` and a `report/` directory. A stage SHALL NOT depend on in-memory state from another stage.

#### Scenario: Stage re-run in isolation
- **WHEN** `ocrbench score --run-id R` is run twice on a run directory with unchanged `manifest.jsonl` and `predictions.jsonl`
- **THEN** both runs produce byte-identical `scores.jsonl` and `fields.jsonl`

#### Scenario: Missing upstream artifact
- **WHEN** `ocrbench score --run-id R` is run and `runs/R/predictions.jsonl` does not exist
- **THEN** the command exits non-zero with a message naming the missing file and the stage that produces it

### Requirement: Declarative run configuration
The system SHALL read which models, datasets and conditions to run from `configs/run.yaml`. Model settings (endpoint env var, served model name, per-task prompts with version tags, output parser) SHALL come from `configs/models/<model>.yaml`. Dataset settings (subset filters, caps, seed, bank layouts, degradation profiles) SHALL come from `configs/datasets/<dataset>.yaml`. When `prepare` runs, the resolved run and dataset settings SHALL be written to `runs/<run_id>/config.resolved.yaml`. When `infer` runs, the resolved settings of the models it used SHALL be written to `runs/<run_id>/config.infer.yaml`. Keeping them separate means a model-config change never invalidates prepared data.

#### Scenario: Resolved config snapshot
- **WHEN** `ocrbench prepare --config configs/run.yaml --run-id R` completes
- **THEN** `runs/R/config.resolved.yaml` exists and contains the run, dataset and condition settings actually used, and no model settings

#### Scenario: Invalid configuration
- **WHEN** `configs/run.yaml` names a model with no matching `configs/models/<model>.yaml`
- **THEN** the command exits non-zero before doing any work and names the missing model config

### Requirement: Reproduce sequence
The full benchmark SHALL be reproducible with the documented command sequence `prepare → infer → score → calibrate → bench-latency → report` for a given run id. The same inputs and configuration SHALL give the same manifest, sample selection and degraded images.

#### Scenario: Deterministic preparation
- **WHEN** `prepare` is run twice with the same config into two different run ids
- **THEN** both manifests list the same `(sample_id, condition)` rows
- **AND** corresponding degraded images are pixel-identical
