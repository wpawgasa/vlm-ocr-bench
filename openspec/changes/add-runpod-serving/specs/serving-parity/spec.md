# Spec Delta

## Purpose

Proves that a Runpod endpoint produces the same outputs as the reference self-hosted serving of the same model on a fixed public sample set, within stated tolerances, before the endpoint may serve client pages.

## ADDED Requirements

### Requirement: Fixed public parity sample set
The system SHALL keep a committed, versioned list of parity samples (`configs/runpod/parity_samples.v<N>.yaml`) made of ThaiOCRBench sample ids from a prepared run, covering full-page OCR, line OCR, table, handwriting and KIE tasks in the `clean` condition. The list SHALL contain ids only. The loader SHALL refuse any id whose source is client data (bank statements).

#### Scenario: Statement id refused
- **WHEN** the parity list contains a sample id from the `bankstmt` source
- **THEN** loading the list fails naming the id, and no request is sent

#### Scenario: Coverage checked
- **WHEN** the parity list is loaded
- **THEN** it has at least 10 samples for each of the five tasks and reports its version

### Requirement: Reference predictions recorded once
The system SHALL record the reference predictions of a model by running the parity set against the self-hosted reference serving with `ocrbench runpod parity <model> --record-reference`. The reference SHALL record the model's vLLM version, prompt version per row, sampling and the serving image digest, and SHALL be stored outside git (DVC-tracked) under the run directory. Recording SHALL refuse an endpoint that is not self-hosted.

#### Scenario: Reference recorded
- **WHEN** `--record-reference` runs against the H100 compose serving of dots.ocr
- **THEN** `data/runs/parity-dotsocr/reference.jsonl` holds one row per parity sample with the vLLM version and prompt version, and the file is DVC-tracked

#### Scenario: Reference from a Runpod host refused
- **WHEN** `--record-reference` runs while `DOTSOCR_BASE_URL` points at `*.api.runpod.ai`
- **THEN** the command exits non-zero without sending requests

### Requirement: Parity gate
`ocrbench runpod parity <model>` SHALL run the parity set against the model's Runpod endpoint with the same model configuration as the reference and compare row by row. The gate SHALL fail before scoring if any row's prompt version differs from the reference or the endpoint's vLLM version differs from the reference's. It SHALL pass only when all of the following hold:
- rows with an error on either side are at most 2% of the set, and at most 1 row has an error on one side only;
- the median pairwise character error rate between the reference text and the endpoint text (canonical text) is at most 0.01 and the 95th percentile at most 0.05;
- the mean of the task's primary ground-truth metric differs from the reference by at most 0.01.

The exact-match rate SHALL be reported but SHALL NOT be required. The gate SHALL write a report with per-row values and the verdict, and SHALL copy the verdict into the model's latest release record.

#### Scenario: Pass within tolerance
- **WHEN** 60 rows compare with median pairwise CER 0.003, p95 0.02, no errors and a mean CER difference of 0.004
- **THEN** the verdict is `pass`, and the release record's parity field becomes `pass` with the report path

#### Scenario: Prompt version drift
- **WHEN** one row's prompt version on the endpoint run differs from the reference row
- **THEN** the gate fails before comparing texts, naming the sample id and both versions

#### Scenario: Tail mismatch
- **WHEN** the median pairwise CER is 0.002 but the 95th percentile is 0.12
- **THEN** the verdict is `fail`, and the report lists the rows above 0.05

#### Scenario: One-sided error
- **WHEN** two rows error on the endpoint and succeed on the reference
- **THEN** the verdict is `fail` naming both sample ids

### Requirement: Promotion requires parity
An endpoint SHALL NOT be marked eligible for client data unless its latest release has a `pass` parity verdict for the same image digest. Any new release SHALL reset the verdict to `pending`.

#### Scenario: Promote without parity
- **WHEN** `deploy dotsocr --promote` runs and the latest release's parity is `pending`
- **THEN** the command exits non-zero and the endpoint stays unpromoted

#### Scenario: New release resets parity
- **WHEN** a new image digest is deployed for a promoted model
- **THEN** the new release record has parity `pending`, and the endpoint's promotion is suspended until the gate passes again
