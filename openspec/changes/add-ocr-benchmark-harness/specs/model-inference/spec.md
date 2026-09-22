# Spec Delta

## Purpose

Runs each OCR model under evaluation over the manifest through its vLLM OpenAI-compatible endpoint. Records raw output, token logprobs, timing and failures, and probes each model's structural output capabilities for Q4.

## ADDED Requirements

### Requirement: Remote-only model access
The harness SHALL reach models only through OpenAI-compatible HTTP endpoints configured per model: `$TELEOCR_BASE_URL`, `$DOTSOCR_BASE_URL`, and optionally `$TYPHOON_BASE_URL`. It SHALL NOT load model weights in-process. TeleOCR and dots.ocr adapters SHALL be provided. A `typhoon-ocr1.5-2b` adapter SHALL be available and run only when listed in `configs/run.yaml`.

#### Scenario: Endpoint unreachable at start
- **WHEN** `infer` starts and a selected model's endpoint health check fails
- **THEN** the command exits non-zero before sending any page, naming the model and URL

### Requirement: Task-specific versioned prompts
Each model SHALL have one prompt per task: `ocr_fullpage`, `ocr_line`, `handwriting`, `table`, `docparse`, `kie`, `kie_map`, `classify`, `statement`. The prompts SHALL be defined in that model's config with a version tag.

- For `kie` and `kie_map`, the prompt SHALL include the sample's field list and request JSON with `null` for absent fields.
- For `statement`, the prompt SHALL request the statement page schema.

The prompt version SHALL be written into every prediction row.

#### Scenario: Prompt version recorded
- **WHEN** a prediction is written
- **THEN** it carries the `prompt_version` of the prompt used

#### Scenario: KIE field list in prompt
- **WHEN** a KIE sample with fields `id_number` and `total_amount` is inferred
- **THEN** the request prompt names both fields and asks for JSON with nulls for absent fields

### Requirement: Deterministic request parameters
Every inference request SHALL use `temperature=0`, `seed=0`, `max_tokens=4096` and `logprobs=1`. The accuracy run's client concurrency SHALL be 8 by default and configurable.

#### Scenario: Logprobs captured
- **WHEN** the endpoint returns token logprobs
- **THEN** the prediction stores one logprob per completion token
- **WHEN** the endpoint returns no logprobs
- **THEN** the prediction stores an empty list and inference continues

### Requirement: Timeout, retry and no silent drops
Each page request SHALL time out after 120 s and SHALL be retried once. If the retry also fails, the system SHALL write a prediction row with a non-null `error` and an empty output. Scoring SHALL count that row as a full miss in every metric. The number of manifest rows SHALL equal the number of prediction rows per model.

#### Scenario: Double failure recorded
- **WHEN** a page times out twice
- **THEN** `predictions.jsonl` contains a row for it with `error` set
- **AND** the page's score rows are worst-case (CER = 1, fields missed), not omitted

#### Scenario: Error-rate summary
- **WHEN** `infer` completes
- **THEN** it reports the per-model error rate and exits non-zero if the manifest and prediction row counts differ

### Requirement: Prediction record
Each row of `predictions.jsonl` SHALL contain `sample_id`, `condition`, `model`, `prompt_version`, `raw` (verbatim text, token logprobs, error), `normalized` (the parsed page), `latency_ms`, `prompt_tokens` and `completion_tokens`. `infer` SHALL be resumable: rows already present for `(sample_id, condition, model)` without error SHALL be skipped on re-run.

#### Scenario: Resume after interruption
- **WHEN** `infer` is interrupted and re-run with the same run id
- **THEN** only missing or errored `(sample_id, condition, model)` rows are requested again

### Requirement: Capability probe
`ocrbench probe` SHALL run 5 fixed statement pages through each model. It SHALL fill a matrix with `yes`, `partial` or `no` for these capabilities: reading-order text, table as rows × cells, block bounding boxes, page index in output, JSON field extraction on request, and token logprobs available. The matrix SHALL be written to the run directory for the report.

#### Scenario: Probe output
- **WHEN** `probe` completes for TeleOCR and dots.ocr
- **THEN** `runs/<run_id>/probe.json` holds a value in {yes, partial, no} for every (capability, model) pair, plus the evidence page ids
