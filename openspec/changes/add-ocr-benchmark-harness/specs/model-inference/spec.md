# Spec Delta

## Purpose

Runs each OCR model under evaluation over the manifest through its vLLM OpenAI-compatible endpoint. Records raw output, token logprobs, timing and failures, and probes each model's structural output capabilities for Q4.

## ADDED Requirements

### Requirement: Remote-only model access
The harness SHALL reach models only through OpenAI-compatible HTTP endpoints configured per model: `$TELEOCR_BASE_URL`, `$DOTSOCR_BASE_URL`, and optionally `$TYPHOON_BASE_URL`. It SHALL NOT load model weights in-process. TeleOCR and dots.ocr adapters SHALL be provided. A `typhoon-ocr1.5-2b` adapter SHALL be available and run only when listed in `configs/run.yaml`.

#### Scenario: Endpoint unreachable at start
- **WHEN** `infer` starts and a selected model's endpoint health check fails
- **THEN** the command exits non-zero before sending any page, naming the model and URL

### Requirement: Prompt policy: native mode, benchmark-question fallback
TeleOCR, dots.ocr and typhoon-ocr1.5 are task-specific models that are trained on fixed prompts and do not follow free-form instructions. Each model SHALL therefore have a per-task *request plan* in its config, with a version tag. The plan SHALL use the model's official prompt or pipeline wherever one matches the task:

| Task | TeleOCR | dots.ocr | typhoon-ocr1.5 |
| --- | --- | --- | --- |
| `ocr_fullpage`, `docparse`, `statement` | official two-stage pipeline | `prompt_layout_all_en` | official single prompt |
| `ocr_line` (text recognition), `handwriting` | text prompt on the whole image | `prompt_ocr` | official single prompt |
| `ocr_line` (fine-grained) | crop of the question's region, then text prompt | `prompt_grounding_ocr` with the region as a pixel bbox | crop, then official single prompt |
| `table` | OTSL table prompt on the whole image | `prompt_layout_all_en`, table blocks | official single prompt |
| `kie`, `kie_map`, `classify` | benchmark question | benchmark question | benchmark question |

Where no native mode matches (`kie`, `kie_map`, `classify`), the plan SHALL send the sample's benchmark `question` verbatim. Fine-grained regions SHALL be read from the question's coordinates, which are on a 0–1000 scale. The request plan's version SHALL be written into every prediction row.

#### Scenario: Native mode used
- **WHEN** a `ocr_fullpage` sample is inferred with dots.ocr
- **THEN** the request uses `prompt_layout_all_en`, not the sample's benchmark question

#### Scenario: Fallback sends the benchmark question
- **WHEN** a `kie` sample is inferred with any of the three models
- **THEN** the request text is the sample's `question`, unchanged

#### Scenario: Unparseable fallback output is a miss
- **WHEN** a model's reply to a `kie` fallback request is not parseable JSON
- **THEN** the prediction is kept with `parse_error` set and every field of the sample is scored as missed

#### Scenario: Prompt version recorded
- **WHEN** a prediction is written
- **THEN** it carries the `prompt_version` of the request plan used

### Requirement: Deterministic request parameters
Every inference request SHALL use `temperature=0`, `seed=0` and token logprobs. `max_tokens` SHALL default to 4096. A model's official decoding and pre-processing settings from its config SHALL take precedence: TeleOCR's presence/frequency penalties and `no_repeat_ngram_size=100`, and typhoon's `max_tokens=10000` and 1800 px maximum image side. The settings used SHALL appear in `config.infer.yaml`. The accuracy run's client concurrency SHALL be 8 by default and configurable.

#### Scenario: Logprobs captured
- **WHEN** the endpoint returns token logprobs
- **THEN** the prediction stores one logprob per completion token
- **WHEN** the endpoint returns no logprobs
- **THEN** the prediction stores an empty list and inference continues

#### Scenario: Official settings applied
- **WHEN** a TeleOCR block request is sent
- **THEN** it carries TeleOCR's configured presence penalty, frequency penalty and `no_repeat_ngram_size`

### Requirement: Multi-request page pipelines
A page MAY need several requests (TeleOCR: one layout request plus one request per detected block), but it SHALL still produce exactly one prediction row.

- `latency_ms` SHALL be the page's wall-clock time.
- Token counts SHALL be summed across the page's requests.
- The row SHALL record `n_requests`.

If the layout request fails after its retry, the page row SHALL have `error` set. If some block requests fail, those blocks SHALL be left empty, `raw.error` SHALL state how many blocks failed, and the page SHALL still be scored.

#### Scenario: Two-stage page
- **WHEN** TeleOCR's layout request returns 6 blocks for a page
- **THEN** 7 requests are sent, one prediction row is written with `n_requests=7`, and its latency covers all 7

#### Scenario: Partial block failure
- **WHEN** 1 of 6 block requests fails twice
- **THEN** the page row has an empty block, `raw.error` of "1 of 6 blocks failed", and it is scored

### Requirement: Timeout, retry and no silent drops
Each request SHALL time out after 120 s and SHALL be retried once. If a page's retry also fails, the system SHALL write a prediction row with a non-null `error` and an empty output. Scoring SHALL count that row as a full miss in every metric. The number of manifest rows SHALL equal the number of prediction rows per model.

#### Scenario: Double failure recorded
- **WHEN** a page times out twice
- **THEN** `predictions.jsonl` contains a row for it with `error` set
- **AND** the page's score rows are worst-case (CER = 1, fields missed), not omitted

#### Scenario: Error-rate summary
- **WHEN** `infer` completes
- **THEN** it reports the per-model error rate and exits non-zero if the manifest and prediction row counts differ

### Requirement: Prediction record
Each row of `predictions.jsonl` SHALL contain `sample_id`, `condition`, `model`, `prompt_version`, `raw` (verbatim text of every request, token logprobs, error), `normalized` (the parsed page), `latency_ms`, `n_requests`, `prompt_tokens` and `completion_tokens`. `infer` SHALL be resumable: rows already present for `(sample_id, condition, model)` without error SHALL be skipped on re-run.

#### Scenario: Resume after interruption
- **WHEN** `infer` is interrupted and re-run with the same run id
- **THEN** only missing or errored `(sample_id, condition, model)` rows are requested again

### Requirement: Capability probe
`ocrbench probe` SHALL run 5 fixed statement pages through each model. It SHALL fill a matrix with `yes`, `partial` or `no` for these capabilities: reading-order text, table as rows × cells, block bounding boxes, page index in output, JSON field extraction on request, and token logprobs available. The matrix SHALL be written to the run directory for the report.

#### Scenario: Probe output
- **WHEN** `probe` completes for TeleOCR and dots.ocr
- **THEN** `runs/<run_id>/probe.json` holds a value in {yes, partial, no} for every (capability, model) pair, plus the evidence page ids
