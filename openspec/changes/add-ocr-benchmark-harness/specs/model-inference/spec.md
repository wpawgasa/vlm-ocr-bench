# Spec Delta

## Purpose

Runs each OCR model under evaluation over the manifest through its vLLM OpenAI-compatible endpoint. Records raw output, token logprobs, timing and failures, and probes each model's structural output capabilities for Q4.

## ADDED Requirements

### Requirement: Remote-only model access
The harness SHALL reach models only through HTTP endpoints configured per model: OpenAI-compatible vLLM endpoints (`$DOTSOCR_BASE_URL`, `$TYPHOON_BASE_URL`, `$OVISOCR2_BASE_URL`, `$PADDLEOCR_VL_BASE_URL`, `$TELEOCR_BASE_URL`) and, for a model whose official pipeline is a vendor server, that server's endpoint (`$PADDLE_PIPELINE_URL`, PaddleX's `/layout-parsing` for PaddleOCR-VL-1.6). It SHALL NOT load model weights in-process. Adapters SHALL be provided for dots.ocr, typhoon-ocr1.5-2b, OvisOCR2, PaddleOCR-VL-1.6 and TeleOCR; a model runs only when listed in `configs/run.yaml` or named with `--models`. TeleOCR is screened out of the benchmark (it does not read Thai) and its adapter is kept for the evidence run only.

#### Scenario: Endpoint unreachable at start
- **WHEN** `infer` starts and a selected model's endpoint health check fails
- **THEN** the command exits non-zero before sending any page, naming the model and URL

#### Scenario: Pipeline endpoint checked too
- **WHEN** PaddleOCR-VL is selected and its vLLM endpoint is healthy but `$PADDLE_PIPELINE_URL` is unset or unhealthy
- **THEN** the command exits non-zero before sending any page, naming the model and the pipeline URL

### Requirement: Prompt policy: native mode, benchmark-question fallback
TeleOCR, dots.ocr, typhoon-ocr1.5, OvisOCR2 and PaddleOCR-VL are task-specific models that are trained on fixed prompts and do not follow free-form instructions. Each model SHALL therefore have a per-task *request plan* in its config, with a version tag. The plan SHALL use the model's official prompt or pipeline wherever one matches the task:

| Task | TeleOCR | dots.ocr | typhoon-ocr1.5 | OvisOCR2 | PaddleOCR-VL-1.6 |
| --- | --- | --- | --- | --- | --- |
| `ocr_fullpage`, `docparse`, `statement` | official two-stage pipeline | `prompt_layout_all_en` | official single prompt | official single prompt | official pipeline (PaddleX server) |
| `ocr_line` (text recognition), `handwriting` | text prompt on the whole image | `prompt_ocr` | official single prompt | official single prompt | official pipeline |
| `ocr_line` (fine-grained) | crop of the question's region, then text prompt | `prompt_grounding_ocr` with the region as a pixel bbox | crop, then official single prompt | crop, then official single prompt | crop, then official pipeline |
| `table` | OTSL table prompt on the whole image | `prompt_layout_all_en`, table blocks | official single prompt | official single prompt | official pipeline, table blocks |
| `kie`, `kie_map`, `classify` | benchmark question | benchmark question | benchmark question | benchmark question | benchmark question, sent to its vLLM endpoint |

Where no native mode matches (`kie`, `kie_map`, `classify`), the plan SHALL send the sample's benchmark `question` verbatim. Fine-grained regions SHALL be read from the question's coordinates, which are on a 0–1000 scale. The request plan's version SHALL be written into every prediction row.

#### Scenario: Native mode used
- **WHEN** a `ocr_fullpage` sample is inferred with dots.ocr
- **THEN** the request uses `prompt_layout_all_en`, not the sample's benchmark question

#### Scenario: Fallback sends the benchmark question
- **WHEN** a `kie` sample is inferred with any of the models
- **THEN** the request text is the sample's `question`, unchanged

#### Scenario: Unparseable fallback output is a miss
- **WHEN** a model's reply to a `kie` fallback request is not parseable JSON
- **THEN** the prediction is kept with `parse_error` set and every field of the sample is scored as missed

#### Scenario: Prompt version recorded
- **WHEN** a prediction is written
- **THEN** it carries the `prompt_version` of the request plan used

### Requirement: Deterministic request parameters
Every vLLM inference request SHALL use `temperature=0`, `seed=0` and token logprobs. `max_tokens` SHALL default to 4096. A model's official decoding and pre-processing settings from its config SHALL take precedence: TeleOCR's presence/frequency penalties and `no_repeat_ngram_size=100`, typhoon's `max_tokens=10000` and 1800 px maximum image side, and OvisOCR2's `max_tokens=16384` with the image resized to multiples of 32 within 448²–2880² pixels. A request to a vendor pipeline server (PaddleOCR-VL) SHALL use the pipeline's own official defaults (greedy decoding) plus output-only options from the config; such requests return no token logprobs. The settings used SHALL appear in `config.infer.yaml`. The accuracy run's client concurrency SHALL be 8 by default and configurable.

#### Scenario: Logprobs captured
- **WHEN** the endpoint returns token logprobs
- **THEN** the prediction stores one logprob per completion token
- **WHEN** the endpoint returns no logprobs
- **THEN** the prediction stores an empty list and inference continues

#### Scenario: Official settings applied
- **WHEN** a TeleOCR block request is sent
- **THEN** it carries TeleOCR's configured presence penalty, frequency penalty and `no_repeat_ngram_size`

### Requirement: Multi-request page pipelines
A page MAY need several requests (TeleOCR: one layout request plus one request per detected block), but it SHALL still produce exactly one prediction row. A page sent to a vendor pipeline server counts as one request (`n_requests=1`, token counts 0), since the pipeline's own model calls are not visible to the harness.

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

#### Scenario: Pipeline page
- **WHEN** a PaddleOCR-VL `statement` page is inferred
- **THEN** one request goes to the pipeline server, the row has `n_requests=1`, empty token logprobs, and blocks parsed from the pipeline's `parsing_res_list`

### Requirement: vLLM serving versions
Models SHALL be served with bf16 on the vLLM image their architecture needs: dots.ocr, typhoon-ocr1.5 and TeleOCR on the pinned `VLLM_IMAGE` (v0.11.0); OvisOCR2 (needs ≥ 0.18) and PaddleOCR-VL-1.6 (needs ≥ 0.11.1) on `VLLM_IMAGE_NEW` (v0.22.1). `infer` SHALL record each selected endpoint's vLLM version (from its `/version` endpoint, or "unknown") in `config.infer.yaml`. Accuracy results MAY be compared across images; latency results SHALL NOT be compared across different vLLM versions without an annotation (latency-benchmark spec).

#### Scenario: Versions recorded
- **WHEN** `infer` runs dots.ocr and OvisOCR2
- **THEN** `config.infer.yaml` records v0.11.0 for dots.ocr and v0.22.1 for OvisOCR2

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
- **WHEN** `probe` completes for the models in `configs/run.yaml`
- **THEN** `runs/<run_id>/probe.json` holds a value in {yes, partial, no} for every (capability, model) pair, plus the evidence page ids
