# Spec Delta

## ADDED Requirements

### Requirement: Per-model endpoint credentials
For each model the harness SHALL read an API key from `$<MODEL>_API_KEY` (the model's `endpoint_env` with `_BASE_URL` replaced by `_API_KEY`), falling back to `$OPENAI_API_KEY`, then to a placeholder for self-hosted servers. The key SHALL be sent as a bearer token on inference requests, on health checks and on version requests, for both the vLLM endpoint and a vendor pipeline endpoint. Keys SHALL never be written to run outputs or `config.infer.yaml`.

#### Scenario: Health check authenticated
- **WHEN** `DOTSOCR_API_KEY` is set and `infer` checks the dots.ocr endpoint's health
- **THEN** the health request carries the key as a bearer token

#### Scenario: Key absent from run outputs
- **WHEN** `infer` writes `config.infer.yaml` for a model with a key set
- **THEN** the file records the base URL's host and `endpoint_kind` but no key

### Requirement: Bounded cold-start wait
The health check SHALL treat an "initialising" response (HTTP 204, or a gateway 502/503 while no worker is ready) as transient and SHALL poll every 5 seconds until the endpoint is healthy or `$OCRBENCH_ENDPOINT_WAIT_S` (default 0) is exhausted, then fail naming the model, URL and last status. During the same budget at the start of a run, an inference request that fails with a cold-start signature SHALL be retried with backoff instead of producing an error row.

#### Scenario: Worker ready within budget
- **WHEN** `OCRBENCH_ENDPOINT_WAIT_S=600` and the endpoint answers 204 for 45 seconds and then 200
- **THEN** `infer` proceeds after about 45 seconds without an error

#### Scenario: Self-hosted default unchanged
- **WHEN** `OCRBENCH_ENDPOINT_WAIT_S` is unset and a self-hosted endpoint is down
- **THEN** `infer` exits non-zero immediately, as before

### Requirement: Serving version recorded without `/version`
When a selected endpoint does not expose `/version`, `infer` SHALL record the vLLM version from the model's latest release record when the endpoint is a Runpod endpoint, and "unknown" otherwise, and SHALL record `endpoint_kind` (`self_hosted` or `runpod`) per model in `config.infer.yaml`.

#### Scenario: Runpod endpoint version from release log
- **WHEN** `infer` runs typhoon against a Runpod endpoint whose `/version` is unreachable and the release log names vLLM 0.11.0
- **THEN** `config.infer.yaml` records `vllm_version: 0.11.0` and `endpoint_kind: runpod` for typhoon

### Requirement: Request size guard
Before sending an inference request the client SHALL compute the request body size and SHALL fail the request locally, with an error naming the size and the limit, when it exceeds the configured maximum for the endpoint (default 25 MB). The page SHALL still get a prediction row with the error set.

#### Scenario: Oversized page
- **WHEN** a page's encoded request body is 31 MB and the limit is 25 MB
- **THEN** no request is sent, and the page's row carries an error naming 31 MB and 25 MB
