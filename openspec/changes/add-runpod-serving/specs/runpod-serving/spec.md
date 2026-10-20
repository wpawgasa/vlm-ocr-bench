# Spec Delta

## Purpose

Serves each benchmark VLM as a Runpod Serverless endpoint built from one worker image and one committed per-model configuration, deployed and rolled back reproducibly, with the operational settings a production caller depends on.

## ADDED Requirements

### Requirement: Per-model serving configuration
Each served model SHALL have one committed configuration file `configs/runpod/<model>.yaml` that references an existing model registry entry (`configs/models/<model>.yaml`) and holds only serving facts: the base image as a digest, the image variant, the weights repository and a pinned revision (a commit hash), the vLLM engine arguments, the endpoint kind, GPU types in priority order, worker counts, idle and execution timeouts, scaler, data centres, CUDA minimum, container disk, client limits (per-worker concurrency, request timeout, cold-start wait, maximum request size) and the SLO targets. The file SHALL NOT contain endpoint ids, keys, served model names, prompts or sampling; those come from the environment or the model registry entry. The system SHALL reject a configuration whose base image is a tag rather than a digest, whose revision is a branch name, or whose model entry does not exist.

#### Scenario: Valid configuration renders
- **WHEN** `ocrbench runpod render dotsocr` runs on a configuration with a digest base image and a commit-hash revision
- **THEN** it prints the `vllm serve` argument list and the template and endpoint settings that a deploy would send, without contacting Runpod

#### Scenario: Unpinned revision refused
- **WHEN** a configuration sets `weights.revision: main`
- **THEN** `render`, `build` and `deploy` exit non-zero naming the model and the field

#### Scenario: Engine arguments match the reference serving
- **WHEN** the configuration test suite runs
- **THEN** every Runpod configuration's engine arguments equal the corresponding compose service's command minus host-specific flags (GPU ids, ports, memory fraction), and any difference fails the test naming the flag

### Requirement: One worker image for every model
The system SHALL build every model's worker image from one Dockerfile parameterised by the model configuration. The image SHALL contain the pinned vLLM base, the model weights at the pinned revision, the serving configurations and an entrypoint that renders the `vllm serve` command from the configuration selected by an environment variable. The TeleOCR variant SHALL add the same pinned plugin and logits processor as the reference serving. At runtime the image SHALL NOT download weights, SHALL run with the Hugging Face hub offline, and SHALL accept only the gpu-memory-utilization and an extra-arguments override from the environment. The image SHALL be identified by its registry digest in every deployment and release record.

#### Scenario: Image built for a model
- **WHEN** `ocrbench runpod build typhoon_ocr15 --push` completes
- **THEN** the pushed image contains the typhoon weights at the configured revision, starts with `OCRBENCH_MODEL=typhoon_ocr15` and serves `GET /v1/models` listing `typhoon`, and the command prints the image digest

#### Scenario: Weights missing at runtime
- **WHEN** a worker starts and the baked weights directory is absent
- **THEN** the entrypoint exits non-zero within 10 seconds naming the path, instead of downloading

#### Scenario: Engine argument override limited
- **WHEN** a template sets `MAX_MODEL_LEN=4096` as an environment variable
- **THEN** the entrypoint ignores it and serves with the configuration's max model length; only `OCRBENCH_GPU_MEMORY_UTILIZATION` and `OCRBENCH_VLLM_EXTRA_ARGS` change the command

### Requirement: PaddleOCR-VL served as one endpoint
The PaddleOCR-VL worker SHALL run its vLLM server and its official PaddleX pipeline in one worker behind one public port. The public port SHALL serve `POST /layout-parsing` (the pipeline), `/v1/*` and `/version` (forwarded to the vLLM server) and `GET /health`. `/health` SHALL return 200 only when the vLLM server and every pipeline process are healthy, and 204 while any of them is still starting. The pipeline's layout stage SHALL run on the worker's GPU.

#### Scenario: One base URL for both clients
- **WHEN** `PADDLEOCR_VL_BASE_URL` is `https://<id>.api.runpod.ai/v1` and `PADDLE_PIPELINE_URL` is `https://<id>.api.runpod.ai`
- **THEN** a `statement` page is answered by `/layout-parsing` and a `kie` question by `/v1/chat/completions`, and both succeed with the unchanged `paddleocr_vl` model configuration

#### Scenario: Partial readiness
- **WHEN** the vLLM server is healthy but one of two pipeline processes is still loading
- **THEN** `GET /health` returns 204 and the load balancer sends no traffic to the worker

### Requirement: Endpoint deployment as code
The system SHALL create or update a model's Runpod template and endpoint from its configuration through the Runpod REST API with `ocrbench runpod deploy <model> --digest <sha256>`. A template SHALL be immutable and named from the model and the configuration hash; a new digest or configuration hash SHALL create a new template and point the endpoint at it; the same digest and hash SHALL be a no-op. The deploy SHALL refuse an empty data-centre list, a tag instead of a digest, a configuration whose rendered command lacks the mandatory logging flags, and an existing endpoint whose kind differs from the configuration. Endpoint and template ids SHALL be written only to a gitignored state file; the management API key SHALL be read from the environment.

#### Scenario: First deploy
- **WHEN** no endpoint named `ocrbench-dotsocr` exists and `deploy dotsocr --digest sha256:…` runs
- **THEN** a template and a load-balancing endpoint are created with the configuration's GPU types, worker counts, idle timeout, scaler, data centres and CUDA minimum, and the ids are stored in `.runpod/state.yaml`

#### Scenario: Repeat deploy is a no-op
- **WHEN** `deploy` runs again with the same digest and an unchanged configuration
- **THEN** no template is created, the endpoint is unchanged, and the command reports "up to date"

#### Scenario: Scaling-only change
- **WHEN** only `workers_min` changes and `deploy` runs with the same digest
- **THEN** the endpoint is patched in place and no new template is created

#### Scenario: Endpoint kind mismatch
- **WHEN** the configuration says `load_balancing` and the existing endpoint is queue-based
- **THEN** `deploy` exits non-zero naming the endpoint and both kinds, and changes nothing

### Requirement: Release records and rollback
Every successful deploy SHALL append a release record to the committed `configs/runpod/releases.yaml` with the model, date, git commit, image digest, configuration hash, weights revision, vLLM version, deployer and the parity verdict (initially `pending`). The record SHALL NOT contain endpoint ids, template ids or keys. `ocrbench runpod rollback <model>` SHALL point the endpoint at the previous release's template and append a release record marked `rollback`.

#### Scenario: Rollback after a bad release
- **WHEN** release N of typhoon fails its smoke check and `rollback typhoon_ocr15` runs
- **THEN** the endpoint serves release N-1's image digest, and the release log gains a record referencing N-1 marked `rollback`

#### Scenario: Secret in a release record refused
- **WHEN** a release record would contain a string matching the Runpod key or endpoint-id patterns
- **THEN** the write is refused and the command exits non-zero

### Requirement: Scaling, placement and GPU class per model
Each endpoint SHALL be deployed with the GPU types, worker counts, idle timeout, scaler and FlashBoot setting from its configuration. The default GPU class SHALL be the 24 GB class with the 48 GB class as fallback. Endpoints that may receive client traffic SHALL support two scaling profiles, `office` (at least one active worker) and `off` (scale to zero), switched by `deploy --profile`. Models not promoted to client traffic SHALL scale to zero.

#### Scenario: Office profile applied
- **WHEN** `deploy dotsocr --profile office` runs
- **THEN** the endpoint's minimum workers is at least 1 and its other settings are unchanged

#### Scenario: Experimental model stays at zero
- **WHEN** `deploy teleocr --profile office` runs for a model that is not promoted
- **THEN** the command refuses the profile and leaves minimum workers at 0

### Requirement: Smoke check per endpoint
`ocrbench runpod smoke <model>` SHALL wait for the endpoint within the configured cold-start budget, verify that `/v1/models` lists the model's served name, send one full-page OCR request on a committed public page and, for PaddleOCR-VL, one `/layout-parsing` request, and SHALL exit non-zero if any step fails or the request exceeds the configured p95 latency target. The check SHALL print the measured cold-start and request latencies. A scheduled run of the check SHALL exist and SHALL alert on two consecutive failures.

#### Scenario: Healthy endpoint
- **WHEN** the endpoint has a ready worker
- **THEN** `smoke` exits 0 and prints the request latency and the served model name

#### Scenario: Cold start within budget
- **WHEN** the endpoint has no worker and one becomes ready after 60 seconds
- **THEN** `smoke` waits, succeeds, and reports a cold start of about 60 seconds

#### Scenario: Wrong model served
- **WHEN** `/v1/models` lists a different served name than the configuration's model entry
- **THEN** `smoke` exits non-zero naming both names before sending any page

### Requirement: Service levels and cost recorded per model
The serving documentation SHALL state, per model and GPU class, the measured cold start, page latency p50 and p95 on the smoke page set, the cost per page derived from the per-second GPU price, and the SLO targets from the configuration. Pages whose output reached the model's `sampling.max_tokens` (loop pages) SHALL be counted and timed separately and SHALL NOT count toward the page latency targets. A model whose measured non-loop p50 or p95, or cold-start p95, exceeds its target SHALL be marked as not meeting its SLO in the documentation and in its release record.

#### Scenario: Measurement recorded
- **WHEN** the latency measurement task completes for dots.ocr on the 24 GB class
- **THEN** `docs/runpod-serving.md` lists its cold start, p50, p95 and cost per page with the date and image digest

#### Scenario: Loop pages reported separately
- **WHEN** 8 of the 60 measured pages reach `sampling.max_tokens`
- **THEN** the documentation reports their count and slowest time, and p50 and p95 are computed over the other 52 pages

#### Scenario: SLO missed
- **WHEN** a model's measured p95 is above `slo.page_p95_s`
- **THEN** the release record carries `slo: missed` and the documentation says so next to the number
