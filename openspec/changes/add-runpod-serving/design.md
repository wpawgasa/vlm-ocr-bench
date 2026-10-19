# Design

## Context

See proposal.md for motivation. The facts that shape this design:

**How the models are served today** (`.devcontainer/compose.yaml`, `scripts/serve_*.sh`)
- Every model is a `vllm/vllm-openai` container with per-model flags. dots.ocr, typhoon-ocr1.5 and TeleOCR run on `v0.11.0` (digest `sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c`); OvisOCR2 (needs vLLM ≥ 0.18) and PaddleOCR-VL-1.6 (≥ 0.11.1) run on `v0.22.1`. All use bf16, `--seed=0`, `--max-logprobs=5`, `--limit-mm-per-prompt={"image": 1}`.
- TeleOCR needs its own image (`.devcontainer/teleocr-vllm.Dockerfile`): the `TeleOCR_vllm` plugin and a `no_repeat_ngram` logits processor, pinned to upstream commit `9921cff`.
- PaddleOCR-VL is two services: the vLLM server (served name `PaddleOCR-VL-1.6-0.9B`, `--gpu-memory-utilization` 0.9 or 0.25) and `paddle_pipeline`, PaddleX's serving app (`POST /layout-parsing`, `/health`) built from `.devcontainer/paddle-pipeline.Dockerfile`, with PP-DocLayoutV3 on CPU (`PIPELINE_DEVICE`) or GPU (`PADDLE_PKG=paddlepaddle-gpu`). One pipeline process handles one page at a time, so compose runs 4 replicas.
- Model sizes (model cards, retrieved 2026-10-14): dots.ocr 3B total (1.7B LLM, `trust_remote_code`), typhoon-ocr1.5 2B, TeleOCR ≈1.2B, OvisOCR2 0.85B, PaddleOCR-VL 0.9B. All fit a 24 GB GPU in bf16 at the compose `--max-model-len` (16384; OvisOCR2 32768).

**How clients call them** (`ocr_bench/models/`)
- `configs/models/<name>.yaml` names an `endpoint_env` (`DOTSOCR_BASE_URL` etc.) and a `served_model_name`; PaddleOCR-VL also names `pipeline.endpoint_env` (`PADDLE_PIPELINE_URL`).
- `VllmClient` (`vllm_client.py`) uses the OpenAI SDK with `api_key=$OPENAI_API_KEY or "EMPTY"`, no retries in the SDK, 120 s per request, one retry on timeout/transport/5xx, never on 4xx. `health()` derives `<base without /v1>/health` and sends **no** auth header; `version()` calls `/version`. `PipelineClient` behaves the same for `/layout-parsing` and `/health`.
- `cli._connect` builds every model, health-checks once and exits 2 on failure. Images are sent as base64 PNG data URLs inside the chat message.

**Runpod Serverless** (all retrieved 2026-10-14 unless noted; "unverified" marks what the docs did not state)
- Two endpoint kinds. *Queue-based*: `/run`, `/runsync`, `/status`, `/health`, results kept 30 min (`/run`) or 1 min (`/runsync`), job TTL default 24 h, execution timeout default 600 s; the console's Requests tab shows job input and output. Runpod's `worker-vllm` (`runpod/worker-v1-vllm`) wraps `vllm serve` for this mode and exposes `https://api.runpod.ai/v2/<id>/openai/v1`. *Load-balancing*: requests go straight to a worker's HTTP server at `https://<id>.api.runpod.ai/<path>`, no queue, no stored results; the worker exposes `PORT` and a health path (`/ping` by default, `HEALTH_CHECK_PATH` to change; 200 = ready, 204 = initialising); requests before any worker is ready fail and the docs ask clients to retry (≥3 retries, 5–10 s apart); 2 min request timeout while no worker is available, 5.5 min processing timeout, 30 MB payload limit, no retry and dropped requests when overloaded. The `worker-vllm` README documents running the stock `vllm/vllm-openai:<version>` image this way ("Option 3") with `PORT=8000`; vLLM serves `/ping` and `/health` itself.
  Sources: https://docs.runpod.io/serverless/load-balancing/overview , https://docs.runpod.io/serverless/endpoints/send-requests , https://github.com/runpod-workers/worker-vllm (README and `docs/configuration.md`).
- `worker-vllm` configuration: any `vllm serve` flag as an upper-cased env var (`MAX_MODEL_LEN`, `TRUST_REMOTE_CODE`, `DTYPE`, `ENABLE_PREFIX_CACHING`…), aliases `MODEL_NAME`, `MODEL_REVISION`, `OPENAI_SERVED_MODEL_NAME_OVERRIDE`, `CUSTOM_CHAT_TEMPLATE`, escape hatch `VLLM_EXTRA_ARGS`, `VLLM_CONFIG_FILE`; `ENABLE_LOG_REQUESTS=false` (the old `DISABLE_LOG_REQUESTS` was removed); `MAX_CONCURRENCY` default 30; weights baked with build args `MODEL_NAME`, `MODEL_REVISION`, `BASE_PATH` (default `/runpod-volume`, i.e. a network volume when attached). Its releases track recent vLLM only (v2.29.0 → vLLM 0.31.0, v2.24.0 → 0.26.0; nothing on the releases page ships 0.11.x or 0.22.x). Source: https://github.com/runpod-workers/worker-vllm/releases .
- Endpoint settings: active (min) workers default 0, max workers default 3, GPUs per worker 1, scaler `QUEUE_DELAY` (default 4 s) or `REQUEST_COUNT`, idle timeout default 5 s (1–3600), execution timeout default 600 s, FlashBoot on by default, a prioritised list of GPU types, `dataCenterIds` (26 ids in the API enum; default all), `allowedCudaVersions`, optional network volume (adds latency and pins the data centre). 24 GB class (L4, A5000, RTX 3090) $0.00019/s; 48 GB (A40, A6000) $0.00034/s; L40S $0.00053/s; A100 80 GB $0.00076/s; H100 PRO $0.00116/s. Billing runs from worker start to stop, rounded up to the second; image pull and cached-model download are not billed; idle time is. Container disk $0.10/GB/month. Default spend limit $80/h. Sources: https://docs.runpod.io/serverless/endpoints/endpoint-configurations , https://docs.runpod.io/serverless/pricing .
- Management API: REST v1 (`https://rest.runpod.io/v1`, `POST /endpoints` with `templateId`, `gpuTypeIds`, `workersMin/Max`, `idleTimeout`, `executionTimeoutMs`, `scalerType`, `scalerValue`, `flashboot`, `dataCenterIds`, `allowedCudaVersions`; `POST /templates` with `imageName`, `env`, `containerDiskInGb`, `dockerStartCmd`, `ports`, `isServerless`, `containerRegistryAuthId`) is deprecated and retires on 2026-11-15. REST v2 is `https://api.runpod.io/v2` (`/v2/serverless`, `/v2/templates`, `/v2/account/secrets`; `PATCH /v2/serverless/{id}`; `workers` and `scaling` are nested objects; template fields `image`, `disk`, `registry`). Bearer auth with a Runpod API key. The v2 reference pages for the serverless create body returned 404 at retrieval time, and **neither v1 nor `runpodctl serverless create` shows a field that selects queue vs load-balancing — unverified whether the API can create a load-balancing endpoint.** *Verified 2026-10-19 against the live API:* `POST /v2/serverless` takes `type: QUEUE|LOAD_BALANCER` (not patchable), `gpu: {pools, excludedTypes, count, minCudaVersion}` with serverless pool ids from `GET /v2/catalog/gpus` (the 24 GB L4/A5000/3090 are pool `AMPERE_24`, which also holds an RTX PRO 6000 Blackwell MIG slice, so `deploy` excludes every pool member not in `gpu_types`), `workers: {min, max, idleTimeout}`, `scaling: {type: REQUEST_COUNT, requestCount}` (required for load balancers), `flashboot: OFF|FLASHBOOT|PRIORITY_FLASHBOOT`, `mounts.network: [{volumeId}]`; `templateId` is applied once and no link is kept, so an endpoint's release is recognised by its `image`. `runpodctl serverless create|update|delete|get|list` exists but takes no config file. Sources: https://docs.runpod.io/api-reference/overview , https://docs.runpod.io/api-reference-v2/migrate-from-v1 , https://docs.runpod.io/api-reference/endpoints/POST/endpoints , https://docs.runpod.io/api-reference/templates/POST/templates , https://docs.runpod.io/runpodctl/reference/runpodctl-serverless .
- Secrets: account-level, referenced from template env values as `{{ RUNPOD_SECRET_<name> }}`, substituted at worker boot, value never readable back. Source: https://docs.runpod.io/pods/templates/secrets and the v2 `account/secrets` reference.
- API keys: a key can be *Restricted* with per-endpoint none / read-only / read-write access, separate from management (GraphQL/REST) access. Source: https://docs.runpod.io/get-started/api-keys .
- Compliance: Runpod lists SOC 2 Type II, SOC 3, HIPAA and GDPR documents in its Trust Center and a published standard DPA; its docs say GDPR compliance applies to data processed in European regions, that Secure Cloud runs in T3/T4 partner data centres, that workers are containers isolated per tenant, and that its terms forbid hosts from inspecting worker data. Third-party sources say the SOC 2 scope is Secure Cloud, not Community Cloud. **Unverified:** how long worker stdout logs are retained, whether Runpod staff can read worker logs, and which Asia-Pacific regions (AP-JP-1 Japan, AP-IN-1 India exist for Pods) serve load-balancing endpoints with 24 GB GPUs. Sources: https://docs.runpod.io/references/security-and-compliance , https://www.runpod.io/legal/compliance , https://www.runpod.io/blog/runpod-soc2-certification .
- vLLM logging flags (v0.11.0 CLI reference, retrieved 2026-10-14): request logging is **off by default** (`--enable-log-requests` default False; `--disable-log-requests` deprecated), `--enable-log-outputs` default False, `--max-log-len` default unlimited, `--api-key` optional, `--uvicorn-log-level` default info. Source: https://docs.vllm.ai/en/v0.11.0/cli/serve.html .

**Model findings that matter for production** (`docs/continuing-on-another-host.md`): TeleOCR does not read Thai in bf16 and was screened out of the benchmark; OvisOCR2 failed the Thai gate (Chinese in Thai text, loops); dots.ocr hits `max_tokens` 8192 on 9.9% of rows at T=0 (Thai repetition loops). The user nevertheless asked for all five benchmark VLMs to be deployable; see Open Questions.

## Goals / Non-Goals

**Goals:**
- A Runpod endpoint per model that a client reaches exactly like a self-hosted vLLM server: `<MODEL>_BASE_URL` + a key, OpenAI chat completions, the same `served_model_name`, prompts and sampling.
- Every deployed endpoint traceable to a git commit, an image digest, a config hash and a model revision, and revertible to the previous release.
- The same safeguards for client pages whether the caller is the harness or a production service, with the parts that code can enforce enforced in code.
- Cold start, latency and cost per page measured, not guessed, before client traffic.

**Non-Goals:**
- Changing prompts, sampling or parsers for production; those stay in `configs/models/*.yaml` (model-inference spec) and are shared by both serving paths.
- Serving on Runpod Pods (always-on VMs); if serverless scale-to-zero turns out not to fit, that is a new change.
- A request queue or batch API of our own; bursts above `workersMax` are rejected by Runpod and retried by the client.
- Benchmark numbers from Runpod GPUs. The latency benchmark stays on the reference hosts.

## Decisions

### D1. Load-balancing endpoints that expose vLLM's own API
Each model is a Runpod **load-balancing** endpoint whose worker is our vLLM-based image listening on `PORT=8000`, health path `/health` (vLLM's own; `HEALTH_CHECK_PATH=/health`). The client base URL is `https://<endpoint-id>.api.runpod.ai/v1`.

Why, over the queue-based `worker-vllm` with its `/openai/v1` route:
- **No stored payloads.** The queue keeps job inputs and results (30 min for `/run`, 1 min for `/runsync`, TTL up to 24 h) and shows them in the console. A load-balancing endpoint proxies the HTTP request and stores nothing, which is the single biggest client-data property we can buy from the platform (D9).
- **The image stays ours.** `worker-vllm` releases are coupled to recent vLLM versions (0.26–0.31), while dots.ocr, typhoon and TeleOCR were validated on 0.11.0 and OvisOCR2/Paddle on 0.22.1. The load-balancing path runs the pinned `vllm/vllm-openai` digests we already benchmarked, with our existing flags. This is what makes "the same image except configuration" true.
- **Unchanged client contract.** A plain `/v1` base URL means `endpoint_env` works as is, `/health` and `/version` exist, and PaddleOCR-VL's pipeline path can be served from the same worker (D3).

Costs accepted: no queue, so the client must retry through cold starts (D5); per-request processing limit 5.5 min (our client timeout is 120 s, fine); 30 MB payload limit (a 4 MP PNG data URL is 3–8 MB; the client rejects larger requests up front); requests above capacity are dropped rather than buffered, so `workersMax` and the client's concurrency cap must agree (D7).

*Alternative:* queue-based `worker-vllm` from source with `VLLM_VERSION` set to our bases. Rejected for the retention and because its handler's compatibility with vLLM 0.11.0 is unverified. It remains the fallback if load-balancing endpoints cannot be created in the chosen data centres or GPU class (Risks); the config schema (D2) carries `endpoint.kind: load_balancing | queue` so the fallback is a config switch, and the queue variant then sets the job TTL to the minimum and documents the retention in the audit (D9).

### D2. One Dockerfile, one config file per model
**Config.** `configs/runpod/<model>.yaml` (committed, no ids or keys) is the single source of per-model serving facts:

```yaml
model: dotsocr                      # the configs/models/<name>.yaml it serves (endpoint_env, served_model_name come from there)
image:
  base: vllm/vllm-openai@sha256:014a95f2…   # pinned digest, the one the benchmark ran on
  variant: plain | teleocr | paddle          # which stage of runpod/Dockerfile to build
weights:
  repo: dots-studio/dots.ocr
  revision: <40-hex commit>                  # pinned; `main` is refused
vllm_args:                                   # exactly the compose flags, minus host-specific ones
  trust-remote-code: true
  dtype: bfloat16
  max-model-len: 16384
  limit-mm-per-prompt: '{"image": 1}'
  gpu-memory-utilization: 0.9
  seed: 0
  max-logprobs: 5
  chat-template: null                        # path inside the image when a model needs one
endpoint:
  kind: load_balancing
  gpu_types: ["NVIDIA L4", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"]   # priority order, 24 GB class
  workers_min: 0
  workers_max: 3
  idle_timeout_s: 120
  scaler: {type: REQUEST_COUNT, value: 4}
  flashboot: true
  data_centers: []                           # empty = refuse to deploy; filled from the allow-list (D9)
  cuda_min: "12.8"
  container_disk_gb: 30
client:
  max_concurrency: 4                         # per worker; the harness caps at workers_max * this
  request_timeout_s: 120
  cold_start_wait_s: 600
  max_request_mb: 25
slo:
  page_p50_s: 6
  page_p95_s: 20
  cold_start_p95_s: 90
```

A pydantic schema in `ocr_bench/runpod/config.py` validates it: the `model` must exist in `configs/models/`, the digest must be a digest, the revision a commit hash, `data_centers` non-empty at deploy time, and the rendered `vllm serve` argument list must contain the logging flags from D9. The config hash (SHA-256 of the canonical YAML) is part of every release record.

**Image.** One multi-stage `runpod/Dockerfile` with `ARG VLLM_IMAGE` and `ARG MODEL` builds every model:
- stage `plain`: `FROM ${VLLM_IMAGE}`; copies `runpod/entrypoint.py` and `configs/runpod/`; downloads the weights with `huggingface_hub.snapshot_download(repo, revision)` into `/models/<repo>` (BuildKit secret `HF_TOKEN` for gated repos, never stored in a layer); sets `HF_HUB_OFFLINE=1`, `PORT=8000`, `OCRBENCH_MODEL=<model>`.
- stage `teleocr`: `plain` plus the plugin and logits processor exactly as `.devcontainer/teleocr-vllm.Dockerfile` installs them (same pinned commit).
- stage `paddle`: `plain` plus PaddleX (D3).

`entrypoint.py` reads `OCRBENCH_MODEL`, loads the baked config, renders `vllm serve /models/<repo> --served-model-name … <vllm_args> <D9 logging flags>` and `exec`s it. Env vars `OCRBENCH_VLLM_EXTRA_ARGS` and `OCRBENCH_GPU_MEMORY_UTILIZATION` are the only runtime overrides, so a template cannot silently change the engine configuration; everything else needs a new image build, which is the point.

Tags and pinning: `code.loolootech.com/loolootech/runpod-<model>:<git-sha>` built by a GitHub Actions workflow on changes under `runpod/` or `configs/runpod/`; the pushed **digest** is what templates reference, never a tag. The registry package is private (the `code.loolootech.com` registry, open question 3); the workflow pushes with the `LOOLOO_REGISTRY_USER`/`LOOLOO_REGISTRY_TOKEN` repository secrets, and Runpod pulls with a registry credential whose id is `registry_credential_id` in `.runpod/state.yaml`, sent as the template's `registry` field (a credential id string in REST v2). Deploy and `--dry-run` refuse when it is missing. Weights licences are the models' own; keeping the packages private avoids redistributing them and keeps the image surface closed.

**Weights baked in, not downloaded or mounted.** Weights are 2–7 GB per model, so the image grows to roughly 15–20 GB on top of the vLLM base. Baking gives a single digest that pins vLLM + plugin + weights + config, no Hugging Face dependency at cold start, no token in the worker, and FlashBoot caches image layers on the host.
*Alternatives:* HF download at cold start (adds 20–60 s and an outbound dependency per cold start, needs `HF_TOKEN` in the worker; allowed only for the `ocrbench runpod smoke --dev` path, never for an endpoint on the client-data list); a network volume (`/runpod-volume`) shared by all models (pins the endpoint to one data centre, adds disk latency, $0.07/GB/month, and a second thing to version); Runpod's cached-model scheduling (`--model-reference`) is designed for HF repos the platform caches and is unverified for these five repos.

**Compile cache.** vLLM compiles CUDA graphs at start; the cache lives on the container disk and dies with the worker. Cold-start time is measured in task 6.2; if it misses the SLO, `--enforce-eager` is the first lever (slower steady state), a warm cache copied into the image at build time the second (needs a GPU build host; deferred).

### D3. PaddleOCR-VL: one worker, two processes, one port
The `paddle` image runs, under a small supervisor (`runpod/paddle_supervisor.py`), (1) `vllm serve` for `PaddlePaddle/PaddleOCR-VL-1.6` on loopback `:8000` at `--gpu-memory-utilization 0.7`, and (2) the PaddleX serving app from `.devcontainer/paddle-pipeline-entrypoint.py` with `VL_SERVER_URL=http://127.0.0.1:8000/v1` and `PIPELINE_DEVICE=gpu:0` (PP-DocLayoutV3 on the same GPU; serverless workers have few vCPUs). The public port `8080` is a thin FastAPI front (`runpod/paddle_front.py`) that
- forwards `POST /layout-parsing` round-robin to `PADDLE_PIPELINES` (default 2) pipeline processes on `:8081…`;
- reverse-proxies `/v1/*`, `/version` to the loopback vLLM, so `kie`/`classify` question requests (which go to the vLLM endpoint in the model config) reach the same endpoint;
- answers `/health` with 200 only when vLLM `/health` and every pipeline `/health` are 200, else 204 (initialising), so Runpod's load balancer sees one readiness signal.

So `PADDLEOCR_VL_BASE_URL=https://<id>.api.runpod.ai/v1` and `PADDLE_PIPELINE_URL=https://<id>.api.runpod.ai` name the same endpoint, and the model config, parsers and harness code are untouched.

*Alternatives:* two endpoints (pipeline worker calling a vLLM endpoint over the public gateway) doubles cold starts, puts a second key inside a worker and sends every crop across the gateway — rejected. Layout client-side (PP-DocLayoutV3 in the client, crops to a plain vLLM endpoint) would move 130 MB of models and paddlepaddle into every client and change the parity-tested pipeline — rejected. Replacing the pipeline with a single prompt would change the model's behaviour — out of scope.

### D4. Deployment as code: `ocrbench runpod`
A typer command group in `ocr_bench/runpod/cli.py`, registered on the existing `ocrbench` app:

| Command | Does |
|---|---|
| `render <model>` | prints the `vllm serve` argv and the template/endpoint JSON the deploy would send (no network) |
| `build <model> [--push]` | `docker build` the stage for the model, with the pinned base and revision; `--push` pushes and prints the digest |
| `deploy <model> --digest <sha256>` | upserts the template `ocrbench-<model>-<config-hash[:8]>` and the endpoint `ocrbench-<model>` through REST v2 (`/v2/templates`, `/v2/serverless`), scaling and placement from the config; refuses an empty `data_centers`, a `main` revision, a tag instead of a digest, or a config whose rendered argv lacks the D9 flags; appends a release record |
| `rollback <model>` | points the endpoint at the previous release's template |
| `status <model>` | endpoint id (masked), release, worker counts, health |
| `smoke <model>` | the smoke check (D5) |
| `parity <model>` | the parity gate (D6) |

**State and secrets.** `RUNPOD_API_KEY` (a management key) comes from the environment; locally `ocrbench` loads it from the gitignored `./.env` at startup (see `.env.example`), without overriding a variable already set, so a CI repository secret wins. Endpoint and template ids are written to `.runpod/state.yaml` (gitignored) and to the release record, never to git. The committed release log `configs/runpod/releases.yaml` holds per model: date, git sha, image digest, config hash, model revision, vLLM version, who deployed, parity result — no ids. Worker-side secrets are not needed (weights are baked, no HF token at runtime); if one ever is, it is a Runpod Secret referenced as `{{ RUNPOD_SECRET_<name> }}` in the template, never a literal.

**Idempotency.** Templates are immutable releases: a new config hash or digest creates a new template; the endpoint is `PATCH`ed to it. Deploying the same digest and hash twice is a no-op. Scaling-only changes `PATCH` the endpoint without a new template.

*Alternatives:* Terraform/Pulumi (a community Runpod provider exists but is unverified and adds a toolchain for five endpoints); the `runpod` Python SDK (uses the legacy GraphQL API, and `runpodctl serverless create` neither reads a config file nor exposes the endpoint kind); REST v1 (retires 2026-11-15). If v2 cannot create a load-balancing endpoint (Risks), the Human creates the endpoint once in the console and `deploy` manages templates and settings from then on; `deploy` detects the kind from the API response and refuses to touch an endpoint whose kind disagrees with the config.

### D5. Client access, retries and cold starts
Environment contract for any client, harness or production service (all per model):
- `<MODEL>_BASE_URL` (the existing `endpoint_env`), e.g. `https://<id>.api.runpod.ai/v1`;
- `<MODEL>_API_KEY`: the restricted Runpod key for that endpoint; falls back to `OPENAI_API_KEY`, then `EMPTY` (self-hosted);
- `OCRBENCH_ENDPOINT_WAIT_S` (default 0): how long `health()` may wait for a worker to become ready.

`VllmClient` and `PipelineClient` changes (`ocr_bench/models/vllm_client.py`, `pipeline_client.py`, `registry.build_model`):
- the key is sent as `Authorization: Bearer` on chat, `/health` and `/version` (today health has no header, which a Runpod gateway rejects);
- `health()` treats 204 and gateway 502/503 as *initialising* and polls every 5 s until 200 or the wait budget is spent, then raises `EndpointUnavailable` with the last status;
- a base URL whose host ends in `api.runpod.ai` must be `https://`; `http://` raises before any request;
- the in-flight retry policy stays "one retry on timeout/transport/5xx, none on 4xx", except that a cold-start signature (502/503 or "no workers available") within the first `cold_start_wait_s` of a run is retried with 5 s backoff up to the budget — a cold start must not produce error rows;
- `version()` falls back to the release record's vLLM version when `/version` is unreachable, and `config.infer.yaml` records `endpoint_kind: runpod|self_hosted`.
- A request whose JSON body would exceed `client.max_request_mb` is failed locally with a clear error instead of a gateway 413.

Concurrency: the harness's `--concurrency` for a Runpod endpoint is capped at `workers_max × client.max_concurrency`; above that a load-balancing endpoint drops requests.

**Smoke check** (`ocrbench runpod smoke <model>`): health with the wait budget; `/v1/models` lists the `served_model_name`; one `ocr_fullpage` request on a committed ThaiOCRBench page (`tests/fixtures/runpod/smoke_page.png`, public data) must return non-empty text with logprobs within `slo.page_p95_s`; for Paddle, one `/layout-parsing` call too. Exit code is the alert. Also run on a schedule (D7).

Production clients outside this repo follow the same contract with any OpenAI SDK; `docs/runpod-serving.md` documents it with a 20-line example and the retry rules.

### D6. Parity gate
`ocrbench runpod parity <model>` proves a Runpod endpoint behaves like the reference serving before it may receive client pages:
- **Sample set:** `configs/runpod/parity_samples.v1.yaml`, a committed list of ThaiOCRBench sample ids from run `2026-09-22-a`'s manifest — 60 pages, `clean` condition: 20 `ocr_fullpage`, 10 `ocr_line`, 10 `table`, 10 `handwriting`, 10 `kie`. No statement pages, ever (the loader refuses a `bankstmt` source id).
- **Reference:** predictions from the self-hosted serving of the same model on the same vLLM version, prompt versions and sampling, produced once by `ocrbench runpod parity --record-reference` on the H100 and DVC-tracked under `data/runs/parity-<model>/reference.jsonl`.
- **Comparison:** the Runpod run must use the same `prompt_version` per row and report the same vLLM version (else the gate fails before scoring). Per row, the pairwise CER between the reference text and the Runpod text (canonical text, `ocr_bench.normalize.text`). Pass when: error rows on either side ≤ 2% and never on one side only for more than 1 row; median pairwise CER ≤ 0.01 and p95 ≤ 0.05; and the ground-truth score (the task's primary metric from the accuracy-scoring spec) differs by at most 0.01 in mean. These tolerances exist because a different GPU (L4 vs H100) changes bf16 kernel order, so T=0 is deterministic per GPU, not across GPUs; exact match is reported but not required.
- **Output:** `data/runs/parity-<model>/<date>-<digest[:12]>.json` with per-row CERs and the verdict, and the verdict copied into the release record. An endpoint whose latest release has no passing parity is refused by `deploy --promote` (the flag that marks it eligible for client data, D9).

### D7. Production settings, SLOs, cost, monitoring, rollback
- **GPU class:** 24 GB (L4 / A5000 / 3090, bf16, $0.00019/s ≈ $0.68/h) for all five; 48 GB (A40 / A6000) as the second priority for availability. OvisOCR2's `--gdn-prefill-backend=triton` on sm_89 and vLLM 0.11.0's CUDA requirement against Runpod's driver pool (`cuda_min`) are verified in task 6.1.
- **Scaling:** endpoints that may receive client traffic run `workers_min: 1` in business hours (no cold start on the first page; ≈$16/day on a 24 GB GPU) and 0 otherwise, toggled by `ocrbench runpod deploy --profile {office,off}`; `workers_max: 3`; `REQUEST_COUNT` scaler with value = `client.max_concurrency`; idle timeout 120 s (a statement is 5–30 pages a few seconds apart; 5 s would thrash). FlashBoot on. Experimental models (TeleOCR, OvisOCR2) stay at `workers_min: 0`.
- **SLOs (initial targets, confirmed or revised by task 6.2's measurement):** page latency p50 ≤ 6 s, p95 ≤ 20 s for the single-request models and Paddle; TeleOCR (one layout + N block requests) p95 ≤ 40 s; cold start (first 204 → first 200) p95 ≤ 90 s; smoke success ≥ 99% over a month. dots.ocr's loop rows (8192 tokens) are the p99 tail and are bounded by the 120 s client timeout, not by an SLO.
- **Cost estimate:** a page at ~5 s on a 24 GB worker ≈ $0.001; a 20-page statement ≈ $0.02 plus idle (120 s × $0.00019 = $0.023 per burst); one warm worker 8 h/day ≈ $5.5/day. Container disk 20 GB × 5 models ≈ $10/month. These are recomputed from measured latency in the serving doc; a Runpod spend limit (Human) caps the blast radius.
- **Monitoring:** (1) the scheduled smoke workflow (GitHub Actions cron, every 30 min in business hours, keys as repository secrets) failing twice in a row alerts the maintainers (workflow failure notification; a webhook is an open question); (2) `ocrbench runpod status` reads Runpod's endpoint health/worker counts; (3) the client records per-request latency, status and token counts as numbers only (no content) in its own logs, which is what the harness already does in `predictions.jsonl`.
- **Versioning and rollback:** an endpoint release = template (immutable) = image digest + config hash + model revision + vLLM version, logged in `configs/runpod/releases.yaml`. `rollback` re-points the endpoint at the previous template; workers restart with the old image. Model-config (`configs/models/*.yaml`) changes are not releases: they change prompts on the client side and are versioned by `prompt_version`.

### D8. What stays identical to self-hosting
The `configs/models/*.yaml` entries, prompt versions, parsers, sampling, image resize and the `served_model_name` are shared. `configs/runpod/<model>.yaml` references the model entry and must not duplicate anything in it; `render` fails if a `served_model_name` or `endpoint_env` appears in the Runpod config. The compose files remain the reference serving and the source of the `vllm_args` values; a drift check in `tests/test_runpod_config.py` compares each Runpod config's `vllm_args` with the compose service's command (minus host-specific flags) so the two cannot diverge unnoticed.

### D9. Client-data safeguards (enforced where possible)
Written as requirements in `specs/client-data-safeguards/spec.md`; the design behind them:

| Safeguard | Enforced by |
|---|---|
| No request/response content in worker logs | `entrypoint.py` always appends `--no-enable-log-requests` (0.22) / `--disable-log-requests` (0.11), `--max-log-len 0`, `--uvicorn-log-level warning`; `deploy` refuses an argv without them; the Paddle front logs only status, size and timing; a `live` test greps a worker's logs after a request for the request's sentinel text |
| Images only in memory, nothing persisted | no network volume on client-data endpoints, `HF_HUB_OFFLINE=1`, `--mm-processor-cache-gb 0`, `--download-dir` unset, writable paths only `/tmp` for vLLM's compile cache; the Paddle front never writes `file` payloads to disk (the pipeline is invoked in memory; `visualize: false`) |
| Platform stores no payloads | load-balancing endpoints (D1); if `kind: queue` is ever used, the audit must state the retention and the endpoint is not promotable |
| TLS | the client refuses `http://` for `*.api.runpod.ai`; the gateway is HTTPS only |
| Keys | restricted per-endpoint Runpod keys (read-write on that endpoint only); management key separate; keys only in env, `.runpod/` (gitignored), GitHub Actions secrets; `gitleaks` pre-commit and CI rule for `rpa_[A-Za-z0-9]{20,}` and `[a-z0-9]{14}\.api\.runpod\.ai` |
| Region / tenancy | `data_centers` must be a non-empty subset of the allow-list in `configs/runpod/data_centers.yaml` (approved by the Human review; Secure Cloud only, compliance-filtered GPUs) |
| Terms reviewed by a human | `docs/runpod-data-processing-review.md`: date, reviewer, documents (DPA, ToS, Trust Center reports, region statement), decision, conditions, next review date; `deploy --promote` refuses while the file's decision is not `approved` or is older than 12 months |
| Audit of endpoints allowed client data | `.runpod/state.yaml` holds `client_data: true|false`, `approved_by`, `approved_on` per endpoint (ids live here, gitignored); `configs/runpod/releases.yaml` carries the promotion (name, date, parity verdict) without ids; `ocrbench infer`/the production client refuse to send a `bankstmt` page to an endpoint not promoted, unless `OCRBENCH_ALLOW_UNPROMOTED=1` is set *and* the source is not client data |
| Retention on our side | unchanged: predictions and statements stay under gitignored `data/`; the smoke and parity sets are public pages only |

## Risks / Trade-offs

- [Load-balancing endpoints cannot be created through the API, or are not offered for the chosen GPU class / data centres] → the Human creates the endpoint in the console once; `deploy` manages templates and settings; if the kind is not available at all, switch `endpoint.kind` to `queue`, build on `worker-vllm` from source with our base image, set job TTL to the minimum and record the retention in the audit before any promotion.
- [vLLM 0.11.0 image incompatible with the CUDA driver on Runpod's 24 GB hosts] → `cuda_min` on the endpoint; if no host satisfies it, re-validate dots.ocr/typhoon on 0.22.1 through the parity gate (that is what the gate is for) and bump the base digest in the config.
- [Cold start above 90 s because of CUDA-graph compilation on a fresh container] → measure; `--enforce-eager` or a baked compile cache; `workers_min: 1` in business hours already removes the cold start from the first page.
- [Bursts above `workers_max × max_concurrency` are dropped, not queued] → the client caps concurrency and retries with backoff; raise `workers_max` per model from measured traffic.
- [A 4 MP page as a PNG data URL approaches the 30 MB limit] → the client checks the body size and downscales under the model's own `max_pixels` first; JPEG is not used because it would change parity.
- [Different GPU → different bf16 numerics → parity failures that are not bugs] → tolerances in D6 are set for that; a failure above them is investigated, not waved through.
- [Runpod log retention and staff access are unverified] → the worker never emits content, so whatever is retained holds none; the Human review records what Runpod states in writing.
- [Models screened out of the benchmark (TeleOCR, OvisOCR2) are deployed anyway] → they are never promoted to client data without a parity pass *and* a Thai gate pass recorded in the release; defaults keep them at `workers_min: 0`.
- [Image size 15–20 GB slows the first pull on a new host] → image pull is not billed and FlashBoot caches it; the smoke schedule keeps the layers warm.
- [Registry credential in Runpod] → stored as a Runpod registry credential (Human), referenced by id in the template, never in git.

## Migration Plan

1. Land the code (config schema, image, entrypoint, client changes, CLI) with unit tests; nothing is deployed yet and self-hosted serving is unchanged.
2. Human: Runpod account, spend limit, restricted keys, registry credential, data-centre allow-list, DPA review document.
3. Build and push the five images; deploy each endpoint with `workers_min: 0`; smoke each.
4. Record the parity references on the H100 (one session, all five models); run the parity gate against each endpoint; fix or re-pin until it passes.
5. Promote only the models the user picks for client traffic; set the office profile; enable the scheduled smoke.
6. Rollback at any step: `ocrbench runpod rollback <model>` (previous template), or delete the endpoint; the self-hosted path is untouched throughout.

## Open Questions

1. Which models should actually be promoted to client traffic first? dots.ocr, typhoon-ocr1.5 and PaddleOCR-VL read Thai; TeleOCR and OvisOCR2 failed the Thai gate. This design deploys all five but promotes none by itself.
2. Data-centre allow-list: EU regions carry Runpod's GDPR statement; AP-JP-1 would be closest to Thailand but its serverless availability and compliance scope are unverified. A legal/latency decision for the Human review. *Experimental scope decided 2026-10-19:* the data owner approved sending this repo's statements to Runpod outside Thailand for experiments; the allow-list is the ten Europe/North-America data centres with a configured 24 GB GPU (`docs/runpod-data-processing-review.md`). The production decision stays with #139.
3. Registry: private GHCR packages with a Runpod registry credential (assumed), or Runpod's own registry support / Docker Hub? *Resolved 2026-10-16:* the private `code.loolootech.com` registry, images under `loolootech/runpod-<model>`, pulled with the existing Runpod credential named `code.loolootech.com`.
4. Who is the production client — the docparse FastAPI service, or a separate client application — and does it need anything beyond the base URL + key contract?
5. Alerting channel for a failing smoke (workflow e-mail is the default; Slack/Line webhook needs a secret).
6. Is a dropped-request behaviour during bursts (no queue) acceptable for the production caller, or should it front the endpoint with its own queue?
7. Monthly budget and the Runpod spend limit value. *Partly resolved 2026-10-16:* Runpod offers no user-set cap; the account runs under Runpod's own default limit ($80/h at retrieval time). Auto-pay is on: when the balance drops below $100 it is refilled to $150, so endpoints never stop for lack of credit, and nothing caps total spend except that hourly limit. The real bound is per-endpoint `workers_max: 3` (≈$2/h at 24 GB pricing). Still open: the monthly budget figure, and whether a billing-page review or alert should accompany each `--promote`.
