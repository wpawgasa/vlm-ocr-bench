# Proposal

## Why

The benchmark VLMs run today only on our own GPU hosts, started by hand through `.devcontainer/compose.yaml` and `scripts/serve_*.sh`. Production use needs endpoints that scale to zero, come back on demand, are reproducible from the repository, and can be handed to a client application without that application knowing about compose, GPU ids or vLLM flags. Runpod Serverless gives pay-per-second GPU workers with an OpenAI-compatible HTTP path, which is exactly how the harness already talks to every model.

The pages these endpoints will read are client bank statements, so serving them off-premises is only acceptable with safeguards that are written down, enforced in code where code can enforce them, and reviewed by a person where it cannot.

## What Changes

- **One serving image, configured per model.** A single `runpod/Dockerfile` builds a worker image from the same pinned vLLM base images the benchmark used, with the model weights baked in at a pinned revision. Everything model-specific (base image, revision, vLLM engine flags, chat template, `trust_remote_code`, max model length, GPU class, scaling) lives in a committed `configs/runpod/<model>.yaml`. The five benchmark VLMs are in scope: dots.ocr, typhoon-ocr1.5, TeleOCR, OvisOCR2 and PaddleOCR-VL-1.6. The docparse role models (Qwen3-VL, Qwen3) are not.
- **PaddleOCR-VL as one worker.** Its official two-stage pipeline (PaddleX layout + vLLM recognition) runs inside one worker image behind one public port; the harness and production clients see one endpoint.
- **Load-balancing endpoints, not the job queue.** Endpoints are Runpod load-balancing endpoints that expose vLLM's own OpenAI API directly, so Runpod stores no request or result payloads and the client base URL is a plain `.../v1`.
- **Deployment as code.** A new `ocrbench runpod` command group renders the model config into a Runpod template and endpoint, creates or updates them idempotently through Runpod's REST API, records the release (image digest, config hash, model revision) and can roll an endpoint back. Secrets and endpoint ids never enter git.
- **Client access.** The harness's model client accepts a per-model API key and a cold-start wait, sends the key on health checks, and refuses plaintext HTTP to a Runpod host. A production client reaches an endpoint the same way: base URL plus key from the environment, matching each model config's `endpoint_env`.
- **Parity gate.** Before an endpoint may serve, its outputs on a fixed public sample set (ThaiOCRBench pages, no client data) are compared with the self-hosted reference serving of the same model, prompt version and sampling, within stated tolerances.
- **Production requirements.** Per-model SLOs, GPU class, autoscaling settings, cost estimate, monitoring (a scheduled smoke check and endpoint health), versioning and rollback.
- **Client-data safeguards** as requirements with scenarios: no payload logging or persistence in the worker, images only in memory, TLS only, restricted API keys outside git, data-centre pinning, Secure Cloud only, a human review of Runpod's data-processing terms, and an audit of which endpoints may receive client data.

Non-goals:
- Benchmark runs on Runpod. Headline numbers stay on the H100/L4 reference hosts (latency-benchmark spec); Runpod is for serving.
- Serving the docparse role models or the docparse service itself.
- Multi-GPU or quantized serving; every model here fits a 24 GB GPU in bf16.
- A general deployment framework for other clouds.

## Capabilities

### New Capabilities

- `runpod-serving`: the per-model serving config, the worker image (build, pinning, weights, the PaddleOCR-VL two-process worker), endpoint and template deployment as code, release records and rollback, scaling and SLO settings, the smoke check and monitoring.
- `serving-parity`: the parity gate between a Runpod endpoint and the reference self-hosted serving on a fixed public sample set, with its tolerances and report.
- `client-data-safeguards`: the conditions under which client statement pages may be sent to a Runpod endpoint, enforced by the worker image, the deployment tool, the client and a recorded human review.

### Modified Capabilities

- `model-inference` (in the unarchived change `add-ocr-benchmark-harness`): endpoint credentials per model, authenticated health checks, a bounded cold-start wait, TLS enforcement for remote hosts, and how the vLLM version is recorded when `/version` is not reachable. Added as new requirements; no existing requirement changes.

## Impact

- **New code**: `runpod/` (Dockerfile, entrypoint, the Paddle supervisor), `configs/runpod/*.yaml`, `ocr_bench/runpod/` (config schema, render, deploy, release, smoke, parity) and `ocrbench runpod ...` subcommands, `tests/test_runpod_*.py`.
- **ocr_bench**: `VllmClient` and `PipelineClient` gain an API key, authenticated health and version calls, a wait-for-ready loop and the TLS check; `registry.build_model` reads `<MODEL>_API_KEY`; `cli._connect` passes the wait budget. Existing self-hosted behaviour is unchanged when the new variables are unset.
- **CI**: a workflow that builds and pushes the worker images on changes under `runpod/` or `configs/runpod/` and records their digests; a scheduled smoke workflow against the deployed endpoints (needs repository secrets).
- **Dependencies**: none new for the harness (`httpx` already covers the Runpod REST API); `gitleaks` or equivalent as a pre-commit secret scan.
- **Accounts and money**: a Runpod account with billing, a spend limit, restricted API keys, a container registry Runpod can pull from, and a human review of Runpod's DPA and terms before any client page is sent.
- **Data**: parity fixtures are ThaiOCRBench pages and model outputs only, DVC-tracked; no statement content, endpoint ids or keys in git.
