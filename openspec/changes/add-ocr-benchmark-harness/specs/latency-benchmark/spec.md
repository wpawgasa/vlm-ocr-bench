# Spec Delta

## Purpose

Measures per-page latency and throughput of each served model on each GPU type across client concurrency levels. The results answer Q8 and give GPU-hours and cost per 1,000 pages.

## ADDED Requirements

### Requirement: Fixed benchmark mix and sweep
`ocrbench bench-latency` SHALL run a fixed mix of 200 pages under the `clean` condition: 100 ThaiOCRBench full-page samples and 100 statement pages, selected with a fixed seed. It SHALL run this mix for each model at each client concurrency in the configured list (default 1, 4, 8, 16, 32). Each concurrency point SHALL process all 200 pages after a 30 s warm-up whose requests are excluded from results.

#### Scenario: Sweep coverage
- **WHEN** `bench-latency --gpu h100 --concurrency 1,4,8,16,32` completes for two models
- **THEN** `latency.jsonl` has 200 non-warm-up page rows for each (model, gpu, concurrency)

### Requirement: Per-page timing record
Each timing row SHALL describe one page: `model`, `gpu`, `concurrency`, `sample_id`, end-to-end `latency_ms`, `ttft_ms`, `n_requests`, `prompt_tokens`, `completion_tokens`, `image_px` and `error`. For a multi-request page (TeleOCR two-stage), `latency_ms` SHALL cover all its requests and `ttft_ms` SHALL be taken from its first request. Client concurrency SHALL count pages in flight.

#### Scenario: Streaming TTFT
- **WHEN** a page completes
- **THEN** `ttft_ms` is the time to the first streamed token of its first request and is ≤ `latency_ms`

#### Scenario: Two-stage page timing
- **WHEN** a TeleOCR page needs 1 layout request and 5 block requests
- **THEN** its row has `n_requests=6` and `latency_ms` spans from the first request's start to the last request's end

### Requirement: Throughput summary and GPU memory
For each (model, gpu, concurrency), the system SHALL report:

- p50, p95 and p99 seconds per page
- pages per hour
- GPU-hours per 1,000 pages
- peak GPU memory, from GPU memory sampling at 1 s intervals during the run

When GPU sampling is unavailable, peak memory SHALL be reported as unavailable, not as zero.

#### Scenario: No GPU telemetry
- **WHEN** GPU memory queries fail in the harness container
- **THEN** the summary marks peak memory "unavailable" and the latency results are still written

### Requirement: Hardware and precision discipline
Runs SHALL record the GPU type, vLLM version and dtype. H100 80 GB and L4 24 GB SHALL use the same vLLM version and bf16. A quantized row on L4 SHALL be allowed only for a released quantized checkpoint named in configuration. The harness SHALL NOT quantize models itself.

#### Scenario: Mismatched environment
- **WHEN** H100 and L4 runs report different vLLM versions
- **THEN** the report shows a warning beside the Q8 table

### Requirement: Multi-page document latency
The system SHALL time 10 multi-page statement files end to end, including per-page inference and page merge. It SHALL report per-document p50 and p95.

#### Scenario: Document timing
- **WHEN** multi-page timing completes
- **THEN** there is one document-level latency row per (model, gpu, file)

### Requirement: Derived cost
Cost per 1,000 pages SHALL be derived in the report as GPU-hours per 1,000 pages × the GPU hourly rate configured in `run.yaml`. The rate SHALL be shown as an input assumption.

#### Scenario: Rate absent
- **WHEN** no GPU hourly rate is configured
- **THEN** the cost column shows "rate not set" and no cost figure is invented
