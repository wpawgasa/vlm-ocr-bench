# Proposal

## Why

The client (บสย. / LooLoo) asked us to evaluate TeleOCR and dots.ocr against client questions Q1–Q5 and Q8 (sheet 01) and sections 1–3 of sheet 02. We have no labeled บสย. documents. Today the repo has only a dev container and vLLM serve scripts. There is no harness yet, so nothing can be measured, reproduced or reported. The benchmark spec (`docs/OCR Benchmark Spec — TeleOCR vs dots.ocr for บสย. LooLoo questions.md`) defines how to answer those questions from two sources:

- **Capability proxy:** ThaiOCRBench, which has human labels.
- **Domain proxy:** unlabeled Thai bank statements, scored with label-free checks.

This change turns that spec into a working, reproducible pipeline.

## What Changes

- Add the `ocr_bench` Python package (Python 3.11, `uv`, `typer`, `pydantic`, `pytest`) and an `ocrbench` CLI. The CLI has the stages `prepare`, `infer`, `probe`, `score`, `calibrate`, `bench-latency` and `report`.
- Each stage reads and writes JSONL under `runs/<run_id>/`, so any stage can be re-run on its own.
- **Data preparation:**
  - Load a ThaiOCRBench subset: 9 of 13 tasks, with a fixed seed and per-task caps.
  - Ingest bank statements: route each file to digital, scanned or photo; rasterize at 200 dpi; use the PDF text layer as ground truth.
  - Apply three deterministic image degradations: `clean`, `scan_low` and `photo`.
  - Write a manifest with one row per `(sample, condition)`.
- **Model adapters** for TeleOCR and dots.ocr, both called through vLLM OpenAI-compatible endpoints. Adapters use each model's native prompts and pipelines (TeleOCR's two-stage layout-then-block parse), with the benchmark question as fallback where no native mode exists. Every request plan carries a version tag. Adapters record logprobs, use a timeout with one retry, and write explicit error rows (no silent drops). An optional third adapter covers `typhoon-ocr1.5-2b`. A capability probe fills the Q4 matrix.
- **Normalization:**
  - Thai NFC text canonicalization.
  - Field rules: dates with Buddhist-year conversion, amounts, account numbers.
  - A statement-table builder.
- **Scoring:**
  - Scorers: CER, WER, a BMFL port, TED, field exact/fuzzy match, KIE P/R/F1 with false-accept/false-reject (FA/FR) rates, ANLS and row F1.
  - Aggregation by slice, with n and a 95% bootstrap CI on every aggregate.
- **Label-free statement evaluation:**
  - Check A: text-layer ground truth.
  - Check B: arithmetic self-consistency and file reconciliation.
  - Check C: cross-model agreement.
  - A review-queue export and a loader for the manual anchor set.
  - A duplicate-detection test set and detector for Q5.
- **Field confidence proxy:** combines logprobs, cross-model agreement and arithmetic checks. It is fitted by logistic regression and reported with calibration bands, ECE, and accept/review/reject thresholds at 99.0% and 99.5% auto-accept accuracy (Q3).
- **Latency and throughput harness:** a concurrency sweep with p50/p95/p99, pages/hour, GPU-hours per 1,000 pages and peak GPU memory, on H100 and L4 (Q8).
- **Reporting:** a scorecard with one row per client question in `.md` and `.xlsx`, plots, and anonymized Q4 samples that follow the normalized output contract with `error_flags`.

Non-goals: fine-tuning either model, building the LooLoo production pipeline, VQA/chart/infographic tasks, Q6 (model governance), Q7 (security) and pricing.

## Capabilities

### New Capabilities

- `bench-cli`: the `ocrbench` command set, run configuration (`configs/run.yaml`, model and dataset YAMLs) and the `runs/<run_id>/` layout that makes every stage re-runnable on its own.
- `dataset-preparation`: the ThaiOCRBench subset loader, bank-statement ingestion, degradation conditions and the manifest schema.
- `model-inference`: the OCR model adapter interface, the vLLM client rules (logprobs, timeout, retry, error rows), versioned per-task request plans (native prompts/pipelines with benchmark-question fallback), predictions output and the capability probe.
- `output-normalization`: parsing raw model output into `NormalizedPage`, and the text/field/statement normalization shared by predictions and ground truth.
- `accuracy-scoring`: the per-sample scorers, per-field results, slice aggregation and bootstrap confidence intervals.
- `statement-evaluation`: the statement schema, checks A/B/C, the review queue and manual anchor set, and duplicate detection.
- `confidence-calibration`: the field confidence proxy, calibration bands, ECE, threshold search and review rate.
- `latency-benchmark`: the concurrency sweep, per-page timing, GPU memory sampling, throughput and cost derivation.
- `scorecard-report`: the question-mapped scorecard, xlsx/markdown/plots, the output contract with error flags, and the anonymizer.

### Modified Capabilities

(none, since the repo has no existing specs)

## Impact

- **New code:** `pyproject.toml`, `ocr_bench/` (data, models, normalize, metrics, confidence, latency, report), `configs/`, `tests/`.
- **Existing files:**
  - `scripts/serve_{teleocr,dotsocr}.sh` and `.devcontainer/compose.yaml` already serve the models with the required vLLM flags; TeleOCR additionally needs its `TeleOCR_vllm` plugin in the serving image. The harness only consumes `$TELEOCR_BASE_URL`, `$DOTSOCR_BASE_URL` and `$TYPHOON_BASE_URL`.
  - `README.md` gains a reproduce section.
- **Dependencies:**
  - Python packages: `typer`, `pydantic`, `httpx`/`openai`, `datasets`, `pdfplumber`, `pillow`, `opencv-python-headless`, `numpy`, `pythainlp`, `rapidfuzz`, `apted` or `zss`, `scikit-learn`, `imagehash`, `openpyxl`, `matplotlib`, `pytest`.
  - System tools: poppler (`pdffonts`, `pdftoppm`, `pdftotext`), already in the image.
- **Data handling:** bank statements are client data. They stay under `data/` and `runs/`, which are gitignored. Only the anonymized samples in `report/samples/` are meant for client-facing use.
- **Compute:**
  - H100 for accuracy runs and latency.
  - L4 session for latency only.
  - V100 dev box for smoke tests only, whose numbers are not quoted.
