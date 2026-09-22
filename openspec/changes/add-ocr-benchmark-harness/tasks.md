# Tasks

## 1. M0 Scaffold

- [ ] 1.1 Create `pyproject.toml` (Python 3.11, `ocrbench = ocr_bench.cli:app` entry point, runtime deps from design D4–D12, `dev` extra with pytest/ruff); verify `uv sync --all-extras` succeeds in the dev container
- [ ] 1.2 Create package skeleton `ocr_bench/{data,models,normalize,metrics,confidence,latency,report}/__init__.py`, `configs/{models,datasets}/`, `tests/`; verify `python -c "import ocr_bench"` succeeds
- [ ] 1.3 Implement `ocr_bench/schemas.py` with all row models from design D2 (incl. `GroundTruth` tagged union, `OutputField` with the `error_flags` enum); verify `tests/test_schemas.py` round-trips each model through JSON and rejects an unknown `gt_kind` and unknown error flag
- [ ] 1.4 Implement `ocr_bench/jsonl.py` (typed read, append-and-flush, atomic write) and run-dir path helpers honoring `$OCRBENCH_DATA_DIR`; verify tests for append/resume and atomic replace
- [ ] 1.5 Implement config models and loaders for `run.yaml`, `models/*.yaml`, `datasets/*.yaml` with a failure on an unknown model reference; add initial `configs/run.yaml`, `configs/models/{teleocr,dotsocr,typhoon_ocr15}.yaml`, `configs/datasets/{thaiocrbench,bankstmt}.yaml`; verify a config test loads them and rejects a bad model name
- [ ] 1.6 Implement `ocr_bench/cli.py` with typer stubs for `prepare`, `infer`, `probe`, `score`, `calibrate`, `bench-latency`, `report` and a missing-upstream-file check helper; verify `ocrbench --help` lists all 7 commands (test via `typer.testing.CliRunner`)
- [ ] 1.7 Add a CI workflow running `ruff check` and `pytest -m "not live"`; verify it passes locally with `uv run pytest`

## 2. M1 Data preparation

- [ ] 2.1 Implement `data/thaiocrbench.py`: HF load, 9-task filter with a task-name assertion listing the present names, seeded (42) domain-stratified caps, GT export per task to `gt/*.json|txt|html`, `critical_fields` from the field-name rules; verify with a tiny fake dataset fixture (caps, determinism, missing-name failure, no-domain path)
- [ ] 2.2 Implement `data/bankstmt.py`: `pdffonts` routing (digital/scanned/photo), bank mapping with a hard failure on unknown, `pdftoppm -r 200` rasterization, `pdftotext -layout` + `pdfplumber` word boxes scaled to 200 dpi; verify on a generated 2-page digital PDF and an image file fixture
- [ ] 2.3 Implement the corpus coverage summary and target-mix warnings written to `prepare_summary.json`; verify a test with an under-target corpus emits the warning
- [ ] 2.4 Implement `data/degrade.py` (`clean`, `scan_low`, `photo`) with a `(sample_id, condition)`-seeded RNG and homography box transform per design D9; verify pixel-identical re-runs and that a transformed box matches the homography-mapped corners
- [ ] 2.5 Implement `data/manifest.py` and wire `ocrbench prepare` (writes images, GT, `manifest.jsonl`, `config.resolved.yaml`); verify manifest rows = 3 × samples on the fixture run and that two prepares yield identical manifests
- [ ] 2.6 Run `prepare` on real ThaiOCRBench + the statement directory on the H100 host; verify the task-name check passes, record counts in `prepare_summary.json`, and eyeball 20 random manifest rows

## 3. M2 Inference

- [ ] 3.1 Implement `models/vllm_client.py` (AsyncOpenAI, base64 image, `temperature=0/seed=0/max_tokens=4096/logprobs`, semaphore concurrency, 120 s timeout + 1 retry, error rows, timing, health check); verify with a fake async server fixture covering success, one timeout then success, double timeout, and missing logprobs
- [ ] 3.2 Implement `models/base.py` `OcrModel` protocol and the prompt registry with version tags (KIE prompts include the field list and request JSON with nulls); verify a test asserts the KIE prompt contains the sample's fields and the prediction carries `prompt_version`
- [ ] 3.3 Implement `models/parsers.py` (`markdown`, `html_table`, `json_fields`, `dots_layout_json`) returning `NormalizedPage` with `parse_error` on failure; verify with recorded responses in `tests/fixtures/responses/`
- [ ] 3.4 Implement `teleocr.py`, `dotsocr.py`, `typhoon.py` adapters and fill per-task prompts in model YAMLs; verify adapter tests on recorded responses for each task
- [ ] 3.5 Wire `ocrbench infer` (resume on existing non-error rows, per-model error-rate summary, non-zero exit on row-count mismatch); verify an interruption/resume test with the fake client
- [ ] 3.6 Implement `ocrbench probe` (5 fixed statement pages → yes/partial/no matrix + evidence ids → `probe.json`); verify with recorded responses
- [ ] 3.7 Smoke-test live on the V100 box (`COMPOSE_EXTRA=.devcontainer/compose.v100.yaml scripts/serve_dotsocr.sh up`) with `pytest -m live` on 5 pages; verify predictions include logprobs and latency
- [ ] 3.8 Run full `infer` for teleocr and dotsocr on the H100; verify zero silent drops (row counts match) and error rate < 2%, and run `probe`

## 4. M3 Normalization and scoring

- [ ] 4.1 Implement `normalize/text.py` (NFC, Thai→Arabic digits, zero-width strip, whitespace collapse, raw kept); verify spec scenarios as unit tests
- [ ] 4.2 Implement `normalize/fields.py` (amount, date with BE→CE, account digits-only, name) with unparseable-value flagging; verify tests for "15/03/2568"→"2025-03-15", "฿152,340.75 บาท"→152340.75, "123-4-56789-0"→"1234567890"
- [ ] 4.3 Implement `metrics/cer_wer.py` (CER over NFC, WER over `newmm` tokens); verify identical text → 0 and hand-computed cases
- [ ] 4.4 Port BMFL from the ThaiOCRBench scorer and add a 20-sample published fixture under `tests/fixtures/bmfl/`; verify all fixture samples within 0.01 (record a gate result consumed by the report)
- [ ] 4.5 Implement `metrics/ted.py` (TEDS on HTML via lxml + apted) and `metrics/anls.py` (threshold 0.5); verify identical table → 1 and ANLS below-threshold → 0
- [ ] 4.6 Implement `metrics/kie_f1.py` (field_exact incl. null==null, field_fuzzy CER ≤ 0.1, micro P/R/F1, false accept/false reject); verify the FA and FR spec scenarios
- [ ] 4.7 Implement scorer dispatch on `(task, gt_kind)` and wire `ocrbench score` writing `scores.jsonl` and `fields.jsonl` with all slice keys, errored predictions scored worst-case; verify byte-identical re-runs and an errored-row test
- [ ] 4.8 Implement slice aggregation with seeded 1,000-resample bootstrap over samples (task×model, condition×model, critical×model, bank×doc_type×model, domain×model); verify each aggregate has mean/n/CI and CI brackets the mean on a fixture

## 5. M4 Statement evaluation

- [ ] 5.1 Implement `normalize/statement.py` (StatementPage from parsed output, merge pages → StatementFile with source page tags); verify a 4-page merge test
- [ ] 5.2 Implement Check A: per-bank row regex layouts in `bankstmt.yaml`, header field + page CER scoring, `row_f1` on (date, amount), n/a marking for banks without a layout; verify on the generated digital PDF fixture
- [ ] 5.3 Implement `metrics/arithmetic.py` (Decimal balance equation, totals, closing balance, `first_break_row`, cross-page `page_gap`, `balance_mismatch`/`total_mismatch` flags, `row_consistency_rate`, `file_reconciles`); verify consistent, broken-row-7 and page-gap scenarios
- [ ] 5.4 Implement Check C cross-model agreement (null when one side missing) and agreement-rate reporting; verify the missing-in-one-model scenario
- [ ] 5.5 Implement `review/queue.csv` export (all disagreements + seeded 10% agreements) and the manual label CSV loader with line-numbered validation errors; verify a label round-trip test and an invalid-row test
- [ ] 5.6 Compute flag precision of Checks B/C against the manual set; verify on a synthetic labeled fixture where flag precision is known
- [ ] 5.7 Implement `metrics/duplicates.py` (variant generator: scan_low, rot180+photo, 5% crop; hard negatives; SHA-256 + Jaccard on (date, amount) + perceptual hash) and the threshold sweep {0.6,0.7,0.8,0.9}; verify exact-duplicate and sweep-row-count tests
- [ ] 5.8 Export the review queue from the H100 run and label 30–50 pages across banks/doc types; verify the loader accepts the labeled CSV and statement accuracy is computed on it

## 6. M5 Confidence calibration

- [ ] 6.1 Implement `confidence/proxy.py` features (token-offset span location with exact then digit-only match, mean/min logprob, `span_found`, `agree`, `arith` with null indicators); verify the span-not-found scenario and a digit-only match case
- [ ] 6.2 Implement `confidence/calibrate.py` logistic fit with `GroupKFold(5)` by `sample_id`, `full` and `logprob_only` variants, out-of-fold confidences; verify no sample_id crosses folds
- [ ] 6.3 Implement bands (5 bands, n/accuracy/CI), ECE (10 bins), critical-only table and reliability plot data → `calibration.jsonl`; verify on a synthetic perfectly-calibrated fixture (ECE ≈ 0)
- [ ] 6.4 Implement threshold search for `--target-acc` targets with "not reachable" handling, bucket shares/accuracies, and `thresholds.json` with weights; wire `ocrbench calibrate`; verify reproduction of bucket shares from `thresholds.json`
- [ ] 6.5 Run `calibrate --target-acc 0.99,0.995` on the H100 run; verify ECE and review rate are reported per model

## 7. M5 Latency benchmark

- [ ] 7.1 Implement `latency/bench.py`: seeded 200-page mix, streaming requests with TTFT, 30 s warm-up excluded, concurrency sweep, `latency.jsonl` rows, vLLM `/version` capture; verify against the fake streaming server (200 rows per point, ttft ≤ latency)
- [ ] 7.2 Implement the `nvidia-smi` 1 s memory sampler with "unavailable" fallback; verify a test with the command absent
- [ ] 7.3 Implement summaries (p50/p95/p99, pages/hour, GPU-hours/1,000 pages, peak memory) and multi-page document timing for 10 files; wire `ocrbench bench-latency`; verify on fixture latency rows with hand-computed percentiles
- [ ] 7.4 Run `bench-latency` on H100 and L4 (bf16, same vLLM version); verify both GPUs have a full sweep in `latency.jsonl`

## 8. M6 Report

- [ ] 8.1 Implement `report/scorecard.py` question-mapped rows (Q1–Q5, Q8) with "not benchmarked — reason" for missing inputs and capability-proxy/domain-proxy labels; verify the missing-latency scenario
- [ ] 8.2 Implement supporting tables (task, condition, critical, bank×doc_type, probe, calibration, checks A/B/C, duplicates, latency, per-file `missing_required`/`page_gap` counts) and cost per 1,000 pages with "rate not set" handling; verify on the end-to-end fixture run
- [ ] 8.3 Implement the output contract builder (error_flags incl. `low_confidence` from `t_accept`@99%) and the anonymizer (`[NAME]`, last-4 account digits, blank redacted bboxes); verify the account masking and low-confidence scenarios
- [ ] 8.4 Implement `report/render.py` (markdown, xlsx one sheet per table, plots with a Thai font: reliability, latency curves, CER by condition) and wire `ocrbench report`; verify every md table has a matching xlsx sheet
- [ ] 8.5 Add an end-to-end CPU smoke test (synthetic run with fake client: prepare → infer → score → calibrate → report); verify it passes in CI
- [ ] 8.6 Update `README.md` with the reproduce command sequence and data-handling notes; verify the commands match `ocrbench --help`
- [ ] 8.7 Generate the report for the H100 run; verify every scorecard row is filled or marked not benchmarked and `samples/` contains only anonymized fields
