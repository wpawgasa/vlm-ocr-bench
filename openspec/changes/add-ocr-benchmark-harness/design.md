# Design

## Context

The repo currently has:

- A dev container: Python 3.11, `uv`, poppler, Thai fonts and opencv runtime libraries, with no CUDA.
- A compose stack that serves TeleOCR, dots.ocr and typhoon-ocr1.5-2b with vLLM `v0.11.0`, and OvisOCR2 and PaddleOCR-VL-1.6 (plus PaddleX's pipeline server) with vLLM `v0.22.1` (D16). The flags are those from spec §4.3, plus `--max-logprobs=5`.
- `scripts/serve_{teleocr,dotsocr,ovisocr2,paddleocr_vl}.sh`.

The harness container sees `$TELEOCR_BASE_URL`, `$DOTSOCR_BASE_URL`, `$TYPHOON_BASE_URL`, `$OVISOCR2_BASE_URL`, `$PADDLEOCR_VL_BASE_URL`, `$PADDLE_PIPELINE_URL` and `$OCRBENCH_DATA_DIR=/data`, and has `nvidia-smi` for utility access only.

Code is written on a V100 dev box, where numbers are smoke-test only. Accuracy and latency runs happen on an H100, with an L4 session for latency.

There is no Python code yet. See proposal.md for motivation. The source-of-truth narrative is `docs/OCR Benchmark Spec — TeleOCR vs dots.ocr for บสย. LooLoo questions.md`, and the specs in this change formalize it.

## Goals / Non-Goals

**Goals:**
- One-directional module flow `data → models → normalize → metrics/confidence → report`. Raw model text is touched only in `normalize`.
- Every stage is a pure function of files in `runs/<run_id>/` plus config, so stages can be re-run and resumed, and inference cost is paid once.
- Every number in the report is traceable to rows and carries n and a CI.
- Tests run on CPU without model servers, by using recorded fixture responses.

**Non-Goals:**
- A general OCR benchmarking framework. Only the 9 tasks and the models in `configs/run.yaml` (four after the H100 screening, D16) are supported.
- Async job orchestration, databases or dashboards. JSONL files plus a CLI are enough.
- Loading model weights in the harness, or any quantization.

## Decisions

### D1. Package layout mirrors spec §2
The package is `ocr_bench/` with the subpackages `data/`, `models/`, `normalize/`, `metrics/`, `confidence/`, `latency/` and `report/`, plus `schemas.py` and `cli.py`. Keeping the spec's layout makes the doc and the code navigable together.
*Alternative:* a flat module per stage. Rejected, because metrics and normalizers are unit-tested independently and grow separately.

### D2. Pydantic models are the only row schema; JSONL is the only interchange
`schemas.py` defines these row types: `Sample`/`ManifestRow`, `GroundTruth`, `RawPrediction`, `NormalizedPage`, `Block`, `FieldValue`, `PredictionRow`, `ScoreRow`, `FieldResult`, `StatementPage`, `StatementFile`, `CalibrationRow`, `LatencyRecord` and `OutputField` (the §9.2 contract).

- A small `jsonl.py` module provides typed `read_rows(path, Model)` and `append_rows(path, rows)`.
- Writes are append-and-flush, so `infer` can resume.
- Final artifacts from `score`, `calibrate` and `report` are written to a temporary file and then renamed, which makes them atomic.

*Alternative:* Parquet. Rejected for now, because JSONL is greppable with `jq` (already in the image) and row counts are small (about 4.5k samples × 3 conditions × 2 models). `pandas` is used only at aggregation time.

### D3. Configuration via YAML + pydantic settings
- `configs/run.yaml` lists models, datasets, conditions, concurrency, the GPU hourly rate and seeds.
- `configs/models/*.yaml` holds each model's endpoint env var, served name, per-task `{prompt, version}` and parser id.
- `configs/datasets/*.yaml` holds task-name mapping, caps, the seed, the bank-name mapping and per-bank row regex layouts.

Configs are loaded into pydantic models, so a bad config fails before any work starts. `prepare` writes `config.resolved.yaml` (run and datasets only) and `infer` writes `config.infer.yaml` (models). The DVC `prepare` stage depends only on `configs/run.yaml`, `configs/datasets`, `data/statements` and `ocr_bench/data`, so model-side changes never mark prepared data stale.

### D4. vLLM access through the `openai` SDK, adapters as request plans
`models/vllm_client.py` wraps `openai.AsyncOpenAI` as follows:

- The image is sent as a base64 PNG data URL.
- Requests set `logprobs=True, top_logprobs=0`, plus `temperature=0` and `seed=0`. `max_tokens` is 4096 unless the model's config overrides it.
- Official per-model decoding goes through `extra_body`: TeleOCR's `presence_penalty`/`frequency_penalty` (standard fields) and `no_repeat_ngram_size`, which only its vLLM plugin understands. Each model config states its settings per request type, and `config.infer.yaml` records them.
- `asyncio.Semaphore(concurrency)` counts **pages** in flight. A page's own requests (TeleOCR's block calls) run concurrently inside it, bounded by a per-page limit of 8.
- `asyncio.wait_for` with a 120 s timeout and 1 retry, **per request**.
- `time.perf_counter` timing, taken per page.
- Streaming is used only in the latency harness (for TTFT).

Adapters (`teleocr.py`, `dotsocr.py`, `typhoon.py`) implement `OcrModel`. Each maps `(task, subtask)` to a **request plan** from its config:

- `single`: one call on the whole image.
- `crop_single`: the harness crops the fine-grained region, then makes one call.
- `grounding`: one call with a bbox embedded in the prompt (dots.ocr).
- `two_stage`: TeleOCR's layout call at 1036×1036, then one call per detected block, following TeleOCR's official client:
  - Crop from the original image, using polygon masks when the box has more than 2 points.
  - Rotate the crop by the block's angle.
  - Pad or upscale via `resize_by_need`: maximum edge ratio, minimum edge.
  - Use a type-specific prompt and sampling.
  - Skip `image`, `list` and `equation_block` blocks.
- `question`: send the sample's benchmark question (the fallback).
- `pipeline`: post the whole page to the model's vendor pipeline server (D15) and parse the structured result it returns.
- `crop_pipeline`: crop the fine-grained region, then `pipeline`.

The plan's version tag is `prompt_version`. The policy table lives in the model-inference spec.
*Alternative:* raw `httpx`. Rejected: the SDK handles the multimodal message shape and logprob parsing, and the code stays small.
*Alternative:* one prompt per task for every model. Rejected: all three models are trained on fixed prompts. See the model-inference spec.

### D5. Parsers per output dialect, not per model
`models/parsers.py` provides these parsers:

- `markdown`, `html_table` and `json_fields`.
- `dots_layout_json`: a list of `{bbox, category, text}`, mapped to `Block`. Table text is HTML.
- `teleocr_layout`: lines of `<box:x1 y1 … ><label:type><angle tag>` with 0–1000 coordinates, possibly polygons, mapped to blocks. Block texts come from the block calls.
- `otsl_to_html`: TeleOCR tables, with spans, ported from TeleOCR's reference `convert_otsl_to_html`.
- `typhoon_markdown`: markdown with embedded `<table>` HTML, `<figure>` and `<page_number>` tags.
- `ovis_markdown`: markdown with `<table>` HTML; figure paragraphs `<img src="images/bbox_l_t_r_b.jpg" />` (0–1000 units) become figure blocks and are dropped from the text, then OvisOCR2's official `clean_truncated_repeats` cuts a looping tail, as the model card's parser does.
- `paddle_layout`: the JSON page of PaddleX's `/layout-parsing` response; `prunedResult.parsing_res_list` blocks (`block_label`, `block_content`, `block_bbox`) become blocks, table content is already HTML. Page text joins every non-figure block, headers and footers included (the pipeline's own markdown drops them, and statements keep account details in headers).

Each model config names a parser per request type. Parsing failures produce a `NormalizedPage` with `parse_error` and never raise, as the output-normalization spec requires. The statement mapper (M4) consumes only `NormalizedPage.tables` and `.blocks`, so it is model-independent.

### D6. Metric implementations
- **CER:** `rapidfuzz.distance.Levenshtein` over NFC strings.
- **WER:** PyThaiNLP `newmm` tokens, with the same Levenshtein over token lists.
- **TED:** `apted`, using the TEDS formula `1 − TED/max(nodes)` on HTML trees parsed with `lxml`. This is the formula ThaiOCRBench and OmniDocBench use.
- **ANLS:** a small local implementation.
- **Official score (`tob_score`, incl. BMFL):** ported from OCRBench v2's evaluator (MIT; ThaiOCRBench's scorer is an unlicensed adaptation of it, so its Thai-specific changes are reimplemented, not copied). A parity fixture of ≥20 published per-sample scores per kept task (ThaiOCRBench `res_folder`, Qwen2.5-VL-72B) gates comparability per task. NLTK METEOR needs WordNet data, downloaded in post-create and CI.
- **Bootstrap:** a seeded numpy resample over the sample index. For field metrics, resampling happens over samples, not fields, to respect clustering.

### D7. Ground-truth representation
`GroundTruth` is a tagged union by `gt_kind`:

- `text`: a string.
- `html`: a table/document tree.
- `json`: fields and a label.
- `text_layer`: `gt_text`, `gt_words` and parsed `StatementPage` headers and rows.
- `none`.

Scorers dispatch on `(task, gt_kind)`. A pair with no scorer means the metric is not applicable, and it is recorded explicitly.

### D8. Statement ingestion
- `pdffonts` routes each file by the number of font lines in its output.
- `pdftoppm -r 200 -png` rasterizes pages.
- `pdftotext -layout` extracts page text.
- `pdfplumber` extracts word boxes, scaled from PDF points to 200 dpi pixels (×200/72).
- The bank comes from a `bank_map` in `bankstmt.yaml`, keyed by filename pattern or subfolder.

Row ground truth comes from per-bank regex layouts. A bank without a layout still gets text and header scores.

### D9. Degradation with OpenCV and seeded RNG
The seed is `int(sha256(f"{sample_id}|{condition}").hexdigest()[:8], 16)`. The transforms are:

- `scan_low`: downscale by 150/200 and upscale back, `GaussianBlur` σ=0.6, then JPEG q40.
- `photo`: builds one 3×3 homography that composes rotation and perspective, applies it with `warpPerspective`, then applies a brightness factor and JPEG q55. The homography is stored in the manifest's GT sidecar and used to transform boxes.

Degraded images are written once, and later stages only read them.

### D10. Confidence proxy fitting
- **Span location:** the token-offset map of the completion is used to find the normalized field's raw string, first by exact match and then by digit-only match for amounts. Logprobs are sliced over that span.
- **Model:** `sklearn.linear_model.LogisticRegression` on the features `[mean_lp, min_lp, agree_1, agree_null, arith_1, arith_null]`. The indicators handle nulls.
- **Folds:** `GroupKFold(5)` by `sample_id`. Bands, ECE and thresholds are computed on out-of-fold predictions only.
- **Threshold search:** a grid over sorted unique confidences. `t_accept` is the smallest threshold whose accepted set reaches the target accuracy with the largest share. `t_reject` is chosen so the rejected set's accuracy is ≤ 50%.

Everything is fitted per model. The two variants are `full` and `logprob_only`.

### D11. Latency harness
The harness uses the same client in streaming mode.

- A background thread samples `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1`. Memory is only visible when the harness runs on the same host as the vLLM container, which is true for the compose setup.
- The vLLM version comes from the server's `/version` endpoint.
- Warm-up requests run for 30 s before the timed pass. They are tagged `warmup=true` and excluded from results.

### D12. Report
- `report/scorecard.py` builds tables as pandas DataFrames from the run files.
- `render.py` writes markdown (`DataFrame.to_markdown`, which needs `tabulate`), xlsx (`openpyxl`, one sheet per table) and matplotlib plots, using a Thai-capable font from `fonts-thai-tlwg`.
- The anonymizer operates on `OutputField` objects before they are serialized.

### D13. Testing strategy
- Unit tests cover normalizers (Buddhist year, Thai digits, amounts), each metric, arithmetic checks, degradation determinism, the threshold search and anonymization.
- Recorded vLLM responses in `tests/fixtures/responses/` drive adapter and parser tests with no server.
- An end-to-end smoke test builds a tiny synthetic run of 3 ThaiOCRBench-like samples and 1 generated digital PDF, then runs `prepare → score → calibrate → report` with a fake model client.
- Live-server tests are marked `@pytest.mark.live` and skipped by default.

### D14. TeleOCR serving with its vLLM plugin
TeleOCR's official vLLM path registers a modified `Qwen2_5_VLForConditionalGeneration` via the `TeleOCR_vllm` plugin (from github.com/caipeng328/TeleOCR), and its sampling uses `no_repeat_ngram_size`. The compose `teleocr` service therefore builds a small derived image, `.devcontainer/teleocr-vllm.Dockerfile` (`FROM ${VLLM_IMAGE}`), which pip-installs the plugin at a pinned commit, instead of using the stock image. dots.ocr and typhoon keep the stock vLLM image.
*Alternative:* the stock image with the upstream Qwen2.5-VL class. Rejected: it silently ignores `no_repeat_ngram_size` and TeleOCR's architecture changes, which would understate the model.

### D15. PaddleOCR-VL-1.6 as its official pipeline server
PaddleOCR-VL is officially two-stage: PP-DocLayoutV3 finds blocks, labels and reading order, then the 0.9B VLM reads each crop with a label-specific prompt (`OCR:`, `Table Recognition:` → OTSL → HTML, `Formula Recognition:`). The block post-processing (layout NMS, per-label merge modes, block merging) is about 3k lines of PaddleX code. Rather than port it, the compose `paddle_pipeline` service runs PaddleX's own serving app (`paddlex --serve`, `POST /layout-parsing`) on the pipeline config shipped with the installed PaddleX (`PaddleOCR-VL-1.6`), changing only `VLRecognition.genai_config` to the documented `vllm-server` backend pointed at the `paddleocr_vl` vLLM service (served as `PaddleOCR-VL-1.6-0.9B`, the name the pipeline requests). Layout runs on CPU (`paddlepaddle` CPU wheel), keeping the GPU for vLLM and the image small (`.devcontainer/paddle-pipeline.Dockerfile`, paddleocr 3.7.0). `models/pipeline_client.py` posts the page as base64 PNG, retries once on timeout/5xx like the vLLM client, and never raises. `kie`/`classify` bypass the pipeline and send the benchmark question straight to the vLLM endpoint.
*Trade-off:* the pipeline returns no tokens or logprobs, so PaddleOCR-VL has no logprob confidence (confidence-calibration spec) and its TTFT is null; its page latency includes the CPU layout stage.
*Alternative:* serve PP-DocLayoutV3 alone and port the crop/merge/assemble logic into the harness. Rejected: about a day of work, and scores could drift from the real product; it would keep logprobs, which a logging proxy between pipeline and vLLM could still add later.

### D16. Model line-up after the H100 screening, and mixed vLLM versions
The first H100 run showed TeleOCR does not read Thai (0% Thai characters on statement pages in bf16 with official sampling; a clean Thai header crop is dropped with presence penalty 1.0 and 0.0 alike, so it is the model, not the harness). It was stopped at 852 rows, kept as evidence, and removed from `run.yaml`. typhoon-ocr1.5-2b (Thai-specific) became a main model, and two Thai-capable parsers were added: OvisOCR2 (0.85B, Qwen3.5 base, end-to-end, one official prompt; no stated Thai support, so a Thai smoke test gates its full run) and PaddleOCR-VL-1.6 (0.9B, 109 languages incl. Thai, D15).
Both new models need a newer vLLM than the pinned v0.11.0 (Qwen3.5's GDN layers and `--gdn-prefill-backend`: ≥ 0.18; PaddleOCR-VL: ≥ 0.11.1). They run on `VLLM_IMAGE_NEW` (v0.22.1); dots.ocr and typhoon stay on v0.11.0 so the finished dots.ocr run is not redone. Accuracy is compared across images; latency only with the version annotated (latency-benchmark spec). OvisOCR2's card passes min/max pixels as vLLM `mm_processor_kwargs`; the harness instead pre-resizes to the same bounds at factor 32, which avoids depending on that API shape and which the server's processor keeps unchanged.
**Sensitivity variants** are separate model configs (e.g. `dotsocr_t01`: dots.ocr at upstream's T=0.1) run in their own run id on a subset (rows where greedy dots.ocr hit `max_tokens`). They explain a behaviour (dots.ocr's Thai repetition loops persist on 392 of 680 capped rows at T=0.1) and are never reported beside headline scores.

### D17. Results are DVC-tracked per file
The DVC `prepare` stage owns only `img`, `gt`, `manifest.jsonl`, `config.resolved.yaml` and `prepare_summary.json`, so each run's results (`predictions.jsonl`, `config.infer.yaml`, infer logs, `evidence/`) are `dvc add`-ed per file, with `.gitignore` letting `data/runs/*/*.dvc` through. With the hardlink cache these files are read-only: `dvc unprotect` them before resuming `infer`, then `dvc add`, `dvc push` and commit the pointers.

## Risks / Trade-offs

- **Task names, domain field or license in ThaiOCRBench differ from expectations** → The loader asserts task names and prints what is present, and the domain slice is optional. Confirm the license before quoting.
- **Official-score port drifts (e.g. PyThaiNLP/NLTK version differences)** → A mandatory per-task parity gate applies. On failure, the report marks BMFL "not comparable" and CER/WER still stand.
- **A model emits no bbox or no JSON** → The probe records `no`. The scorecard never errors on a missing capability.
- **Logprob span matching fails often (for example, the model reformats numbers)** → The digit-only fallback is used, and `span_found` rate is reported. The `agree` and `arith` features still carry signal.
- **Some handwriting questions are QA ("what does item 3 say?")** → Native OCR returns the whole text, so these samples score poorly by construction. The report shows handwriting separately, with this caveat.
- **TeleOCR is trained for Chinese and English only** → Realized on the H100: it does not read Thai, and it was screened out (D16). The report's screening note states it with evidence.
- **OvisOCR2 has no stated Thai support** → A Thai smoke test (header crop plus a few statement and ThaiOCRBench pages) gates its full run; a failure is reported like TeleOCR's.
- **PaddleOCR-VL returns no logprobs through its pipeline** → Its confidence uses agreement and arithmetic features only, and the report says so (D15).
- **Mixed vLLM versions across models** → Versions are recorded per endpoint, accuracy is compared, latency is annotated (D16).
- **Shared GPU host with a tight root disk** → The containerd image store keeps each vLLM image twice (~35–38 GB); a full disk crashed another team's MongoDB once. Check headroom before pulls, archive unused images to GCS, and guard long runs with a free-space check.
- **Native prompts make KIE/classification mostly unsupported** → The fallback question is still sent and scored. The probe records `no` for JSON extraction, which answers Q4 honestly.
- **Too few digital statement pages (< 40)** → `prepare` warns, and the manual set becomes the primary statement label source.
- **Calibration is fitted on benchmark data, not บสย. documents** → The report states that thresholds must be re-fitted on client documents. `thresholds.json` makes the procedure repeatable.
- **Client data leakage** → `data/`, `runs/`, `review/` and `*.pdf` are gitignored, only `report/samples/` is anonymized, and there is a test for the anonymizer.
- **Different vLLM versions on H100 vs L4** → Versions are recorded and a report warning is shown.
- **GPU memory is invisible from the harness when the servers are remote** → Peak memory is reported as "unavailable", not zero.

## Migration Plan

This is a greenfield addition with nothing to migrate. For rollout:

1. Build milestones M0–M2 on the V100 box against a smoke-mode server.
2. Run the full `prepare/infer/score` on the H100.
3. Label the manual set.
4. Run `calibrate`, latency on the H100 and L4, then `report`.

Rollback is deleting `runs/<run_id>/`.

## Open Questions

- Which GPU hourly rate the cost line uses. This is a config input and does not affect code.
- Who labels the manual anchor set (about 4 hours for 50 pages). This does not affect code.
- ~~Whether typhoon-ocr1.5-2b runs as a third reference.~~ Resolved: it is a main model (D16).
- The size and bank mix of the statement corpus. This affects only the warnings and the strength of the conclusions.
