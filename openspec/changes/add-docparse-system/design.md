# Design

## Context

See proposal.md for motivation. Several facts about the current state shape this design:

**Data**
- The statement corpus is 31 files and 458 pages: 130 digital, 328 scanned, 7 banks (`data/runs/2026-09-22-a/prepare_summary.json`).
- Only **27 pages** have a usable text layer, and 21 of those come from one KBank file.
- 103 digital pages fail `text_layer_usable` (`ocr_bench/data/bankstmt.py:101`).
- The rest of the ground truth will come from the manual anchor set (ocr_bench task 5.8, 30–50 pages, not labelled yet).

**Models in place**
- **PaddleOCR-VL-1.6** already runs as an official two-stage pipeline: PP-DocLayoutV3, then a per-crop "OCR:" read (`configs/models/paddleocr_vl.yaml`). It gives layout, reading order and table HTML, but returns no logprobs. It had 0 errors on 6,846 rows.
- **dots.ocr** `prompt_layout_all_en` returns blocks with text, but loops on Thai: `max_tokens` on 9.9% of rows.
- **typhoon-ocr1.5** returns markdown with no boxes.
- None of the three follows free-form instructions reliably (457 of 1,149 KIE replies are not JSON).

**Code to reuse**
- `ocr_bench.models.vllm_client` and `pipeline_client`.
- The `paddle_layout` and `dots_layout_json` parsers (`ocr_bench/models/parsers.py`).
- Thai normalization (`ocr_bench/normalize/text.py`, `fields.py`).
- The statement mapper, with `COLUMN_KEYWORDS` and the header label table (`ocr_bench/normalize/statement.py`).
- Check B arithmetic (`ocr_bench/metrics/arithmetic.py`).
- `BankOverrides` (`ocr_bench/config.py:168`).
- The cluster bootstrap (`ocr_bench/metrics/aggregate.py`).

**Hardware**
- Benchmark and training runs happen on one H100 (80 GB).
- The V100 dev container has no CUDA, so unit tests must run on recorded responses.

An adversarial design review (Fable 5.1, 2026-09-23) shaped decisions D3, D5–D10 and D12. The user accepted all its recommendations except dropping the FastAPI service.

## Goals / Non-Goals

**Goals:**
- A staged parse with typed, cached artifacts per stage, so any stage can be re-run and scored on its own.
- Agentic behaviour only in the verifier and the extractor, each with hard budgets.
- No value enters a field or an answer without two independent readings or a recomputation.
- A measured answer to "does docparse beat the best single VLM?" on a leak-free eval set.

**Non-Goals:**
- A general workflow engine or a queue.
- Tuning the service for latency beyond the budgets in D11.
- Ruling-line table detection: Paddle's table HTML supplies the structure.
- Classifying from more than the first two pages.

## Decisions

### D1. Package layout and reuse
Add a new package, `docparse/`, with these modules:
- `registry`, `classify`, `layout`, `segment`, `recognize/{base,paddle_crop,trocr}`, `reconcile`, `html`, `verify`, `extract/{tools,plans,agent,verify}`;
- `trocr/{data,train,eval}`;
- `service/{cli,api}`, and `artifacts`.

Configuration lives in `configs/docparse/pipeline.yaml` and `configs/docparse/classes/*.yaml`. docparse imports from `ocr_bench` (clients, parsers, normalizers, mapper, arithmetic). `ocr_bench` imports docparse only in its `docparse` adapter.

*Alternative:* a separate repo that calls ocr_bench over HTTP. Rejected by the user, because it would duplicate the normalizers and prevent same-run scoring.

### D2. Artifacts
Each document gets a directory, `runs/docparse/<doc_id>/`. It holds one JSON artifact per stage (`classify`, `layout`, `segments`, `readings`, `reconcile`, `verify`), plus `parsed.html`, `fields.json`, `status.json` and `crops/`.

- Each artifact records the stage's config hash and the hashes of its inputs.
- A stage is skipped when both hashes match.
- `doc_id` is the SHA-256 of the file bytes, truncated to 16 hex characters. Re-uploading the same file reuses the stored work.

Pydantic models live in `docparse/schemas.py`. Artifacts are written atomically with `ocr_bench.jsonl`.

### D3. Model roles and serving
| Role | Model | Where |
|---|---|---|
| Layout and first reading | PaddleOCR-VL-1.6 pipeline (default); dots.ocr `prompt_layout_all_en` (alternative) | existing `paddle_pipeline` and vLLM services |
| Classify; verifier crop reads | Qwen3-VL-8B-Instruct | new vLLM service `qwen3vl` |
| Verifier field lookup; extraction agent | Qwen3-8B (tool calling, thinking off) | new vLLM service `qwen3` |
| Text-line detection | PP-OCRv5 DB detector | in-process, GPU if available |
| Line reading | `paddle_crop` or Thai TrOCR | pipeline service, or in-process |

The OCR VLMs are fixed-prompt models and cannot classify or answer focused prompts, hence the separate instruction-following VLM. All three vLLM servers share the H100 at `gpu-memory-utilization` 0.25 / 0.30 / 0.25. TrOCR and the detector use what is left.

*Alternative:* prompting typhoon-ocr1.5 for classification. Rejected because of the measured non-JSON rate.

### D4. Classification
Qwen3-VL receives the first two pages at 1024 px on the long side, plus the registry's class ids and descriptions. It returns JSON: `{class, confidence, bank}`. When a document has two pages, both are sent in one request. Output that fails the JSON schema or names an unknown class becomes `unknown` with `invalid_reply`.

The confidence is the model's self-report. It is not used for any decision except `unknown` below a configured floor (0.5).

### D5. Layout mode
The Paddle pipeline is the default: no loops, table HTML, 0 errors. dots.ocr `prompt_layout_all_en` is the alternative. `prompt_layout_only_en` is never used, because reconciliation needs a second text reading.

When a reply ends with `finish_reason=length`, every block text on that page is flagged `truncated` and treated as absent. The boxes are kept.

Block ids are `p{page}-b{order}`.

### D6. Segmentation
- The PP-OCRv5 DB detector runs on each non-figure block crop, and its line boxes are clipped to the block.
- Table blocks use the Paddle table HTML (rows, cols, spans) to assign each detected line to a cell by the largest overlap with the cell. Cell geometry comes from the detector boxes grouped by row and column bands.
- Lines wider than `max_aspect` (10) are split at the widest whitespace gaps, found from a column projection of the binarized crop.
- If the detector finds no lines where the block has text, a horizontal projection-profile split is used, flagged `fallback`.

### D7. Thai TrOCR
**Model**
- `VisionEncoderDecoderModel`.
- Encoder: `microsoft/trocr-base-printed` (a DeiT-base/16 encoder, per the Fable review; confirm against the checkpoint config in task 4.1). It runs at **96×768** with interpolated position embeddings. Lines are resized preserving aspect ratio to height 96 and padded right to 768.
- Decoder: **character-level, trained from scratch.** 6 layers, d=512, 8 heads, over a vocabulary of about 250 symbols: U+0E00–U+0E7F, ASCII printable, and `฿ € $ – — · •`, with NFC targets.

*Why character-level:* Thai subword vocabularies (XLM-R, WangchanBERTa) bring a large softmax and ambiguous segmentation around combining marks. Cross-attention would be freshly initialised either way. Per-character confidence also makes D8's alignment direct.

**Data**
- 1.5–2M synthetic lines. Fonts: TLWG, Sarabun, Noto Sans Thai, Kanit. Text: Thai Wikipedia plus statement-shaped generators (amounts with separators, Thai and Gregorian dates, Thai month names, channel and description phrases). Degradations reuse `ocr_bench/data/degrade.py`.
- At least 20k real crops from non-eval files only:
  - text-layer lines (task 1.2 first tries to reclaim the 103 unusable pages, e.g. PUA glyphs);
  - pseudo-labels where `paddle_crop` and typhoon crop reads agree exactly after normalization.

**Training**
- Stage 1: synthetic only. Stage 2: a mix of about 70% synthetic and 30% real.
- AdamW with cosine decay, bf16, on the H100.
- Held-out: 5% of synthetic data, plus real lines from non-training, non-eval files.

**Gate (spec line-recognition):** clean CER ≤ 2%, degraded CER ≤ 6%, and no worse than `paddle_crop` on the same lines. The relative condition is the one that matters. Otherwise the TrOCR branch adds latency without gain.

### D8. Reconciliation by alignment voting
1. For each segment, find its span in the block text: fuzzy-locate the reader text in the block text with `rapidfuzz` partial alignment, in reading order, and consume matched spans.
2. Align the reader text and the block span with Levenshtein editops, and group them into agreed and disputed spans.
3. For each disputed span, apply these tie-breakers in order:
   - the field-type validator, when the segment belongs to a typed field or column (`amount`, `date`, `account`, via the mapper's column roles);
   - a third reading of that segment only: `paddle_crop` if the reader was TrOCR, TrOCR if a checkpoint exists, otherwise a Qwen3-VL crop read. The 2-of-3 exact match wins;
   - otherwise keep the line-reader text, with `data-disputed` set and `data-alt` holding the other readings.
4. `data-conf` is the minimum over the segment's characters of each source's **isotonically calibrated** P(char correct). Calibration is fitted per source on the gate's held-out lines, and a source with no calibration gets no `data-conf` for its own contribution. Raw logprobs are never compared across sources.

**HTML**
- A `<section data-page>` per page; `<h*>` for titles; `<p>` per text block, with `<span>` per segment; `<table>` with spans.
- Every element carries `data-bbox`, `data-block-id`, `data-src` and `data-conf`.
- Attributes are serialized in sorted order, so the output is byte-deterministic.

### D9. Verifier
The field lookup is two-step:
1. Rules first: label aliases from the registry and mapper, and the value found in the same or the next cell or span.
2. Then Qwen3-8B, given only the candidate blocks and returning `{block_id, value}`, validated like any tool call.

A field goes to recovery if it is not found, fails a validator, or its `data-conf` is below `min_conf`.

**The recovery ladder**
- Regions, in order: region hint → nearest label block → whole page at 2× resolution.
- For each region: crop with a 10% margin and upscale to a height of at least 64 px per line, then run the detector.
- No text means `absent` for that region; move to the next region.
- Otherwise read the crop twice: Qwen3-VL with a focused prompt naming the field and its format, and the line reader on the detected lines.
- Accept the value only if both readings agree after `normalize_field` and the validators pass.
- Then patch by block id, keeping `data-prev`, and set `data-src="verify"`.
- Re-run the cross-field validators, which is Check B for statements. Revert the patch if they now fail.

**Budgets**
- `max_retries` 3 per field, and `max_vlm_calls` 12 per document.
- The report is regenerated from the HTML after each patch.
- The audit (crops, prompts, replies, decisions) goes to `verify.json` and `crops/`.

### D10. Extraction harness
Qwen3-8B via the vLLM OpenAI tool-calling API, temperature 0.

**Step 1: route the query.** One LLM call classifies the query as `header_field`, `table_aggregate`, `balance_on_date`, `reconciliation` or `free_form`, and fills that type's parameter schema (e.g. `{field}`; `{filter: {date_from, date_to, description_contains, direction}, agg: sum|count|max|min|list}`).

**Step 2: run the plan.** A template is a fixed list of tool calls. The LLM is called again only when a parameter fails validation, and at most `max_step_retries` times. `free_form` runs a bounded ReAct loop (8 tool calls), with the answer flagged.

**Tools** read the parse through a typed view:
- the HTML DOM, for `outline`, `find` and `get_block`;
- `fields.json`, for `get_field`;
- `StatementRow`s built by the ocr_bench mapper from the reconciled tables, each with a row id and a block id, for `get_table` and `table_query`.

`compute` evaluates an expression AST over `Decimal` with only `+ - * / abs round`, and never calls `eval`.

**Verification**
- A read value must match its cited block's text after normalization.
- An aggregate is recomputed from the cited row ids.
- A field that is `absent` in `fields.json` makes the answer `absent`.

### D11. Latency budget (H100, per page, Paddle layout)
| Stage | Happy path |
|---|---|
| Layout | 2–6 s (4 pipeline replicas) |
| Detection | ≈0.1 s |
| Line reading | TrOCR 1–2 s for about 300 segments batched; `paddle_crop` 3–8 s |
| Classify | 1 call per document |

The happy path totals about 10–20 s per page. The verifier's worst case is 12 VLM calls per document. Extraction takes 2–4 LLM calls for template queries and at most 8 tool calls for free-form ones.

The service processes pages of a document concurrently, up to a configured limit. No latency SLO is promised in v1. The measured numbers go in the evaluation report.

### D12. Evaluation
- `configs/docparse/eval_files.v1.yaml` lists the file ids: every file with a usable text-layer page, plus the files in the manual anchor set. It is frozen before any docparse tuning (task 1.3).
- The `ocr_bench` adapter `docparse` (plan kind `docparse`) runs the parse pipeline per manifest page and condition. It emits `parsed.html` converted to the harness's `NormalizedPage`, so existing scorers and Check B apply.
- `docparse eval compare` computes paired cluster-bootstrap CIs (clusters = files) of docparse − best VLM, per condition and bank, on field F1 and CER. The best VLM is chosen by mean field F1 on the same eval samples.
- The Q/A set generator draws on the text-layer and anchor ground truth, and keeps 10% of its queries unanswerable (bank-conditional absent fields, out-of-period dates).

## Risks / Trade-offs

- **[The eval set is small and dominated by one bank]** → Report per bank, and state the CI width. The manual anchor set (ocr_bench 5.8) is on the critical path, so its labelling starts in task group 1.
- **[TrOCR never beats `paddle_crop`]** → The reader interface makes that outcome cheap. `paddle_crop` stays the default, and the report says so. TrOCR training starts only after Spike A has measured the baseline's CER.
- **[Three vLLM servers on one H100 contend for memory and compute]** → Fixed memory fractions (D3). Qwen3-VL and Qwen3 are idle on the happy path. The measured throughput goes in the report.
- **[The verifier cannot reach `complete` on degraded photos]** → The field states make that visible as `unverified` or `absent`, never as a wrong value. The review queue can consume `unverified` fields later.
- **[An 8B LLM misroutes queries]** → Routing is scored on the Q/A set. Parameters are schema-validated, and a wrong plan fails verification rather than returning a wrong value.
- **[Pseudo-labels copy shared reader errors into TrOCR training]** → An exact 2-of-2 agreement is required, and pseudo-labelled lines are capped at 50% of the real crops.

## Migration Plan

This is a new package, so nothing needs migrating. The new vLLM services are opt-in compose profiles (`docparse`). Rolling back means removing the profile. ocr_bench runs that do not select the `docparse` model are unaffected.

## Open Questions

- The exact Qwen3-VL and Qwen3 checkpoints and quantization can be settled when the services are brought up in task 2.1. The model roles and interfaces don't depend on them.
