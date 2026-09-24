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

An adversarial design review (Fable 5.1, 2026-09-23) shaped decisions D3, D5–D10 and D12. The user accepted all its recommendations except dropping the FastAPI service. D7 was later revised to use the existing muocr checkpoint instead of training a model from scratch.

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
- `trocr/{data,train}` (only for the conditional fine-tune);
- `service/{cli,api}`, and `artifacts`.

Configuration lives in `configs/docparse/pipeline.yaml` and `configs/docparse/classes/*.yaml`. docparse imports from `ocr_bench` (the model registry and `build_model`, clients, parsers, normalizers, mapper, arithmetic). `ocr_bench` imports docparse only in its `docparse` adapter.

*Alternative:* a separate repo that calls ocr_bench over HTTP. Rejected by the user, because it would duplicate the normalizers and prevent same-run scoring.

### D2. Artifacts
Each document gets a directory, `runs/docparse/<doc_id>/`. It holds one JSON artifact per stage (`classify`, `layout`, `segments`, `readings`, `reconcile`, `verify`), plus `parsed.html`, `fields.json`, `status.json` and `crops/`.

- Each artifact records the stage's config hash and the hashes of its inputs.
- A stage is skipped when both hashes match.
- `doc_id` is the SHA-256 of the file bytes, truncated to 16 hex characters. Re-uploading the same file reuses the stored work.

Pydantic models live in `docparse/schemas.py`. Artifacts are written atomically with `ocr_bench.jsonl`.

### D3. Model roles and serving
| Role (`pipeline.yaml`) | Used for | Default registry entry → model | Where |
|---|---|---|---|
| `layout` | Layout and first reading | `paddleocr_vl` → PaddleOCR-VL-1.6 pipeline; alternative `dotsocr` → dots.ocr `prompt_layout_all_en` | existing `paddle_pipeline` and vLLM services |
| `classifier` | Classify | `qwen3vl` → Qwen3-VL-8B-Instruct | new vLLM service `qwen3vl` |
| `verifier_reader` | Verifier crop reads; the third reading in D8 | `qwen3vl` | the same service |
| `llm` | Verifier field lookup; extraction agent | `qwen3` → Qwen3-8B (tool calling, thinking off) | new vLLM service `qwen3` |
| (none) | Text-line detection | PP-OCRv5 DB detector, not in the registry | in-process, GPU if available |
| `line_reader` (D7) | Line reading | `paddle_crop` (through the `paddleocr_vl` pipeline) or Thai TrOCR (a checkpoint path, not in the registry) | pipeline service, or in-process (PyTorch; ONNX optional) |

**Binding to the model registry.** `pipeline.yaml` names a registry entry for each role:

```yaml
roles:
  layout: paddleocr_vl
  classifier: qwen3vl
  verifier_reader: qwen3vl
  llm: qwen3
line_reader: paddle_crop
```

- Each role names an `ocr_bench` registry entry (`configs/models/<name>.yaml`), built with `build_model` the same way the harness builds a model. Switching a role's model is a config edit, plus a new YAML for a model not yet in the registry. Only a new `layout` model needs code: a layout adapter that produces the D5 block schema.
- Prompts are model-specific, so each role's prompts, sampling and chat-template settings live in the entry's `roles.<role>` section, each with a version. Examples: `roles.classifier.prompt` with a `{classes}` placeholder, and `roles.llm.sampling.chat_template_kwargs: {enable_thinking: false}`.
- `ModelConfig` changes in a backward-compatible way. It gains an optional `roles` section. `plans` becomes optional, and the harness refuses to run an entry that has none. `Sampling` gains `chat_template_kwargs`, sent through `extra_body`. `VllmClient` gains a chat call for text-only messages and tool definitions.
- Startup validation refuses a missing entry, a missing `roles.<role>` section, or a `layout` model without an adapter. Artifacts record the model name, `served_model_name`, role prompt version and vLLM version. The resolved entry is part of the stage config hash (D2), so a swap re-runs only the stages that use that role and the stages downstream of them.
- TrOCR and the detector load weights in-process, and the registry is HTTP-only (model-inference spec), so they stay out of it.

The OCR VLMs are fixed-prompt models and cannot classify or answer focused prompts, hence the separate instruction-following VLM. All three vLLM servers share the H100 at `gpu-memory-utilization` 0.25 / 0.30 / 0.25. TrOCR and the detector use what is left.

**Checkpoints (task 2.1):** `Qwen/Qwen3-VL-8B-Instruct` and `Qwen/Qwen3-8B`, both bf16 with no quantization, on `VLLM_IMAGE_NEW` (v0.22.1; Qwen3-VL needs vLLM ≥ 0.11). Both run with `max-model-len` 16384. `qwen3vl` accepts 2 images per prompt, for two-page classification. `qwen3` serves tool calls with the `hermes` parser; thinking is turned off per request through `chat_template_kwargs`. The bf16 weights take about 16.3 GiB of the 23.9 GiB (0.30) budget and 15.3 GiB of the 19.9 GiB (0.25) budget. At 147 KB per token of KV cache, that leaves roughly 30k tokens for `qwen3vl` and 20k for `qwen3`, after activations and CUDA graphs. Both fit one 16,384-token request, but `qwen3` has little room for concurrent long requests. If vLLM refuses to start because the KV cache is smaller than `max-model-len`, raise `QWEN3_GPU_MEM`. The throughput measured in task 2.1 goes in the report. The compose services are `qwen3vl` and `qwen3` (profile `docparse`), and `PADDLEOCR_VL_GPU_MEM=0.25` sets Paddle's share.

*Alternative:* prompting typhoon-ocr1.5 for classification. Rejected because of the measured non-JSON rate.
*Alternative:* a docparse-only model config. Rejected: one served model would be described in two places, and the harness could not benchmark the role models.

### D4. Classification
The `classifier` model (Qwen3-VL by default) receives the first two pages at 1024 px on the long side, plus the registry's class ids and descriptions. It returns JSON: `{class, confidence, bank}`. When a document has two pages, both are sent in one request. Output that fails the JSON schema or names an unknown class becomes `unknown` with `invalid_reply`.

The confidence is the model's self-report. It is not used for any decision except `unknown` below a configured floor (0.5).

### D5. Layout mode
The Paddle pipeline is the default: no loops, table HTML, 0 errors. dots.ocr `prompt_layout_all_en` is the alternative. `prompt_layout_only_en` is never used, because reconciliation needs a second text reading.

When a reply ends with `finish_reason=length`, every block text on that page is flagged `truncated` and treated as absent. The boxes are kept.

Block ids are `p{page}-b{order}`.

### D6. Segmentation
- The PP-OCRv5 DB detector runs on each non-figure block crop, and its line boxes are clipped to the block.
- Table blocks use the Paddle table HTML (rows, cols, spans) to assign each detected line to a cell by the largest overlap with the cell. Cell geometry comes from the detector boxes grouped by row and column bands.
- Lines wider than `max_aspect` are split at the widest whitespace gaps, found from a column projection of the binarized crop. `max_aspect` defaults to the active reader's input aspect ratio: 6 for muocr's 64×384. For `paddle_crop`, which has no fixed input, it is 10.
- If the detector finds no lines where the block has text, a horizontal projection-profile split is used, flagged `fallback`.

### D7. Thai TrOCR (muocr)
**Model (as delivered).** We use the existing fine-tuned checkpoint `models/muocr-base-26m-stage2-finetuned-20240820-v1` (gitignored; DVC/GCS, task 4.1). Its training data does not overlap the client statements or ThaiOCRBench (confirmed by the user, 2026-09-23). It is:
- a `VisionEncoderDecoderModel` (transformers 4.43);
- a ViT-base/16 encoder at a **64×384** input, 3 channels, normalized with mean and std 0.5;
- a 2-layer TrOCR decoder, d=1024, 16 heads;
- a 62,312-token Thai+English SentencePiece vocabulary (`source.spm`/`target.spm`, `vocab.json`);
- an ONNX export (encoder, decoder, decoder-with-past).

*Why not the from-scratch character-level decoder from the first draft:* the checkpoint already reads Thai. Its own held-out results report mean CER 0.9% on PDF lines, 1.2% on scanned and 0.9% on camera. Those results are on forms, not statements; the per-row files were deleted because they contained personal data. So from-scratch training would spend GPU time to reach where muocr already is. What we do not know is how it reads **statement** lines, which Spike A and the gate measure.

**Inference**
- Resize crops keeping aspect ratio to height 64, then pad right to 384 with the background colour. Segments are pre-split to at most 6:1 (D6), so nothing is squashed.
- Decoding: beam 4, `max_new_tokens` 120 and **`no_repeat_ngram_size: 0`**. The checkpoint's default of 3 blocks repeated token trigrams, which corrupts amounts such as `1,000,000.00` and account numbers with repeated digits. Its own number set showed 40% exact match despite a 6% mean CER. Spike A measures numeric lines with both settings, and the setting is recorded in the reader version.
- Confidence: each generated token's probability, taken from the beam's final scores, is assigned to every character that token decodes to. This gives the per-character confidences D8 calibrates.
- Runtime: PyTorch on GPU in batches by default. The ONNX export is an optional CPU path for the service, and must reproduce the PyTorch text on the gate lines.

**Conditional fine-tune (only if muocr fails the gate, task 4.4)**
- Start from muocr and keep its tokenizer.
- Data: about 150k synthetic statement-shaped lines plus at least 20k real crops, taken from non-eval files only. Synthetic lines use TLWG, Sarabun, Noto Sans Thai and Kanit fonts; the generators cover amounts with separators, Thai and Gregorian dates, Thai month names, and channel and description phrases; degradations reuse `ocr_bench/data/degrade.py`. Real crops are text-layer lines (after Spike B) and 2-of-2 pseudo-labels (`paddle_crop` and typhoon crop reads agreeing exactly), capped at 50%.
- Training: a low learning rate (1e-5 to 3e-5) with AdamW and cosine decay, in bf16 on the H100. Held out: 5% of the synthetic data, plus real lines from files used neither for training nor for eval.
- The fine-tuned checkpoint must pass the same gate.

**Gate (spec line-recognition):** clean CER ≤ 2%, degraded CER ≤ 6%, and no worse than `paddle_crop` on the same lines. The relative condition is the one that matters, because otherwise the TrOCR branch adds latency without gain.

### D8. Reconciliation by alignment voting
1. For each segment, find its span in the block text: fuzzy-locate the reader text in the block text with `rapidfuzz` partial alignment, in reading order, and consume matched spans.
2. Align the reader text and the block span with Levenshtein editops, and group them into agreed and disputed spans.
3. For each disputed span, apply these tie-breakers in order:
   - the field-type validator, when the segment belongs to a typed field or column (`amount`, `date`, `account`, via the mapper's column roles);
   - a third reading of that segment only: `paddle_crop` if the reader was TrOCR, TrOCR if a checkpoint exists, otherwise a crop read by the `verifier_reader` model. The 2-of-3 exact match wins;
   - otherwise keep the line-reader text, with `data-disputed` set and `data-alt` holding the other readings.
4. `data-conf` is the minimum over the segment's characters of each source's **isotonically calibrated** P(char correct). For muocr, the per-character input is its token probability spread over the token's characters (D7). Calibration is fitted per source on the gate's held-out lines, and a source with no calibration gets no `data-conf` for its own contribution. Raw logprobs are never compared across sources.

**HTML**
- A `<section data-page>` per page; `<h*>` for titles; `<p>` per text block, with `<span>` per segment; `<table>` with spans.
- Every element carries `data-bbox`, `data-block-id`, `data-src` and `data-conf`.
- Attributes are serialized in sorted order, so the output is byte-deterministic.

### D9. Verifier
The field lookup is two-step:
1. Rules first: label aliases from the registry and mapper, and the value found in the same or the next cell or span.
2. Then the `llm` model (Qwen3-8B by default), given only the candidate blocks and returning `{block_id, value}`, validated like any tool call.

A field goes to recovery if it is not found, fails a validator, or its `data-conf` is below `min_conf`.

**The recovery ladder**
- Regions, in order: region hint → nearest label block → whole page at 2× resolution.
- For each region: crop with a 10% margin and upscale to a height of at least 64 px per line, then run the detector.
- No text means `absent` for that region; move to the next region.
- Otherwise read the crop twice: the `verifier_reader` model with a focused prompt naming the field and its format, and the line reader on the detected lines.
- Accept the value only if both readings agree after `normalize_field` and the validators pass.
- Then patch by block id, keeping `data-prev`, and set `data-src="verify"`.
- Re-run the cross-field validators, which is Check B for statements. Revert the patch if they now fail.

**Budgets**
- `max_retries` 3 per field, and `max_vlm_calls` 12 per document.
- The report is regenerated from the HTML after each patch.
- The audit (crops, prompts, replies, decisions) goes to `verify.json` and `crops/`.

### D10. Extraction harness
The `llm` model (Qwen3-8B by default) via the vLLM OpenAI tool-calling API, temperature 0.

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
| Line reading | muocr 2–4 s for about 300 segments batched with beam 4 (to be measured in Spike A); `paddle_crop` 3–8 s |
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
- **[muocr was trained on forms, not statements]** → Its CER on statement lines, dense numeric columns and long descriptions is unknown until Spike A. The gate decides. If it fails, a conditional fine-tune starts from muocr, not from scratch.
- **[TrOCR never beats `paddle_crop`, even after fine-tuning]** → The reader interface makes that outcome cheap. `paddle_crop` stays the default, and the report says so.
- **[Trigram blocking corrupts numbers]** → Decoding sets `no_repeat_ngram_size: 0`, and Spike A checks numeric lines with both settings.
- **[Three vLLM servers on one H100 contend for memory and compute]** → Fixed memory fractions (D3). Qwen3-VL and Qwen3 are idle on the happy path. The measured throughput goes in the report.
- **[The verifier cannot reach `complete` on degraded photos]** → The field states make that visible as `unverified` or `absent`, never as a wrong value. The review queue can consume `unverified` fields later.
- **[An 8B LLM misroutes queries]** → Routing is scored on the Q/A set. Parameters are schema-validated, and a wrong plan fails verification rather than returning a wrong value.
- **[Pseudo-labels copy shared reader errors into the conditional fine-tune]** → An exact 2-of-2 agreement is required, and pseudo-labelled lines are capped at 50% of the real crops.

## Migration Plan

This is a new package, so nothing needs migrating. The new vLLM services are opt-in compose profiles (`docparse`). Rolling back means removing the profile. ocr_bench runs that do not select the `docparse` model are unaffected.

## Open Questions

- Settled in task 2.1: the checkpoints are Qwen3-VL-8B-Instruct and Qwen3-8B in bf16, with no quantization (D3). Whether the KV-cache headroom holds under concurrent verifier and extraction load is measured on the H100.
