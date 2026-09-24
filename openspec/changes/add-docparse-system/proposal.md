# Proposal

## Why

The first H100 run (`2026-09-22-a`) showed that no single Thai-capable VLM parses bank statements reliably:
- dots.ocr hits `max_tokens` on 9.9% of rows, stuck in Thai repetition loops.
- typhoon-ocr1.5 and PaddleOCR-VL do not follow instructions: 457 of 1,149 KIE replies are not JSON.
- None of them gives a trustworthy per-field confidence.

The client needs complete, verifiable fields, not a best-effort page transcript. This change builds `docparse`, a system that orchestrates several models:
- A parsing pipeline produces a verified HTML version of the document.
- An extraction pipeline answers queries over that HTML with a small LLM, and every answer carries provenance and is verified.

The `ocr_bench` harness is already in place, so docparse can be scored against the single-VLM baselines on the same run.

## What Changes

- Add a `docparse` Python package and CLI in this repo. It reuses `ocr_bench`'s vLLM and pipeline clients, layout parsers, Thai normalizers, statement mapper and Check B arithmetic.
- **Document registry**: YAML document classes with required fields, label aliases, region hints, validators and per-bank `required_when` rules. v1 fully specifies `bank_statement` only; other classes can be added through config.
- **Model roles**: every model docparse calls over HTTP is bound by role (`layout`, `classifier`, `verifier_reader`, `llm`) to an entry of the `ocr_bench` model registry (`configs/models/*.yaml`). A role's model is switched in configuration. The defaults are PaddleOCR-VL, Qwen3-VL-8B and Qwen3-8B.
- **Parsing pipeline**, where each stage writes a typed artifact:
  1. **Classify**: an instruction-following VLM assigns a registry class.
  2. **Layout**: layout blocks with bbox, category, reading order and a first text reading. The PaddleOCR-VL pipeline is the default and dots.ocr `prompt_layout_all_en` the alternative.
  3. **Segment**: blocks are split into line segments, with a maximum aspect ratio.
  4. **Recognize**: a pluggable line reader reads each segment. `paddle_crop` is the baseline and the existing Thai TrOCR checkpoint (muocr) is the target backend.
  5. **Reconcile**: character-alignment voting merges the readings into HTML that carries bbox, page, confidence, source and dispute attributes.
  6. **Verify**: a bounded verifier agent checks the required fields of the class. It crops and re-reads missing or invalid fields, and accepts a recovery only when two readers agree and the validator passes.
- **Thai TrOCR (muocr)**:
  - The model: the existing fine-tuned checkpoint `muocr-base-26m-stage2-finetuned-20240820-v1` (a ViT encoder at 64×384 with a Thai+English SentencePiece decoder). It is used as-is, and nothing is trained from scratch.
  - Its gate: absolute CER targets **and** no worse than `paddle_crop` on the same held-out statement lines. It becomes the default reader only if it passes.
  - A stage-3 fine-tune from muocr on statement-shaped lines (synthetic plus real crops, never from eval files) runs **only if muocr fails the gate**, and must then pass the same gate.
- **Extraction pipeline**: a small LLM (Qwen3-8B) with tool calls over the parse (`outline`, `find`, `get_block`, `get_field`, `get_table`, `table_query`, `compute`).
  - Query types map to plan templates, and the LLM fills in their parameters. A free-form plan is only a flagged fallback.
  - Read values are verified by normalized match in the cited block. Computed values are verified by recomputation from the cited rows.
  - An unanswerable query returns `unverified` or `absent` instead of a guess.
- **Service**: `docparse parse`, `docparse extract`, and a FastAPI app (`POST /parse`, `POST /extract`) for uploaded documents.
- **Evaluation**:
  - A frozen, file-level eval set: the 27 text-layer pages plus the manual anchor set.
  - docparse is registered as an `ocr_bench` model.
  - A templated statement Q/A set that includes unanswerable queries.
  - A paired-bootstrap success criterion against the best single VLM, reported per condition and per bank.

Non-goals:
- Classes other than `bank_statement` with full field schemas.
- Fine-tuning the layout VLMs or the extraction LLM.
- Multi-tenant auth, queueing or horizontal scaling of the service.
- Handwriting.
- Replacing the `ocr_bench` benchmark.

## Capabilities

### New Capabilities

- `document-registry`: document classes, required-field schemas, label aliases, region hints, validators and per-bank `required_when` rules.
- `document-classification`: assigning a registry class (or `unknown`) with a confidence to an uploaded document.
- `layout-analysis`: layout blocks with bbox, category, reading order and a first text reading, through pluggable layout backends.
- `line-segmentation`: splitting layout blocks into line segments that respect the reader's geometry limits.
- `line-recognition`: the line-reader contract, the `paddle_crop` and `trocr` backends, and the gate that promotes TrOCR to the default reader.
- `trocr-thai-training`: the conditional fine-tune of the TrOCR checkpoint (when it runs, its base checkpoint, training data, leakage rules and reproducible outputs).
- `parse-reconciliation`: merging readings by alignment voting and the parsed-HTML output contract.
- `field-verification`: the verifier agent loop, field states, recovery guards, budgets and the audit trail.
- `query-extraction`: the extraction agent harness, its tools, query-type plans, provenance, verification and abstention.
- `docparse-service`: the CLI, the HTTP API, the binding of model roles to the model registry, the per-document artifact layout and error semantics.
- `docparse-evaluation`: the frozen eval set, docparse as an `ocr_bench` model, the statement Q/A set and the success criterion.

### Modified Capabilities

(none: `openspec/specs/` is empty, and the `ocr_bench` capabilities are still in the unarchived change `add-ocr-benchmark-harness`. docparse depends on them and does not change their requirements.)

## Impact

- **New code**:
  - `docparse/`: registry, classify, layout, segment, recognize, reconcile, verify, extract, trocr, service.
  - `configs/docparse/`: pipeline config and document classes.
  - `tests/docparse/`.
  - `scripts/serve_{qwen3vl,qwen3}.sh`.
- **ocr_bench**:
  - a `docparse` adapter under `ocr_bench/models/`, so the harness can score docparse output;
  - a backward-compatible `ModelConfig` extension (an optional `roles` section, optional `plans`, `chat_template_kwargs` in `Sampling`) and a text-only/tool chat call in `VllmClient`;
  - registry entries `configs/models/qwen3vl.yaml` and `qwen3.yaml`.
  Every existing model YAML still loads, and no harness requirement changes.
- **Serving**: two new vLLM services, Qwen3-VL-8B-Instruct (classify, verifier crop reads) and Qwen3-8B (field lookup, extraction), alongside the PaddleOCR-VL pipeline. They all share one H100 at reduced `gpu-memory-utilization`. TrOCR and the PP-OCRv5 text detector run in-process.
- **Dependencies**:
  - `transformers`, `torch`, `paddleocr` (detector only), `fastapi`, `uvicorn`.
  - Conditional fine-tune only: `trdg`-style rendering with the Thai fonts already in the image, and `datasets`.
  - Optional: `onnxruntime` for the checkpoint's ONNX export (a CPU path for the service).
- **Data**: bank statements stay client data under gitignored `data/` and `runs/`. TrOCR checkpoints (muocr and any fine-tune) live under the gitignored `models/` and on the DVC GCS remote, and are never committed. The frozen eval file list is committed as ids only.
- **Compute**: H100 for layout, VLM and LLM serving, evaluation, and the conditional muocr fine-tune (about 200k lines, only if gated out). The V100 dev box handles unit tests and CPU-only stages.
