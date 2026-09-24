# Tasks

The implementer tier follows the model-routing skill:
- **[Opus]:** hard or ambiguous work.
- **[Sonnet]:** fully specified, pattern-following work.
- **[Human]:** needs a person, or H100 access.

## 1. D0 Spikes and eval foundation

- [ ] 1.1 [Opus] Spike A, the line-reader bake-off on the H100. Use ThaiOCRBench Text recognition plus Fine-grained (≈400 lines) and the lines of the 27 text-layer pages. Compare `paddle_crop`, a typhoon crop read, dots.ocr grounding and the muocr checkpoint. Run muocr with `no_repeat_ngram_size` 3 and 0, and report numeric lines separately. Report CER per condition and the muocr segment throughput in `runs/docparse/spikes/line_readers.json`. Verify the file exists and its per-reader CER is quoted in design D7.
- [x] 1.2 [Opus] Spike B, reclaiming text layers. Diagnose why the 103 digital pages fail `text_layer_usable` (`ocr_bench/data/bankstmt.py:101`). Fix causes that are mappable, such as PUA glyphs. Record which pages become usable. Verify with a unit test for each fixed cause and an updated count in a spike note.
- [ ] 1.3 [Sonnet] Write `configs/docparse/eval_files.v1.yaml`: every file with a usable text-layer page after 1.2, plus the anchor-set files. Add a loader that exposes the list version. Verify the loader test and that the list is committed as file ids only (no content).
- [ ] 1.4 [Human] Label the manual anchor set (ocr_bench task 5.8) before any docparse tuning. Verify that `ocrbench review load` accepts the labels and the anchor files appear in the eval list.

## 2. D1 Scaffold, registry and serving

- [ ] 2.1 [Sonnet] Add the vLLM services `qwen3vl` (Qwen3-VL-8B-Instruct) and `qwen3` (Qwen3-8B, tool calling), with `scripts/serve_{qwen3vl,qwen3}.sh` and a `docparse` compose profile at the D3 memory fractions. Record the chosen checkpoints in design D3. Verify both health checks pass on the H100, with the Paddle pipeline running at the same time.
- [x] 2.2 [Sonnet] Create the `docparse/` package skeleton, `docparse/schemas.py` (all stage artifacts from D2), `configs/docparse/pipeline.yaml`, `tests/docparse/`, and the dependencies in `pyproject.toml`. Verify `uv sync --all-extras` and JSON round-trip tests for every schema.
- [x] 2.3 [Sonnet] Implement `docparse/artifacts.py`: the doc id (a hash of the file content), per-stage config and input hashes, skip-if-unchanged, and atomic writes. Verify tests showing that a second run skips the stage and that a config change re-runs only that stage and the stages downstream of it.
- [ ] 2.4 [Sonnet] Implement `docparse/registry.py` and `configs/docparse/classes/bank_statement.yaml`. The field schema, validators and `required_when` come from the `document-registry` spec. Aliases are imported from `ocr_bench/normalize/statement.py`, and bank rules reuse `BankOverrides`. Verify the spec scenarios (valid load, bad type, new class, KBank opening balance, shared alias).
- [ ] 2.5 [Sonnet] Bind model roles to the model registry per design D3. Extend `ModelConfig` with an optional `roles` section and optional `plans` (the harness refuses an entry without plans), add `chat_template_kwargs` to `Sampling`, and add a text-only/tool chat call to `VllmClient`. Write the registry entries `configs/models/qwen3vl.yaml` and `qwen3.yaml` (roles only, no benchmark plans) for the services from 2.1. Add the `roles` loader in `docparse/`, with startup validation and the model identity recorded in artifacts. Verify both "Model roles bound to the model registry" scenarios, that every existing `configs/models/*.yaml` still loads, and that `ocrbench infer` rejects a roles-only entry.

## 3. D2 Parse skeleton (no TrOCR yet)

- [ ] 3.1 [Sonnet] Implement `docparse/classify.py` per design D4. Verify the spec scenarios with recorded replies from the `classifier` model (Qwen3-VL) (statement, `unknown`, prose reply → `invalid_reply`, bank detection).
- [ ] 3.2 [Sonnet] Implement `docparse/layout.py` with the `paddleocr_vl` and `dotsocr` layout adapters, selected by `roles.layout`, reusing the `paddle_layout` and `dots_layout_json` parsers, and add the `truncated` flag and page-level errors. Verify recorded-response tests for both backends, a `finish_reason=length` page and a failed page (→ `partial`).
- [ ] 3.3 [Opus] Implement `docparse/segment.py` per design D6, adding `paddleocr` (detector only) to the dependencies: the DB detector, table cell assignment from table HTML, aspect-ratio splitting and the projection fallback. Verify on synthetic rendered fixtures: the paragraph, table, 1500×30 line and faint-print scenarios.
- [ ] 3.4 [Sonnet] Implement `docparse/recognize/base.py` (the `LineReader` protocol) and `paddle_crop.py`, batched, with per-segment errors. Verify the spec scenarios with recorded pipeline replies.
- [ ] 3.5 [Opus] Implement `docparse/reconcile.py` per design D8: span location, editops voting, validator tie-break, third-reader call and `data-alt`/`data-disputed`. Add `docparse/html.py`, which renders deterministic HTML. Verify every `parse-reconciliation` scenario, plus a byte-identical HTML test.
- [ ] 3.6 [Sonnet] Wire `docparse parse`, which runs classify → layout → segment → recognize → reconcile with `status.json`, and exits non-zero on an error input. Verify an end-to-end CLI test with a fake client and the unreadable-PDF batch scenario.
- [ ] 3.7 [Opus] Add the `ocr_bench` adapter `docparse` (plan kind `docparse`), which converts `parsed.html` to `NormalizedPage`, and run it on the H100 on run `2026-09-22-a`'s eval samples. Verify that `ocrbench score` shows docparse rows next to the baselines. Record the first docparse-vs-VLM numbers.

## 4. D3 Thai TrOCR (muocr)

- [ ] 4.1 [Human] Track `models/muocr-base-26m-stage2-finetuned-20240820-v1` with DVC, push it to the GCS remote and record its content hash as the checkpoint version in `configs/docparse/pipeline.yaml`. Verify that `dvc push` succeeded, that `models/` stays gitignored, and that a fresh clone can `dvc pull` the checkpoint.
- [ ] 4.2 [Sonnet] Implement `docparse/recognize/trocr.py` for the muocr checkpoint per design D7, adding `torch` and `transformers` to the dependencies: a 64×384 resize that keeps aspect ratio and pads right, batched beam-4 decoding with `no_repeat_ngram_size: 0`, per-character confidences spread from token probabilities, a version string made of the checkpoint hash plus the decoding settings, and an optional ONNX path. Verify on a tiny random-weight model with the same config (a CPU test), the `1,000,000.00` scenario with a stubbed decoder, and that ONNX and PyTorch outputs match on 20 fixture crops when the checkpoint is present (marked `live`).
- [ ] 4.3 [Sonnet] Implement `docparse gate`, which writes `gate.json`: per-reader CER by condition and for numeric lines, and the pass or fail of the absolute and relative conditions. Also fit the per-source isotonic calibration used by `data-conf`. Verify the gate scenario (absolute pass, relative fail → default stays `paddle_crop`) and the calibration fit on fixtures.
- [ ] 4.4 [Human] Run `docparse gate` for muocr on the H100. If it passes, set `line_reader: trocr` in `pipeline.yaml`, and tasks 4.5–4.6 are not needed. Verify that `gate.json` is committed as numbers only.
- [ ] 4.5 [Opus] **Only if 4.4 fails.** Implement the fine-tune per design D7 and the `trocr-thai-training` spec: `docparse/trocr/data.py` (synthetic statement-shaped renderer, real-crop builder with capped 2-of-2 pseudo-labels, tokenizer-unknown rejection, eval-leakage refusal, manifest hash) and `docparse/trocr/train.py` (initialize from muocr, keep its tokenizer and 64×384 input, low-learning-rate AdamW, bf16). The run is refused unless a failed `gate.json` is given. Verify every `trocr-thai-training` scenario, plus a 200-step CPU smoke run on a tiny config that overfits 50 lines.
- [ ] 4.6 [Human] **Only if 4.4 fails.** Run the fine-tune on the H100, push the checkpoint to DVC, and re-run `docparse gate` on it. Verify held-out CER per condition in the run output, and a new `gate.json` for the fine-tuned checkpoint.

## 5. D4 Verifier

- [ ] 5.1 [Opus] Implement the field lookup in `docparse/verify.py` per design D9 (rules, then the `llm` model on the candidate blocks), with the field report derived from the HTML. Verify the complete-statement scenario and the `not_required` field state.
- [ ] 5.2 [Opus] Implement the recovery loop: the region ladder, the detector-based `absent` check, the two-reader agreement guard (the `verifier_reader` model and the line reader), validator gating, patching by block id with `data-prev`, cross-field re-validation with revert, both budgets, and the audit trail. Verify every `field-verification` scenario with recorded replies.
- [ ] 5.3 [Sonnet] Wire the verify stage into `docparse parse` and `status.json`, so a document is `complete` or `partial` depending on its field states. Verify re-running verify alone without calling the upstream stages again.

## 6. D5 Extraction

- [ ] 6.1 [Opus] Implement `docparse/extract/tools.py`: the typed parse view (DOM, `fields.json` and `StatementRow`s with row and block ids) and all seven tools, with a `Decimal` AST in `compute`. Verify unit tests for each tool, including invalid arguments returning an error result.
- [ ] 6.2 [Opus] Implement `docparse/extract/plans.py` and `agent.py`: query routing through the `llm` model with parameter schemas, the templates for the four query types, and a bounded free-form ReAct loop with the flag. Verify recorded-reply tests for each query type and the unsupported-phrasing scenario.
- [ ] 6.3 [Opus] Implement `docparse/extract/verify.py`: read-value matching, recomputation of computed values, step retries and `absent` propagation. Verify the hallucinated-citation and unanswerable scenarios.
- [ ] 6.4 [Sonnet] Wire `docparse extract` to write the answer JSON with its plan and trace. Verify a CLI test on a fixture parse.

## 7. D6 Service

- [ ] 7.1 [Sonnet] Implement the FastAPI app in `docparse/service/api.py` (`POST /parse`, `POST /extract`, `GET /documents/{id}`), with type and size checks, reusing the CLI pipeline. Verify the upload-and-parse and unsupported-upload scenarios with `TestClient` and fake clients.
- [ ] 7.2 [Sonnet] Add a `docparse-api` compose service and a README section covering serving, parsing and extraction. Verify that `curl` on one statement returns a document id and HTML on the H100.

## 8. D7 Evaluation and report

- [ ] 8.1 [Sonnet] Implement the statement Q/A generator from the eval ground truth, covering every query type with at least 10% unanswerable items. Verify coverage and the unanswerable fraction in a unit test.
- [ ] 8.2 [Opus] Implement `docparse eval compare`: paired cluster-bootstrap CIs of docparse − best VLM per condition and bank on field F1 and CER, the decision rule, the Check B row-consistency rate, and the Q/A scores (exact match, provenance, abstention precision and recall). Verify the inconclusive-result scenario and a fixture where docparse is clearly better.
- [ ] 8.3 [Human] Run the full evaluation on the H100 and write `docs/docparse-evaluation.md` with the results, latency per stage (D11) and the TrOCR gate outcome (muocr, and the fine-tune if one was needed). Verify that the report quotes the eval list version and every number traces to a committed or DVC-tracked artifact.
