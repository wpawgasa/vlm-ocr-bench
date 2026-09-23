# Tasks

The implementer tier follows the model-routing skill:
- **[Opus]:** hard or ambiguous work.
- **[Sonnet]:** fully specified, pattern-following work.
- **[Human]:** needs a person, or H100 access.

## 1. D0 Spikes and eval foundation

- [ ] 1.1 [Opus] Spike A, the line-reader bake-off on the H100. Use ThaiOCRBench Text recognition plus Fine-grained (≈400 lines) and the lines of the 27 text-layer pages. Compare `paddle_crop`, a typhoon crop read and dots.ocr grounding. Report CER per condition in `runs/docparse/spikes/line_readers.json`. Verify the file exists and its per-reader CER is quoted in design D7.
- [ ] 1.2 [Opus] Spike B, reclaiming text layers. Diagnose why the 103 digital pages fail `text_layer_usable` (`ocr_bench/data/bankstmt.py:101`). Fix causes that are mappable, such as PUA glyphs. Record which pages become usable. Verify with a unit test for each fixed cause and an updated count in a spike note.
- [ ] 1.3 [Sonnet] Write `configs/docparse/eval_files.v1.yaml`: every file with a usable text-layer page after 1.2, plus the anchor-set files. Add a loader that exposes the list version. Verify the loader test and that the list is committed as file ids only (no content).
- [ ] 1.4 [Human] Label the manual anchor set (ocr_bench task 5.8) before any docparse tuning. Verify that `ocrbench review load` accepts the labels and the anchor files appear in the eval list.

## 2. D1 Scaffold, registry and serving

- [ ] 2.1 [Sonnet] Add the vLLM services `qwen3vl` (Qwen3-VL-8B-Instruct) and `qwen3` (Qwen3-8B, tool calling), with `scripts/serve_{qwen3vl,qwen3}.sh` and a `docparse` compose profile at the D3 memory fractions. Record the chosen checkpoints in design D3. Verify both health checks pass on the H100, with the Paddle pipeline running at the same time.
- [ ] 2.2 [Sonnet] Create the `docparse/` package skeleton, `docparse/schemas.py` (all stage artifacts from D2), `configs/docparse/pipeline.yaml`, `tests/docparse/`, and the dependencies in `pyproject.toml`. Verify `uv sync --all-extras` and JSON round-trip tests for every schema.
- [ ] 2.3 [Sonnet] Implement `docparse/artifacts.py`: the doc id (a hash of the file content), per-stage config and input hashes, skip-if-unchanged, and atomic writes. Verify tests showing that a second run skips the stage and that a config change re-runs only that stage and the stages downstream of it.
- [ ] 2.4 [Sonnet] Implement `docparse/registry.py` and `configs/docparse/classes/bank_statement.yaml`. The field schema, validators and `required_when` come from the `document-registry` spec. Aliases are imported from `ocr_bench/normalize/statement.py`, and bank rules reuse `BankOverrides`. Verify the spec scenarios (valid load, bad type, new class, KBank opening balance, shared alias).

## 3. D2 Parse skeleton (no TrOCR yet)

- [ ] 3.1 [Sonnet] Implement `docparse/classify.py` per design D4. Verify the spec scenarios with recorded Qwen3-VL replies (statement, `unknown`, prose reply → `invalid_reply`, bank detection).
- [ ] 3.2 [Sonnet] Implement `docparse/layout.py` with the Paddle and dots backends, reusing the `paddle_layout` and `dots_layout_json` parsers, and add the `truncated` flag and page-level errors. Verify recorded-response tests for both backends, a `finish_reason=length` page and a failed page (→ `partial`).
- [ ] 3.3 [Opus] Implement `docparse/segment.py` per design D6: the DB detector, table cell assignment from table HTML, aspect-ratio splitting and the projection fallback. Verify on synthetic rendered fixtures: the paragraph, table, 1500×30 line and faint-print scenarios.
- [ ] 3.4 [Sonnet] Implement `docparse/recognize/base.py` (the `LineReader` protocol) and `paddle_crop.py`, batched, with per-segment errors. Verify the spec scenarios with recorded pipeline replies.
- [ ] 3.5 [Opus] Implement `docparse/reconcile.py` per design D8: span location, editops voting, validator tie-break, third-reader call and `data-alt`/`data-disputed`. Add `docparse/html.py`, which renders deterministic HTML. Verify every `parse-reconciliation` scenario, plus a byte-identical HTML test.
- [ ] 3.6 [Sonnet] Wire `docparse parse`, which runs classify → layout → segment → recognize → reconcile with `status.json`, and exits non-zero on an error input. Verify an end-to-end CLI test with a fake client and the unreadable-PDF batch scenario.
- [ ] 3.7 [Opus] Add the `ocr_bench` adapter `docparse` (plan kind `docparse`), which converts `parsed.html` to `NormalizedPage`, and run it on the H100 on run `2026-09-22-a`'s eval samples. Verify that `ocrbench score` shows docparse rows next to the baselines. Record the first docparse-vs-VLM numbers.

## 4. D3 Thai TrOCR

- [ ] 4.1 [Opus] Implement `docparse/trocr/data.py`: the synthetic renderer (fonts, corpus and statement generators, degradations from `ocr_bench/data/degrade.py`), the real-crop builder (text-layer lines and 2-of-2 pseudo-labels capped at 50%), the vocabulary and its out-of-vocabulary rejection, the eval-leakage refusal and the manifest hash. Verify the `trocr-thai-training` scenarios, including the leakage refusal and the deterministic rebuild.
- [ ] 4.2 [Opus] Implement `docparse/trocr/train.py` per design D7: the `trocr-base-printed` encoder at 96×768 with interpolated position embeddings, the character-level decoder, and two-stage training. Verify a 200-step CPU smoke run that overfits 50 lines to CER < 5%, and that its outputs contain the checkpoint, vocabulary, config and manifest hash.
- [ ] 4.3 [Human] Run the full TrOCR training on the H100 and push the checkpoint to the DVC GCS remote. Verify held-out CER per condition in the run output and that the `dvc push` succeeded.
- [ ] 4.4 [Sonnet] Implement `docparse/recognize/trocr.py` (batched, per-character confidences) and `docparse gate`, which writes `gate.json`. Also fit the per-source isotonic calibration used by `data-conf`. Verify the gate scenario (absolute pass, relative fail → default stays `paddle_crop`) and the calibration fit on fixtures.
- [ ] 4.5 [Human] Run `docparse gate` on the H100. If TrOCR passes, set `line_reader: trocr` in `pipeline.yaml`. Verify that `gate.json` is committed as numbers only.

## 5. D4 Verifier

- [ ] 5.1 [Opus] Implement the field lookup in `docparse/verify.py` per design D9 (rules, then Qwen3-8B on the candidate blocks), with the field report derived from the HTML. Verify the complete-statement scenario and the `not_required` field state.
- [ ] 5.2 [Opus] Implement the recovery loop: the region ladder, the detector-based `absent` check, the two-reader agreement guard, validator gating, patching by block id with `data-prev`, cross-field re-validation with revert, both budgets, and the audit trail. Verify every `field-verification` scenario with recorded replies.
- [ ] 5.3 [Sonnet] Wire the verify stage into `docparse parse` and `status.json`, so a document is `complete` or `partial` depending on its field states. Verify re-running verify alone without calling the upstream stages again.

## 6. D5 Extraction

- [ ] 6.1 [Opus] Implement `docparse/extract/tools.py`: the typed parse view (DOM, `fields.json` and `StatementRow`s with row and block ids) and all seven tools, with a `Decimal` AST in `compute`. Verify unit tests for each tool, including invalid arguments returning an error result.
- [ ] 6.2 [Opus] Implement `docparse/extract/plans.py` and `agent.py`: query routing with parameter schemas, the templates for the four query types, and a bounded free-form ReAct loop with the flag. Verify recorded-reply tests for each query type and the unsupported-phrasing scenario.
- [ ] 6.3 [Opus] Implement `docparse/extract/verify.py`: read-value matching, recomputation of computed values, step retries and `absent` propagation. Verify the hallucinated-citation and unanswerable scenarios.
- [ ] 6.4 [Sonnet] Wire `docparse extract` to write the answer JSON with its plan and trace. Verify a CLI test on a fixture parse.

## 7. D6 Service

- [ ] 7.1 [Sonnet] Implement the FastAPI app in `docparse/service/api.py` (`POST /parse`, `POST /extract`, `GET /documents/{id}`), with type and size checks, reusing the CLI pipeline. Verify the upload-and-parse and unsupported-upload scenarios with `TestClient` and fake clients.
- [ ] 7.2 [Sonnet] Add a `docparse-api` compose service and a README section covering serving, parsing and extraction. Verify that `curl` on one statement returns a document id and HTML on the H100.

## 8. D7 Evaluation and report

- [ ] 8.1 [Sonnet] Implement the statement Q/A generator from the eval ground truth, covering every query type with at least 10% unanswerable items. Verify coverage and the unanswerable fraction in a unit test.
- [ ] 8.2 [Opus] Implement `docparse eval compare`: paired cluster-bootstrap CIs of docparse − best VLM per condition and bank on field F1 and CER, the decision rule, the Check B row-consistency rate, and the Q/A scores (exact match, provenance, abstention precision and recall). Verify the inconclusive-result scenario and a fixture where docparse is clearly better.
- [ ] 8.3 [Human] Run the full evaluation on the H100 and write `docs/docparse-evaluation.md` with the results, latency per stage (D11) and the TrOCR gate outcome. Verify that the report quotes the eval list version and every number traces to a committed or DVC-tracked artifact.
