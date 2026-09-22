# OCR Benchmark Spec — TeleOCR vs dots.ocr for บสย. / LooLoo questions

2026-09-22 · @Someone

## 1. Purpose & scope

This repo (`ocr-bench`) produces one scorecard that answers client questions Q1, Q2, Q3, Q4, Q5 and Q8 (sheet 01) and sections 1–3 of sheet 02 for TeleOCR and dots.ocr, without any labeled บสย. documents. Q6 (model governance), Q7 (security) and pricing are written answers outside this repo.

Two data sources play two roles: ThaiOCRBench (typhoon-ai) is the *capability proxy* with human labels; the unlabeled bank statements are the *domain proxy*, scored with label-free checks (text-layer ground truth, arithmetic consistency, cross-model agreement). Results are reported separately and never pooled.

| Client question | What the repo produces | Source |
| --- | --- | --- |
| Q1 Scope (Thai, handwriting, multi-page tables, bank statements) | Per-task scores + supported/unsupported matrix | ThaiOCRBench + statements |
| Q2 Field-level accuracy (P/R, exact-match, FA/FR, by doc type / image quality / critical field) | CER, WER, TED, field F1, exact-match, stratified | ThaiOCRBench KIE/OCR/Table + statements |
| Q3 Confidence vs accuracy, thresholds, review rate | Calibration table, auto-accept/review/reject thresholds, review rate at 99% auto-accept accuracy | Same, with confidence proxy |
| Q4 Output structure (page, bbox, normalization, error flags) | Capability matrix + normalized JSON output contract | Statements |
| Q5 Duplicates, reconciliation, false positives | Running-balance check pass rate, duplicate detection recall/FPR | Statements |
| Q8 Throughput / p95 | p50/p95 sec per page, pages per hour per GPU, GPU-hours per 1,000 pages | Any subset, H100 and L4 |

Non-goals: fine-tuning either model, building the LooLoo production pipeline, evaluating VQA/chart/infographic tasks.

## 2. Repo layout

Python 3.11, `uv` for deps, `typer` CLI, `pydantic` schemas, `pytest`. Every stage reads and writes JSONL under `runs/<run_id>/` so any stage can be re-run alone.

```markdown
ocr-bench/
├── pyproject.toml
├── configs/
│   ├── models/{teleocr,dotsocr,typhoon_ocr15}.yaml   # vLLM args, prompts, output parser
│   ├── datasets/{thaiocrbench,bankstmt}.yaml         # subset filters, degradation profiles
│   └── run.yaml                                      # which models × datasets × conditions
├── ocr_bench/
│   ├── cli.py              # `ocrbench prepare | infer | score | calibrate | bench-latency | report`
│   ├── schemas.py          # Sample, Prediction, FieldResult, LatencyRecord (pydantic)
│   ├── data/
│   │   ├── thaiocrbench.py # HF loader, task/domain filter, manifest writer
│   │   ├── bankstmt.py     # PDF/scan ingestion, text-layer GT, page rasterization
│   │   ├── degrade.py      # clean → degraded image conditions
│   │   └── manifest.py     # read/write manifest.jsonl
│   ├── models/
│   │   ├── base.py         # OcrModel interface
│   │   ├── vllm_client.py  # OpenAI-compatible client, logprobs, retries, timing
│   │   ├── teleocr.py
│   │   ├── dotsocr.py
│   │   └── parsers.py      # markdown/HTML/JSON → NormalizedPage
│   ├── normalize/
│   │   ├── text.py         # Thai NFC, digit/whitespace canonicalization
│   │   ├── fields.py       # date, amount, account number, name rules
│   │   └── statement.py    # rows → StatementTable
│   ├── metrics/
│   │   ├── cer_wer.py      # BMFL-compatible text metrics (Thai-aware)
│   │   ├── ted.py          # tree edit distance for tables/documents
│   │   ├── kie_f1.py       # field P/R/F1, exact-match, FA/FR
│   │   ├── anls.py         # document classification
│   │   ├── arithmetic.py   # running-balance and total checks
│   │   └── duplicates.py   # near-duplicate detection
│   ├── confidence/
│   │   ├── proxy.py        # logprob + agreement + arithmetic → score
│   │   └── calibrate.py    # bands, thresholds, review rate
│   ├── latency/bench.py    # batch sweeps, p50/p95, pages/hour
│   └── report/
│       ├── scorecard.py    # question → metric table
│       └── render.py       # markdown + xlsx + plots
├── scripts/serve_{teleocr,dotsocr}.sh
├── tests/
└── runs/<run_id>/{manifest,predictions,scores,calibration,latency}.jsonl + report/
```

Module responsibilities are one-directional: `data → models → normalize → metrics/confidence → report`. `normalize` is the only place that touches raw model text; metrics operate on `NormalizedPage` only.

## 3. Data

### 3.1 ThaiOCRBench subset

Load `typhoon-ai/ThaiOCRBench` from Hugging Face and keep 9 of the 13 tasks; drop Chart parsing, Diagram VQA, Cognition VQA and Infographics VQA. Verify the exact task name strings against the dataset's metadata on first load and fail loudly if a name is missing.

| Task | Metric | Client mapping | Cap per task |
| --- | --- | --- | --- |
| Full-page OCR | BMFL (CER/WER) | Q1/Q2 Thai text | 200 |
| Text recognition | BMFL | Q2 | 200 |
| Fine-grained text recognition | BMFL | Q2 small print | 200 |
| Handwritten content extraction | BMFL | Q1 handwriting | all |
| Table parsing | TED | Q1 multi-page tables | all |
| Document parsing | TED | Q1/Q4 structure | 200 |
| Key information extraction | F1 | Q2 field-level | all |
| Key information mapping | F1 | Q2 field-level | all |
| Document classification | ANLS | Q1 routing | all |

Caps are random with fixed seed 42, stratified by domain. If the dataset carries a domain field, tag Government and Finance samples and report that slice separately. Expected total: 1,200–1,500 samples.

### 3.2 Bank statements

Input: a directory of PDFs and images from Thai banks (KBank, SCB, BBL, KTB, Krungsri, TTB, GSB — record the bank per file in the manifest). Ingestion:

1. `pdffonts` on each PDF. Fonts present → `digital`; none → `scanned`. Images → `photo`.
2. Rasterize every page at 200 dpi PNG (`pdftoppm`). One sample per page.
3. For `digital` pages, extract the text layer with `pdftotext -layout` and per-word boxes with `pdfplumber`; store as `gt_text` and `gt_words`. These pages are the free ground truth.
4. Redact nothing at ingestion; anonymization happens only in `report` for the client-facing samples (Q4 asks for anonymized examples).

Target mix: at least 40 digital pages and 40 scanned/photo pages across at least 4 banks; at least 10 multi-page statements for table continuity.

### 3.3 Degradation conditions

Every sample is evaluated under three conditions so Q2's image-quality breakdown has three rows.

| Condition | Transform |
| --- | --- |
| `clean` | original |
| `scan_low` | resample to 150 dpi equivalent, JPEG quality 40, gaussian blur σ=0.6 |
| `photo` | random rotation ±3°, perspective warp ≤2%, brightness ±15%, JPEG quality 55 |

Transforms are deterministic per `(sample_id, condition)` via seeded RNG and are applied to ThaiOCRBench and statement images alike. Ground truth is unchanged; bounding boxes for `photo` are transformed with the same homography.

### 3.4 Manifest schema

One `manifest.jsonl` row per `(sample, condition)`:

```csv
sample_id,source,task,domain,bank,doc_type,page_no,n_pages,condition,image_path,gt_kind,gt_path,critical_fields
TOB-kie-00017,thaiocrbench,kie,Government,,form,1,1,clean,img/TOB-kie-00017.png,json,gt/TOB-kie-00017.json,"id_number;total_amount"
BS-kbank-0003-p2,bankstmt,statement,Finance,kbank,digital,2,4,scan_low,img/BS-kbank-0003-p2_scan_low.png,text_layer,gt/BS-kbank-0003-p2.json,"closing_balance;account_no"
```

`gt_kind` ∈ {`json`, `text`, `html`, `text_layer`, `none`}. `critical_fields` is the list of field names in this sample that count as critical (IDs, amounts, dates, names, account numbers); empty for non-KIE tasks.

## 4. Model adapters

Both models are served with vLLM behind an OpenAI-compatible endpoint; the harness never loads weights in-process. TeleOCR needs `--trust-remote-code`. A third adapter for `typhoon-ocr1.5-2b` is optional and gives a Thai-native reference with published ThaiOCRBench numbers.

### 4.1 Interface

```markdown
class OcrModel(Protocol):
    name: str
    def prompt_for(self, task: Task) -> str            # task-specific prompt from configs/models/*.yaml
    def infer(self, image: Path, task: Task) -> RawPrediction
    def parse(self, raw: RawPrediction, task: Task) -> NormalizedPage

class RawPrediction(BaseModel):
    text: str                      # model output verbatim
    token_logprobs: list[float]    # from vLLM logprobs=1, may be empty
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None

class NormalizedPage(BaseModel):
    text: str                      # reading-order plain text, NFC
    blocks: list[Block]            # type ∈ {text, table, figure, header, footer}, bbox: [x0,y0,x1,y1] | None, page: int
    tables: list[list[list[str]]]  # rows × cells, empty if none
    fields: dict[str, FieldValue]  # KIE only; FieldValue = {value, raw, confidence, bbox|None}
    label: str | None              # document classification only
```

### 4.2 Tasks and prompts

`Task` ∈ {`ocr_fullpage`, `ocr_line`, `handwriting`, `table`, `docparse`, `kie`, `kie_map`, `classify`, `statement`}. Prompts live in the model YAML, one per task; for KIE the prompt includes the field list from the sample and asks for JSON with `null` for absent fields. Every prompt is version-tagged and the tag is written into each prediction row so Q6-style change logs come for free.

### 4.3 Serving and client rules

- vLLM: `--max-model-len 16384`, `--limit-mm-per-prompt image=1`, `--gpu-memory-utilization 0.9`; batch size is the client's concurrency (8 for accuracy runs, swept in section 8).
- Request `logprobs=1`, `temperature=0`, `max_tokens=4096`; `seed=0`.
- Timeout 120 s per page; one retry; on second failure write `error` and count the page as a full miss in every metric (no silent drops).
- Output written to `predictions.jsonl`: `sample_id, condition, model, prompt_version, raw, normalized, latency_ms, tokens`.

### 4.4 Capability probe (Q4)

`ocrbench probe` runs 5 fixed statement pages per model and fills this matrix from the parsed output, then writes it to the report.

| Capability | TeleOCR | dots.ocr |
| --- | --- | --- |
| Reading-order text |  |  |
| Table as rows × cells |  |  |
| Block bounding boxes |  |  |
| Page index in output |  |  |
| JSON field extraction on request |  |  |
| Token logprobs available |  |  |

Cells are filled by the probe (`yes` / `partial` / `no`); a `no` on bbox means Q4 is answered with the other model or with a layout stage.

## 5. Metrics

All scorers take `(NormalizedPage pred, GroundTruth gt, Sample meta)` and return one `ScoreRow` per sample plus, for KIE, one `FieldResult` per field. Aggregation is a separate step so any slice (task, condition, domain, bank, critical flag) is a groupby, not a re-run.

### 5.1 Normalization before scoring

Applied identically to prediction and ground truth: Unicode NFC; Thai digits → Arabic digits; collapse whitespace; strip zero-width characters; `฿`/`บาท` removed from amounts; dates parsed to ISO with Buddhist-year → Gregorian conversion (year > 2400 → minus 543). Raw strings are kept beside normalized ones.

### 5.2 Scorers

| Scorer | Definition | Used for |
| --- | --- | --- |
| `cer` | Levenshtein at character level over NFC codepoints ÷ gt length | OCR tasks, statements (text\_layer) |
| `wer` | Levenshtein over PyThaiNLP `newmm` tokens ÷ gt tokens | OCR tasks |
| `bmfl` | ThaiOCRBench's official text score, reimplemented from the benchmark's scorer; unit-tested against 20 published samples | OCR tasks, for comparability |
| `ted` | Tree edit distance on HTML table / document tree, normalized to \[0,1\] (1 = identical) | Table parsing, Document parsing |
| `field_exact` | 1 if normalized pred == normalized gt else 0; `null` == `null` counts as correct | KIE, statements |
| `field_fuzzy` | 1 if CER ≤ 0.1 on the field string | KIE, reported beside exact |
| `kie_f1` | Micro P/R/F1 over (field, value) pairs; pred non-null on gt-null = false accept; pred null on gt non-null = false reject | KIE, Q2 FA/FR |
| `anls` | Standard ANLS, threshold 0.5 | Document classification |
| `row_f1` | Statement rows matched on (date, amount) with tolerance 0; F1 over rows | Statements |

### 5.3 Required slices in the output

`scores.jsonl` rows carry `model, task, condition, domain, bank, doc_type, is_critical` so `report` can produce these tables without re-scoring:

- by task × model (the headline table, comparable to published ThaiOCRBench rows)
- by condition × model (Q2 image quality)
- by critical vs non-critical field × model (Q2 critical fields)
- by bank × doc\_type × model (Q1 statement coverage)

Every aggregate reports n and a 95% bootstrap CI (1,000 resamples) so the client's sample-size question is answered on the same line as the number.

## 6. Bank statements — label-free evaluation

Three checks give statement scores without annotation; a 30–50 page manual set anchors them. The statement task extracts a fixed schema per page and merges pages per file.

### 6.1 Statement schema

```markdown
StatementPage:
  bank: str | None
  account_no: str | None          # digits only after normalize
  account_name: str | None
  period_start, period_end: date | None
  opening_balance, closing_balance: Decimal | None
  rows: list[Row]                 # Row = {date, description, debit, credit, balance, channel|None, bbox|None}
  page_no: int
StatementFile = merge(pages) with continuity check: last balance of page i == first balance basis of page i+1
```

### 6.2 Check A — text-layer ground truth (`digital` pages)

`pdftotext -layout` text is `gt_text`; `pdfplumber` words with boxes are `gt_words`. Score `cer` on the page text and `field_exact` on header fields after both sides pass the same normalizer. Rows are parsed from `gt_text` with the bank's known column order (`configs/datasets/bankstmt.yaml` has one regex layout per bank) and scored with `row_f1`. These pages are the statement numbers with a real label.

### 6.3 Check B — arithmetic self-consistency (all pages)

For each extracted file, evaluate:

```latex
b_i = b_{i-1} + c_i - d_i \quad \forall i,\qquad \sum_i c_i = C_{\text{summary}},\qquad \sum_i d_i = D_{\text{summary}},\qquad b_n = B_{\text{closing}}
```

Report per model: `row_consistency_rate` (share of rows where the balance equation holds), `file_reconciles` (all four hold), and `first_break_row` histogram. A broken row flags an error in one of three cells; this is also the mechanism behind sheet 02 items 1.2 and 2.1, and the `error_flags` column in the output contract (section 9).

### 6.4 Check C — cross-model agreement

For each field and row present in both models' outputs: `agree` if normalized values match. Agreement rate is reported as its own column and is one input to the confidence proxy. Disagreements, plus a random 10% of agreements, are exported to `review/queue.csv` for the manual set.

### 6.5 Manual anchor set

30–50 pages, stratified across banks and doc types, labeled for header fields and every row's (date, debit, credit, balance) in a CSV. Labeling tool: the exported queue opened in a spreadsheet, one row per field. This set is the sample size quoted to the client for statements, and it is used to measure how well checks B and C predict true errors (precision of the flags).

### 6.6 Duplicate detection (Q5)

Build a test set from the statement files: 20 originals, each with 3 variants (re-scan via `scan_low`, rotate 180° then `photo`, crop 5% margins) plus 20 distinct statements from the same accounts as hard negatives. Detector: exact SHA-256 on bytes, then Jaccard over the set of `(date, amount)` row tuples, threshold 0.8, plus perceptual hash on the page image as a second signal. Report recall on variants and false-positive rate on hard negatives, per threshold in {0.6, 0.7, 0.8, 0.9}.

## 7. Confidence & calibration

Neither model emits a field confidence, so the repo defines one per field and calibrates it on labeled fields (ThaiOCRBench KIE + digital statement pages + manual set). Q3 is answered by the calibration table, not by a claimed score.

### 7.1 Field confidence proxy

```latex
\text{conf}(f) = \sigma\big(w_0 + w_1\,\overline{\log p}(f) + w_2\,\min \log p(f) + w_3\,\text{agree}(f) + w_4\,\text{arith}(f)\big)
```

- `mean_logprob`, `min_logprob`: over the completion tokens that produced the field value (token spans located by matching the value string back into the completion; fall back to whole-completion stats if not found and set `span_found=false`).
- `agree`: 1 if the other model's normalized value matches, else 0, `null` if the other model has no value.
- `arith`: 1 if the field participates in a balance/total equation that holds, 0 if it breaks, `null` if not applicable.
- Weights fitted by logistic regression on `field_exact` with 5-fold CV grouped by `sample_id`; a `logprob_only` variant is fitted alongside for the case where a single model runs in production.

### 7.2 Calibration output

`calibration.jsonl` and a table in the report:

| Band | conf range | n fields | accuracy | 95% CI |
| --- | --- | --- | --- | --- |
| 5 | ≥ 0.98 |  |  |  |
| 4 | 0.90–0.98 |  |  |  |
| 3 | 0.75–0.90 |  |  |  |
| 2 | 0.50–0.75 |  |  |  |
| 1 | < 0.50 |  |  |  |

Also reported: ECE (10 bins), reliability plot, and the same table restricted to critical fields.

### 7.3 Thresholds and review rate

`calibrate` searches `t_accept` and `t_reject` so that accuracy on auto-accepted fields ≥ 99.0% and ≥ 99.5% (two rows), and reports for each: auto-accept share, review share, reject share, and accuracy inside each bucket. Review rate = share of fields sent to human review; this is the number quoted for Q3. Thresholds are stored in `runs/<run_id>/thresholds.json` with the fitted weights so the same operating point can be reproduced.

## 8. Throughput & latency harness

`ocrbench bench-latency` runs a fixed 200-page mix (100 ThaiOCRBench full-page, 100 statement pages, `clean` condition) against each model on each GPU and writes `latency.jsonl`.

- Sweep client concurrency ∈ {1, 4, 8, 16, 32}; 30 s warm-up excluded; each point runs the full 200 pages.
- Record per request: `latency_ms` (end to end), `ttft_ms`, `prompt_tokens`, `completion_tokens`, `image_px`.
- Report per (model, GPU, concurrency): p50, p95, p99 sec/page; pages/hour; GPU-hours per 1,000 pages; peak GPU memory from `nvidia-smi --query-gpu=memory.used` sampled every 1 s.
- GPUs: H100 80 GB and L4 24 GB, same vLLM version and dtype (bf16); on L4 add an fp8/AWQ row only if a released quantized checkpoint exists — never quantize ad hoc for the benchmark.
- Multi-page throughput: also time the 10 multi-page statement files end to end, including merge, to give a per-document p95.

Cost per page is derived in the report from GPU-hours per 1,000 pages × a configurable GPU hourly rate in `run.yaml`; the rate is an input, not a benchmark result.

## 9. Reporting

`ocrbench report <run_id>` writes `runs/<run_id>/report/` with `scorecard.md`, `scorecard.xlsx` (one sheet per table), `plots/` (reliability curve, latency curves, CER by condition) and `samples/` (anonymized examples for Q4).

### 9.1 Scorecard table

One row per client question, so the client's sheet can be answered line by line.

| Question | Metric | TeleOCR | dots.ocr | n | Data | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 Thai text | CER clean / scan\_low / photo |  |  |  | ThaiOCRBench OCR tasks |  |
| Q1 Handwriting | CER |  |  |  | Handwritten extraction |  |
| Q1 Tables | TED |  |  |  | Table parsing |  |
| Q1 Bank statements | row F1, file\_reconciles |  |  |  | Statements |  |
| Q2 Field accuracy | P / R / F1 / exact |  |  |  | KIE + statements | critical-only row beneath |
| Q2 FA / FR | rate |  |  |  | KIE |  |
| Q3 Confidence | accuracy per band, review rate @99% |  |  |  | labeled fields |  |
| Q4 Output | capability matrix |  |  |  | probe |  |
| Q5 Duplicates | recall / FPR @0.8 |  |  |  | statement variants |  |
| Q5 Reconciliation | row\_consistency\_rate |  |  |  | statements |  |
| Q8 Throughput | p95 sec/page, pages/hour, H100 and L4 |  |  |  | 200-page mix |  |

### 9.2 Output contract (Q4 sample)

Every extracted field in the client-facing sample follows this shape; the anonymizer replaces names with `[NAME]`, account numbers with the last 4 digits, and blanks bboxes for redacted spans.

```markdown
{
  "field": "closing_balance",
  "value": 152340.75,
  "raw": "152,340.75",
  "type": "decimal",
  "unit": "THB",
  "definition": "Balance after the last transaction in the statement period",
  "source": {"file": "stmt_0003.pdf", "page": 4, "bbox": [812, 1390, 964, 1412]},
  "normalization": ["strip_thousands_sep", "decimal"],
  "confidence": 0.97,
  "error_flags": [],
  "model": "dots.ocr", "prompt_version": "stmt-v3", "run_id": "2026-09-25-a"
}
```

`error_flags` ∈ {`balance_mismatch`, `total_mismatch`, `model_disagreement`, `low_confidence`, `missing_required`, `page_gap`}. Sheet 02 items 3.3 and 3.4 (untracked values, incomplete results) are answered by counting `missing_required` and `page_gap` per file in the report.

## 10. Implementation plan

Four working days on one H100 plus one L4 session; two engineers can split M1/M2 from M3.

| Milestone | Deliverable | Acceptance |
| --- | --- | --- |
| M0 Scaffold (day 1 am) | repo skeleton, schemas, CLI stubs, CI running `pytest` | `ocrbench --help` lists all 6 commands |
| M1 Data (day 1) | `prepare` builds manifest for ThaiOCRBench subset + statements + 3 conditions | manifest row count = samples × 3; 20 random rows opened and checked by eye |
| M2 Inference (day 2) | both adapters via vLLM, `infer` writes predictions with logprobs and latency | zero silent drops; error rate < 2%; probe matrix filled |
| M3 Scoring (day 2–3) | all scorers + slices; `bmfl` matches published sample scores within 0.01 | scorecard tables render with n and CI |
| M4 Statements (day 3) | text-layer GT, arithmetic checks, agreement, duplicate set, review queue exported | manual set of 30–50 pages labeled and loaded |
| M5 Calibration + latency (day 3–4) | fitted proxy, bands, thresholds, review rate; latency sweeps on H100 and L4 | ECE reported; p95 table for both GPUs |
| M6 Report (day 4) | `scorecard.md/.xlsx`, plots, anonymized Q4 samples | one row per client question filled or marked not-benchmarkable |

### Task list

- [ ] M0: `pyproject.toml`, `schemas.py`, `cli.py` stubs, `tests/test_schemas.py`
- [ ] M1: `thaiocrbench.py` loader with task-name assertion; `bankstmt.py` ingestion with `pdffonts` routing; `degrade.py` with seeded transforms; `manifest.py`
- [ ] M2: `vllm_client.py` (logprobs, timeout, retry, timing); `teleocr.py`, `dotsocr.py` adapters; `parsers.py`; `serve_*.sh`; `probe` command
- [ ] M3: `text.py`/`fields.py` normalizers with Buddhist-year conversion tests; `cer_wer.py`, `bmfl` port with fixture test; `ted.py`; `kie_f1.py` with FA/FR; `anls.py`; slice aggregation with bootstrap CI
- [ ] M4: `statement.py` merge + continuity; `arithmetic.py`; cross-model agreement; `duplicates.py` + variant generator; `review/queue.csv` export and CSV label loader
- [ ] M5: `proxy.py` span matching + features; `calibrate.py` (logistic fit, bands, ECE, threshold search); `latency/bench.py` concurrency sweep + `nvidia-smi` sampler
- [ ] M6: `scorecard.py`, `render.py` (md, xlsx, plots), anonymizer; README with reproduce command

### Reproduce command

```markdown
ocrbench prepare --config configs/run.yaml --run-id 2026-09-25-a
ocrbench infer   --run-id 2026-09-25-a --models teleocr,dotsocr
ocrbench score   --run-id 2026-09-25-a
ocrbench calibrate --run-id 2026-09-25-a --target-acc 0.99,0.995
ocrbench bench-latency --run-id 2026-09-25-a --gpu h100 --concurrency 1,4,8,16,32
ocrbench report  --run-id 2026-09-25-a
```

## 11. Open questions & risks

- [ ] ThaiOCRBench task names, domain field and license terms: confirm on first load; if no domain field, drop the Government/Finance slice.
- [ ] TeleOCR bbox and JSON support: unknown until the probe runs; if `no`, dots.ocr carries Q4 and TeleOCR is text-only in the scorecard.
- [ ] Statement corpus: how many files, which banks, and what share is digital PDF; fewer than 40 digital pages weakens check A.
- [ ] Manual labeling owner and time (about 4 hours for 50 pages).
- [ ] Whether to include `typhoon-ocr1.5-2b` as a third reference model (adds \~1 day of inference, gives a published baseline).
- [ ] GPU hourly rate to use for the cost line.

Risks: ThaiOCRBench is not บสย. paper, so scores are a capability proxy and must be labeled as such in the client deck; the confidence proxy is fitted on the benchmark and will need re-fitting on real บสย. documents; BMFL reimplementation may drift from the official scorer, so the fixture test against published numbers is mandatory before quoting comparability.
