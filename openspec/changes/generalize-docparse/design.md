# Design

## Context

See proposal.md for motivation. The facts that shape this design:

**docparse state (`add-docparse-system`, 7 of 34 tasks done)**
- Implemented: `docparse/schemas.py`, `artifacts.py`, `config.py`, `registry.py`, `eval_files.py`, `configs/docparse/pipeline.yaml`, `classes/bank_statement.yaml`, `eval_files.v1.yaml` (PR #59, frozen). Stubs: `classify.py`, `layout.py`, `segment.py`, `recognize/*`, `reconcile.py`, `html.py`, `verify.py`, `extract/*`, `service/*`, `trocr/*`.
- Spike A: muocr is the best statement line reader; Spike B: 97 usable text-layer pages (70 of them KTB).
- Design decisions D1–D12 of `add-docparse-system` stay in force unless a decision below says otherwise.

**Statement assumptions in the code** (the audit for D1 below): `ClassifyArtifact.bank`; `registry.FieldType` (5 types); `BankIn`/`BankNotIn` `required_when`; `ClassSpec.banks_from` loading `BankOverrides`; `aliases_from: header:<field>` from `stmt.HEADER_LABELS`; `columns: column:<role>` from `stmt.COLUMN_KEYWORDS`; `FIELD_VALIDATORS.known_bank`; `CROSS_VALIDATORS.running_balance` typed on `StatementFile`; `DocumentClass.validate_statement`; `EvalFile.harness_id` (`BS-<bank>-NNNN`) and `EvalFile.bank`; design D4 (`{class, confidence, bank}`), D9 ("Check B for statements"), D10 (`StatementRow`s, `balance_on_date`, `reconciliation`), D12 (per-bank CIs, statement Q/A).

**Harness state (`add-ocr-benchmark-harness`, 43 of 58 tasks)**
- `Task` = `ocr_fullpage, ocr_line, handwriting, table, docparse, kie, kie_map, classify, statement`; `Source` = `thaiocrbench, bankstmt`; `GtKind` = `json, text, html, text_layer, none`.
- Dataset kinds: `thaiocrbench` (HF) and `bankstmt` (a directory, bank from the file name). `text_layer_usable` accepts a page only if it contains a statement keyword. `pdf_repair.repair_tounicode` fixes identity ToUnicode maps. `degrade.encode_condition` is deterministic per `(sample_id, condition)`.
- Metrics: `cer`/`wer`, `ted` (TEDS), `ted_docparse`, `anls`, `kie_f1` with FA/FR, statement row F1 and header fields, Check B, `tob_score`. No layout or reading-order metric.
- Slices: task × model, condition × model, task × condition × model, bank × doc_type × model (statements only), domain × model (ThaiOCRBench Government/Finance). Cluster bootstrap in `aggregate.py`.
- ThaiOCRBench is prepared in run `2026-09-22-a` with the tasks Full-page OCR, Text recognition, Fine-grained, Handwriting, Table parsing, Document parsing, KIE, KIE mapping, Document classification.
- The manual statement anchor set is `ocrbench review export/load` (field and row cells); not labelled yet (docparse task 1.4).

**Environment**
- Public repository. Client data under `data/` (gitignored, DVC on GCS). Fixtures must not contain client content.
- GPU work is batched into H100 sessions (`docs/continuing-on-another-host.md` §0b). The dev box has no CUDA.
- The dev container has `fonts-thai-tlwg` only. D7 of `add-docparse-system` already plans Sarabun, Noto Sans Thai and Kanit for TrOCR synthetic lines.

## Goals / Non-Goals

**Goals:**
- Any document gets a parse (layout, text, tables) and can be queried; a class with fields gets verification and typed answers.
- Adding a class is a YAML file plus, at most, one validator function; no stage code changes.
- Every class has its own frozen eval list and a falsifiable per-class criterion, plus one generic criterion that makes "docparse is better on documents in general" a measured claim.
- Ground truth for new classes comes first from real born-digital PDFs and ThaiOCRBench, and only then from synthetic pages, with provenance that keeps the two apart in every report.
- Nothing in the statement path changes behaviour: the v1 list, the gate, Check B and the statement criterion stay as they are.

**Non-Goals:**
- Per-class model fine-tuning beyond the existing conditional TrOCR fine-tune.
- A class taxonomy beyond the four classes here; more classes follow the same recipe.
- Multilingual beyond Thai and English.
- Cross-page reading order (reading order is scored within a page).

## Decisions

### D1. Audit: every statement assumption and its class-generic replacement

| Where | Statement assumption | Replacement |
|---|---|---|
| `schemas.ClassifyArtifact.bank` | a bank for statements | `variant: str \| None`: the detected member of the class's variant list, or `unknown` when the class has variants, else `None` |
| `registry.BankIn/BankNotIn` | `required_when` keyed on banks | `variant_in` / `variant_not_in` / `always` / `never` (`never` = optional: looked up, never recovered, never missing) |
| `registry.ClassSpec.banks_from` + `BankOverrides` | one bank file per class, overrides typed as `BankOverrides` | `variants_from: <path>` to a variant file (`variants: [...]`, `overrides: {<variant>: {...}}`); overrides are stored as a plain mapping on the class and parsed only by the cross-validator family that uses them (statements: `BankOverrides`). `configs/docparse/variants/banks.yaml` replaces `../../datasets/bankstmt.yaml` as the statement's variant file and is generated from the dataset config by a test, so the two cannot drift |
| `registry.FieldType` | `text, amount, date, account, table` | add `number`, `identifier`, `percent`; `account` stays as an `identifier` with the account validator |
| `aliases_from: header:<field>` | only the statement mapper's labels | an alias-provider namespace: `statement_header:<field>` (the existing provider, renamed), and inline `aliases:` for every other class. The "no second copy" rule stays for statements |
| `columns: {name: column:<role>}` | only `stmt.COLUMN_KEYWORDS` roles | a column spec is either a provider reference (`statement_column:<role>`) or inline `{type, aliases}`; a `table` field may also declare `columns: infer`, which takes the header row as read |
| `FIELD_VALIDATORS.known_bank` | bank list | `known_variant`; new validators `number`, `percent`, `thai_tax_id` (13 digits, checksum), `regex:<pattern>` |
| `CROSS_VALIDATORS.running_balance(StatementFile)` | typed on the statement model | cross validators take a `DocumentView` (typed fields, typed tables, variant, overrides) and return `(ok, reason)`. `running_balance` builds its `StatementFile` from the view. New: `line_items_sum` (invoice: Σ line totals = subtotal, subtotal + vat = total, within 0.01), `date_order:<a>,<b>`, `fields_consistent:<a>=<b>` |
| `DocumentClass.validate_statement` | statement model | `validate_document(view)`; the verifier and the extractor call only this |
| `classify` prompt (D4) | `{class, confidence, bank}` | `{class, confidence, variant}`; the prompt lists each class's variants only when the class has them |
| `verify` (D9) | "Check B for statements" | the class's cross validators; region ladder and guards unchanged |
| `extract` (D10) | `StatementRow`s via the mapper; `balance_on_date`, `reconciliation` | a generic `TableView` (D5); query types are declared per class from a shared template catalogue |
| `eval_files.EvalFile` | `harness_id` must be `BS-<bank>-NNNN`; `bank` required | per-class lists with `{file, source, harness_id?, variant?}`; the v1 statement file keeps its schema and is loaded unchanged |
| `ocr_bench.data.bankstmt.text_layer_usable` | statement keywords | a class-neutral check (D7.2) |
| docparse adapter (task 3.7) | `parsed.html` → statement `NormalizedPage` | one mapping per harness task (D6.5) |

Every row is a config or module change inside the existing stage boundaries; no stage gains a new input or output file.

*Alternative:* keep `bank` and add parallel fields per class. Rejected: every new class would need schema edits, which is what the registry exists to avoid.

### D2. The class model

A class file (`configs/docparse/classes/<id>.yaml`) declares:
- `id`, `description` (for the classifier);
- `variants_from` (optional) and `variant_aliases` (optional, strings the classifier may use for each variant);
- `fields`: each with `name`, `type`, `aliases` or `aliases_from`, `region_hint`, `validators`, `required_when`; `table` fields carry `columns`;
- `cross_validators`;
- `query_types`: the subset of the template catalogue the class supports, with the bindings each template needs (e.g. `lookup_by_key: {table: transactions, key: date, value: balance}`);
- `line_reader` (optional): overrides `pipeline.line_reader` for this class (D9).

The registry validates every reference at load time and names the file and key on failure, as today. `unknown` is a built-in class that no file may define: no fields, no variants, no cross validators, `query_types: [find, table_filter_aggregate, free_form]`.

**Why a template catalogue and not per-class plan code:** the extractor's safety comes from fixed tool sequences with verified parameters (D10 of `add-docparse-system`). Keeping templates in code and only their bindings in YAML preserves that while letting a class opt in to what makes sense for it.

### D3. The first classes

| Class | Why first | Fields (required unless noted) | Table | Variants | Cross validators |
|---|---|---|---|---|---|
| `bank_statement` | exists | unchanged | `transactions` | banks | `running_balance` |
| `invoice` | the most common commercial KIE target; CORD/SROIE give structure analogues; synthetic templates are easy and realistic | `vendor_name`, `vendor_tax_id` (`thai_tax_id`, optional), `document_no`, `issue_date`, `buyer_name` (optional), `subtotal`, `vat_amount` (optional), `total` | `line_items` (`description`, `quantity`, `unit_price`, `amount`) | `tax_invoice`, `receipt`, `invoice` (the printed title decides) | `line_items_sum` |
| `government_form` | the largest born-digital Thai corpus with free text layers (gazette, agency letters, forms); ThaiOCRBench's Government domain | `document_no`, `issue_date`, `issuing_agency`, `subject`, `reference_no` (optional) | none required; tables parsed if present | none in v1 | `date_order` only when an expiry date exists (optional field `valid_until`) |
| `tabular_report` | annual reports, price lists, university schedules: the payload is tables and prose, with no fixed header fields; it exercises layout, reading order and TEDS | `title` (optional) | `tables` with `columns: infer` | none | none |
| `unknown` | fallback | none | inferred | none | none |

Field names are the extractor's vocabulary and the Q/A generator's, so they are fixed here. Aliases (Thai and English) and region hints go into the class files in task 2.x.

*Alternative:* `receipt` as its own class. Rejected for v1: the same fields, and the variant covers the difference.
*Alternative:* a `letter` class separate from `government_form`. Deferred: the born-digital corpus will show whether letters need different fields (open question Q3).

### D4. Classification with variants

As D4 of `add-docparse-system`, with `variant` in place of `bank`. The prompt lists class ids and descriptions, and under each class that has variants, its variant ids and aliases. The reply `variant` is accepted only if it is in the chosen class's list; otherwise `unknown`. ThaiOCRBench Document classification labels are scored through a committed label map (`configs/docparse/classify_labels.yaml`: benchmark label → registry class or `unknown`), so the harness's ANLS applies to docparse's class ids.

### D5. Generic verification and extraction

- **DocumentView** (`docparse/extract/view.py`): typed fields from `fields.json`, typed tables from the reconciled `<table>` elements, the variant and the class overrides. It is the only input of cross validators and extraction tools.
- **TableView**: for a `table` field with declared columns, the header row is matched to the column aliases (the statement keeps the mapper for this, since it also handles bank quirks); for `columns: infer`, each header cell becomes a column and the column type is inferred with `ocr_bench.normalize.fields.field_type` over its cells. Every row keeps its row id and block id for provenance. Cells are normalized with `normalize_field` by column type.
- **Template catalogue:** `field` (header field), `table_filter_aggregate` (filter and sum/count/min/max/list), `lookup_by_key` (the value of one column on the row where the key column matches), `cross_check` (run a named cross validator and report), `find` (text search, no typing), `free_form` (bounded ReAct, flagged). Statement `balance_on_date` = `lookup_by_key` bound to `transactions.date → balance`; `reconciliation` = `cross_check: running_balance`. Routing stays one LLM call with the parameter schemas of the class's query types only.
- **Verification** of read and computed values is unchanged. For `columns: infer` tables, a read value must still match the cited cell after normalization.

### D6. Evaluation

**D6.1 Eval lists.** One list per class: `configs/docparse/eval/<class>.v<N>.yaml`, each entry `{file, source, harness_id?, variant?, pages?}`; `source` is a harness `Source`. `bank_statement` keeps `configs/docparse/eval_files.v1.yaml` (frozen) and later `v2` from docparse task 1.4; the per-class loader reads that file through a compatibility shim and never rewrites it. `docparse.eval_files.all_eval_file_ids()` returns the union over every class's latest list and is the single input of the leak check (D8). The version of every list used is recorded in each evaluation result.

A list is frozen before any tuning of docparse on that class (prompts, thresholds, region hints, class aliases). Tasks in group 4 freeze; tuning tasks come after.

**D6.2 Criteria per class.** All against the best single VLM on the same samples (chosen per class by the class's primary metric), paired cluster bootstrap with files as clusters, per condition:

| Class | Primary | Also reported | "Better" means |
|---|---|---|---|
| `bank_statement` | statement field F1 | CER, row F1, Check B rate, Q/A | unchanged (`add-docparse-system`) |
| `invoice` | KIE field F1 (class fields, FA/FR) | page CER, line-item TEDS, Q/A exact match, provenance, abstention | field-F1 lower CI bound > 0 and CER upper bound < 0 |
| `government_form` | KIE field F1 | page CER, reading order, Q/A | same rule |
| `tabular_report` | table TEDS | page CER, layout block F1, reading order | TEDS lower bound > 0 and CER upper bound ≤ 0 |
| `unknown` (ThaiOCRBench Document parsing, open sets) | page CER | `ted_docparse`, TEDS, layout block F1, reading order | CER upper bound < 0 and TEDS not worse |

Q/A sets are generated per class from the class's fields and tables (the statement generator generalized), with ≥10% unanswerable items. KIE on ThaiOCRBench uses the benchmark's own keys: docparse answers each key as a `field`/`find` query, and `kie_f1` scores it.

**D6.3 The generic criterion.** Over all real eval pages of all classes together, the paired cluster-bootstrap CI (clusters = files) of docparse − best VLM on page CER and, where a page has a table, TEDS. docparse is "generally better" only if the CER difference's upper bound is < 0 and the TEDS difference's lower bound is ≥ 0, with the best VLM chosen per class. A class that fails its own rule is still reported; the generic verdict never hides a per-class failure.

*Why CER and TEDS:* they exist for every page regardless of fields, they are what the harness already scores, and a reading-order error shows up in CER on canonical text.

**D6.4 Reporting rule.** Every number is reported per class × source × condition with `n` and a 95% CI, and additionally per variant where a class has variants (banks for statements). The headline tables use real sources only (`thaiocrbench`, `bankstmt`, `thaipdf`, structure sets). Synthetic results appear in their own table, titled as such, and in no aggregate that mixes sources. A slice with n < 10 files is shown with its n and no verdict.

**D6.5 The harness scores docparse on every task.** The adapter (`ocr_bench/models/docparse.py`, from docparse task 3.7) maps `parsed.html` and `fields.json` to the `NormalizedPage` each task expects: `ocr_fullpage` → canonical text in reading order; `table` → the page's table HTML; `docparse` → markdown with headings from `<h*>`; `kie`/`kie_map` → one extraction per key; `classify` → the mapped label; `statement` → the mapper, as before; `layout` (new, D6.6) → blocks with bbox and category. New scorer pairs in `dispatch.py`: `(ocr_fullpage, text_layer)` → cer/wer against `gt_text`; `(table, text_layer)` → TEDS against text-layer tables when the page has them (D7.2); `(layout, layout)` → block F1; `(docparse|ocr_fullpage, layout)` → reading order.

**D6.6 New metrics.** `layout_f1`: category-aware block matching at IoU ≥ 0.5, micro F1 over the page (DocLayNet categories mapped to docparse's seven). `reading_order`: for matched blocks, normalized Kendall τ between predicted and ground-truth order (1 = identical). Both are structure-only and language-independent, so open sets can score them.

**D6.7 Slices.** `aggregate.py` gains `class_source_condition_model` (keyed on the new `doc_class` and `source`) and `class_variant_model`. The `bank_doctype_model` slice stays.

### D7. Data plan by role

**D7.1 Real evaluation data first.**

*ThaiOCRBench* (already prepared): task → docparse stage mapping: Document parsing → full parse (`unknown`), Table parsing → TEDS, Full-page OCR → CER, KIE → extraction per key, Document classification → class map (D4). Its licence terms are checked once and recorded (task 3.1).

*Born-digital Thai PDFs* (`kind: pdfcorpus`, source `thaipdf`): a directory `data/thaipdf/<class>/<variant?>/<file>.pdf` plus `sources.yaml` entries. `prepare` routes digital vs scanned as for statements, repairs ToUnicode maps with `pdf_repair`, extracts text and word boxes, and applies the class-neutral usability check (D7.2). Pages become samples with `doc_class` from the directory and `gt_kind=text_layer`; the three conditions come from `degrade.encode_condition` unchanged. Candidate sources, each needing a licence decision (open question Q1): Royal Gazette and agency circulars (government), SET-listed annual reports (56-1 One Report; publicly downloadable, copyright retained: evaluation use only, never redistributed), university announcements and course documents, and invoices (few public ones exist; the class leans on synthetic and anchor pages, see Q2).

*Class-neutral text-layer check (D7.2).* `text_layer_usable` becomes `ocr_bench/data/text_layer.py:usable(text)`: NFC-normalize; reject if any private-use codepoint remains after repair; tokenize with `pythainlp` `newmm` and require that ≥ 0.6 of alphabetic tokens are dictionary words or ASCII words, and that the Thai or Latin letter share is ≥ 0.5 of non-space characters. The statement keyword test becomes a class-specific extra for `bank_statement` only. Thresholds are measured on the 130 digital statement pages (97 usable, 33 not) and must reproduce that split exactly before the check is used elsewhere.

*Table ground truth from text layers.* `pdfplumber.extract_tables` on ruled tables gives cell text with positions. It is accepted as `table` ground truth only when the extraction is rectangular, has a non-empty header row, and the page's manual anchor spot-check (task 4.4, 30 tables) agrees on ≥ 95% of cells. Until that check passes, text-layer pages score CER only.

**D7.3 Open structure sets** (`kind: structure_set`, one source value per set), scoped to language-independent stages, capped per set (default 200 pages, seeded), test splits only for evaluation:

| Set | Stage | Metric | Licence (to confirm, Q1) |
|---|---|---|---|
| DocLayNet | layout | `layout_f1` | CDLA-Permissive-1.0 |
| OmniDocBench | layout, reading order, tables | `layout_f1`, `reading_order`, TEDS | check (data terms differ from code) |
| PubTabNet, FinTabNet | tables | TEDS | CDLA-Permissive-1.0 |
| FUNSD, XFUND | form structure (`government_form` key–value layout) | `layout_f1` on key/value blocks, field F1 on linked pairs | FUNSD research-only; XFUND CC BY-NC-SA 4.0 |
| CORD, SROIE | receipt structure (`invoice` fields mapped) | field F1 on mapped fields, TEDS on line items | CORD CC BY 4.0; SROIE ICDAR terms |

They test the layout adapter, segmentation, table reconstruction and the verifier's geometry, not Thai reading. Their text metrics are reported but carry no verdict. Non-commercial sets are used for internal evaluation only, and never for training, until the user decides otherwise (Q1).

**D7.4 Synthetic documents** (`kind: synthetic`, source `synthetic`): HTML templates per class under `ocr_bench/data/synthetic/templates/<class>/`, filled from seeded generators (Thai names, addresses, amounts, dates in Thai and Gregorian calendars, agency names, line items), rendered with WeasyPrint to PDF, so the text layer and the generator's own field and table JSON give ground truth. Fonts: TLWG, Sarabun, Noto Sans Thai, Kanit (OFL), chosen per page by seed. Degradation reuses `degrade`. Every page records template id, seed, font and condition. Uses: TrOCR training lines (already planned), unit-test fixtures (the only pages that may be committed, because they contain no client content), rare-case coverage (negative amounts, multi-page tables, stamps and watermarks as overlays) and a separately labelled eval slice. Train and eval synthetic pages use disjoint seed ranges, and a template edited after an eval slice is frozen gets a new template id.

*Alternative:* LaTeX templates. Rejected: tables and invoice layouts are easier in HTML/CSS, WeasyPrint shapes Thai through Pango, and no TeX toolchain enters the container.
*Alternative:* Chromium via Playwright. Rejected: a browser in the dev container for the same output.

**D7.5 Manual anchor set per class** ([Human]): 30–50 pages per new class, from the real corpus where it exists (government, reports) and from collected real invoices or, failing that, printed-and-scanned synthetic invoices (Q2). The label is one JSON file per page (`manual-anchor-sets` spec): full text in reading order, tables as HTML, fields by class name, bboxes optional. Pre-fill is allowed from one VLM that is neither a comparison baseline for that class nor a model bound to a docparse role (Qwen3-VL is the verifier reader, so pre-filling with it would make the labels agree with docparse's own recoveries); it is recorded in the label, and the labeller must confirm every field and every table cell. Client-sourced pages are never sent to an external service, so they are labelled without pre-fill (Q6). `ocrbench anchor validate` checks the schema, the class, normalization of typed fields and the pre-fill record. The statement anchor set keeps the `review` queue format.

### D8. Leakage

The `trocr-thai-training` rule generalizes: a training-set build refuses any crop or page from a file in `all_eval_file_ids()`, any page of an open set's test split, and any synthetic page whose seed is in an eval range. Statement v1 stays in that union. The check runs in `docparse/trocr/data.py` and in the synthetic builder, and names every offending id.

### D9. Line reader per class

`docparse gate` reports CER per reader per class (text-layer lines of that class), beside the pooled numbers. The gate decision stays global (one default reader), but a class file may set `line_reader` to the other reader when its own lines show the opposite result; the override is recorded in the readings artifact. Segmentation's `max_aspect` follows the active reader as before.

### D10. Public repository and data boundaries

- Committed: class files, eval lists (ids, source, sha256 of the file, optional URL), `sources.yaml` (name, URL, licence, permitted uses, retrieval date), templates, label schemas, label maps.
- Never committed: PDFs, images, text layers, labels of real pages, model replies on real pages. They live under `data/<corpus>/` and `runs/`, tracked by DVC where they must survive a host change.
- Fixtures under `tests/fixtures/` come only from synthetic pages or from open sets whose licence allows redistribution of samples.
- `*.pdf` is already gitignored; the synthetic fixtures that tests need are PNG renders of synthetic pages, committed small.

### D11. Compute plan

Dev box: everything with recorded replies and synthetic fixtures (groups 1–3, 5–6). H100 sessions, batched: (a) baselines (dots.ocr, typhoon-ocr1.5, PaddleOCR-VL, Qwen3-VL as a fourth baseline) and docparse on `thaipdf` and the structure sets; (b) the per-class gate; (c) the full per-class evaluation. Each session follows the runbook's prerequisite list (DVC push of new corpora first).

### D12. Sequencing and archive

- Groups 1–2 (contracts and classes) land before `add-docparse-system` tasks 3.1, 5.1 and 6.1 are implemented, so classify, verify and extract are written once against the generic contracts. Statement scenarios of `add-docparse-system` remain the acceptance tests of those tasks.
- Spec deltas here are `ADDED` requirements (the parent changes are unarchived). Task 8.1 converts the superseding ones to `MODIFIED`/`REMOVED` after `add-docparse-system` and `add-ocr-benchmark-harness` are archived, so the main specs end up with one requirement per behaviour.

## Risks / Trade-offs

- **[Born-digital corpora are born-digital]** → Their `scan_low`/`photo` conditions are synthetic degradations, not scans. The anchor sets include real scans and photos (D7.5), and the report labels the condition origin.
- **[Few public Thai invoices]** → The invoice class may have < 30 real pages at freeze time. Then its list is frozen with what exists, its verdict is reported as "n too small" until the anchor set lands, and synthetic invoices stay out of the verdict (Q2).
- **[Open-set licences are mixed, some non-commercial]** → Used for internal evaluation only and never for training until the user decides; each set is a separate `sources.yaml` entry that `prepare` refuses to load without a `permitted_uses` field.
- **[A class-neutral usability check admits bad text layers]** → It must reproduce the statement 97/33 split before use, and the anchor spot-check compares text-layer text with the page on 30 pages per corpus.
- **[Inferred table columns are noisy]** → `columns: infer` cells are verified by normalized match like any read value, and aggregates over inferred columns are flagged `inferred_columns` in the answer.
- **[The generic criterion rewards the layout backend, not docparse]** → The best VLM baseline is the same layout backend used alone, so the difference measures what reconciliation and verification add.
- **[More corpora, same H100 budget]** → Caps per set (200 pages), and the per-class lists bound the eval size; the structure sets run in the baseline session only once.
- **[Pre-filled labels bias the anchor sets]** → The pre-fill model is excluded from that class's comparison set, and the label records it.
- **[Variant rename touches merged code]** → `bank_statement.yaml`, `registry.py` and their tests change in one task with the old scenarios kept; the eval v1 file is read through a shim and not edited.

## Migration Plan

- `configs/docparse/classes/bank_statement.yaml` changes `banks_from`/`bank_not_in` to `variants_from`/`variant_not_in`; the registry refuses the old keys with a message naming the new ones. Nothing else in the statement path changes.
- New dataset kinds are new YAML files; existing runs and `configs/run.yaml` are untouched. `manifest.jsonl` gains `doc_class` with a default of `null`, so older manifests still validate.
- Rollback: remove the new class files and dataset YAMLs; the registry and harness run as before.

## Open Questions

Decisions that need the user, listed in tasks as [Human]:

- **Q1 Licences.** Which open sets may be used, and for what (evaluation only vs training), given that docparse serves a commercial client: FUNSD (research-only), XFUND (CC BY-NC-SA), SROIE (ICDAR terms), OmniDocBench (data terms to confirm); and the terms for ThaiOCRBench, SET annual reports and Royal Gazette documents. Task 3.1 records the answers in `sources.yaml`.
- **Q2 Invoices.** Can the client or the user supply real Thai invoices and receipts (even 30 pages) for the anchor set? If not, the invoice anchor set is printed-and-scanned synthetic invoices, and the class verdict is reported as provisional.
- **Q3 Class split.** Should official letters be a class separate from `government_form`, and should `tabular_report` be split by domain (financial vs academic)? The born-digital corpus survey (task 4.1) is the input; the default is to keep four classes.
- **Q4 Priority.** Which new class matters most to the client, so its anchor set and H100 time come first? The default order is `government_form`, `tabular_report`, `invoice`.
- **Q5 Generic verdict threshold.** D6.3 requires CER better and TEDS not worse. Is "not worse on both, better on at least one" acceptable instead, which is easier to reach? The default is the stricter rule.
- **Q6 Anchor pre-fill model.** Which model pre-fills the anchor labels of public-source pages, given that it can be neither a baseline (dots.ocr, typhoon-ocr1.5, PaddleOCR-VL, Qwen3-VL-8B) nor a docparse role model (Qwen3-VL, Qwen3)? The default is an external VLM API on public-source pages only, and no pre-fill for client-sourced pages. The alternative is to drop Qwen3-VL-8B from the comparison set of the new classes and pre-fill with it, accepting the bias noted in D7.5.
