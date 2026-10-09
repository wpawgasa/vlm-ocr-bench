# Proposal

## Why

docparse (`add-docparse-system`) is designed and evaluated on one document class, Thai bank statements. The user wants it to parse any document. Today the statement shows through in the code and the plan:
- The registry has a `bank` field, bank-keyed `required_when` rules and `BankOverrides`, and its aliases, column roles and cross validator (`running_balance`, Check B) come from the `ocr_bench` statement mapper.
- The classifier reports a `bank`, and the eval list format requires a bank and a `BS-` harness id.
- Extraction routes queries into statement query types (`balance_on_date`, `reconciliation`) over `StatementRow`s.
- Evaluation has a single frozen list (`configs/docparse/eval_files.v1.yaml`, 6 files, 1 anchor set pending) and a success criterion built on statement field F1 and Check B.
- Text-layer ground truth is accepted only when the page contains a statement keyword (`text_layer_usable`).
- The data is 31 client files from 7 banks. Nothing else has ground truth.

The pipeline stages (classify, layout, segment, recognize, reconcile, verify, extract) are already class-agnostic in shape. What is missing is a class-generic contract for everything the statement supplied by hand, a data plan that gives other classes ground truth, and an evaluation that can say "better" for each class without statement field F1.

## What Changes

- **Class-generic registry.** `bank` becomes a per-class `variant` (the statement's variants are its banks). Field types, validators, alias providers, column specs and cross validators become a catalogue that any class file can use. Statement-only code paths (mapper aliases, `BankOverrides`, Check B) stay, but are reached only through the `bank_statement` class file.
- **First document classes:** `invoice` (invoices, tax invoices, receipts), `government_form` (Thai government forms and official letters), `tabular_report` (reports whose payload is tables: annual reports, price lists, schedules), beside `bank_statement`. `unknown` remains the fallback and still gets the full parse (layout, full text, tables), so every document is parsable and queryable.
- **Classification** reports `variant` instead of `bank`, from the variant list of the chosen class.
- **Verification and extraction** work from the class file: cross validators run on a typed document view, tables are read through a generic typed table view (header-derived columns), and the query types a class supports are declared in its class file. Statement behaviour is unchanged.
- **Evaluation per class:** one frozen, versioned eval list per class (`configs/docparse/eval/<class>.v<N>.yaml`, ids only), success criteria per class (page CER, table TEDS, layout block F1, reading order, KIE field F1 and Q/A where a class has fields), one cross-class generic criterion, and results reported per class, per source and per condition. Synthetic pages are a separately labelled slice and never in the headline.
- **Data, by role:**
  - real evaluation data first: ThaiOCRBench (run `2026-09-22-a`), and a born-digital Thai PDF corpus (government, annual reports, university documents, invoices) whose text layers give free ground truth through `pdf_repair` and a class-neutral usability check, degraded into `clean`/`scan_low`/`photo`;
  - open structure sets (DocLayNet, OmniDocBench, PubTabNet/FinTabNet, FUNSD/CORD/SROIE/XFUND) for language-independent stages only: layout, tables, form structure;
  - synthetic documents rendered from HTML templates with Thai fonts, for training, fixtures and rare cases;
  - a manual anchor set of 30–50 pages per new class.
  Every source carries a recorded licence, and every eval file is excluded from every training set.
- **Harness:** new dataset kinds (`pdfcorpus`, `structure_set`, `synthetic`), a `doc_class` manifest column, new scorer pairs (text-layer CER for non-statement pages, layout block F1, reading order), a class × source × condition slice, and the docparse adapter mapped to every harness task.
- **TrOCR:** the leak rule covers every class's eval list; the gate reports per class.

Non-goals:
- Handwriting, forms filled by hand, and documents without a text region.
- Fine-tuning the layout VLMs or the extraction LLM.
- Training a layout model: layout stays a bound registry model.
- Re-opening `configs/docparse/eval_files.v1.yaml`. It stays frozen; the statement list evolves only through `add-docparse-system` task 1.4 (v2).
- A labelling UI. Anchor labels are JSON files validated by a schema.
- Collecting or redistributing any dataset whose licence forbids it.

## Capabilities

### New Capabilities

- `synthetic-documents`: rendering class-shaped synthetic Thai documents from templates, with ground truth, provenance and the rule that keeps them out of the headline.
- `manual-anchor-sets`: the per-class manual anchor set, its label format, its pre-fill rule and its validation.

### Modified Capabilities

All of these are still unarchived deltas (`add-docparse-system`, `add-ocr-benchmark-harness`), and `openspec/specs/` is empty. This change therefore writes **ADDED** requirements with non-colliding names, and names the requirement each one supersedes. Task 8.1 converts them to `MODIFIED`/`REMOVED` deltas once the parent changes are archived.

- `document-registry`: variants replace banks; the field-type, validator, alias-provider and column catalogues; the first classes; `unknown` behaviour. Supersedes "Bank statement class in v1" (in part) and "Conditional required fields per bank".
- `document-classification`: variant detection. Supersedes "Bank detection for statements".
- `field-verification`: class-generic cross validators and variant-conditional fields.
- `query-extraction`: the generic table view and class-declared query types. Supersedes "Query-type plans" (in part).
- `line-recognition`: per-class gate breakdown and per-class reader override.
- `trocr-thai-training`: leak rule over every class's eval list. Supersedes "No eval leakage".
- `docparse-evaluation`: per-class eval lists, per-class and generic success criteria, reporting rules, docparse scored on every harness task. Supersedes "Frozen eval file list" and "Success criterion" (which stay true for `bank_statement`).
- `dataset-preparation`: the `pdfcorpus`, `structure_set` and `synthetic` dataset kinds, class-neutral text-layer usability, the source licence record, the `doc_class` manifest column.
- `accuracy-scoring`: new scorer pairs, layout and reading-order metrics, class × source slices.

## Impact

- **Code (docparse):** `docparse/registry.py` (variants, catalogues, document view), `docparse/schemas.py` (`variant`, `doc_class` in reports), `docparse/config.py` (per-class overrides, eval section), `docparse/eval_files.py` (per-class lists, union for leak checks), `docparse/classify.py`, `docparse/verify.py`, `docparse/extract/*` (generic table view, class query types), `docparse/recognize/` gate breakdown, `docparse/trocr/data.py` (leak check over all lists).
- **Code (ocr_bench):** `ocr_bench/data/` gains `pdfcorpus.py`, `structure_set.py`, `synthetic/`, `text_layer.py` (class-neutral usability) and `anchor.py`; `ocr_bench/config.py` gains the three dataset kinds and a `sources` licence file; `ocr_bench/schemas.py` gains `doc_class`, a `layout` ground-truth kind and new `Source` values; `ocr_bench/metrics/` gains `layout.py` and `reading_order.py` and new dispatch pairs; `ocr_bench/metrics/aggregate.py` gains the class × source slice; `ocr_bench/models/docparse.py` (the adapter from `add-docparse-system` 3.7) maps every task. Every existing dataset and model YAML still loads, and statement scoring is unchanged.
- **Configs:** `configs/docparse/classes/{invoice,government_form,tabular_report}.yaml`, `configs/docparse/variants/banks.yaml` (moved from the statement class's `banks_from`), `configs/docparse/eval/<class>.v<N>.yaml`, `configs/datasets/{thaipdf,structure_sets,synthetic}.yaml`, `configs/datasets/sources.yaml` (licences), `configs/docparse/classify_labels.yaml` (ThaiOCRBench label → class map).
- **Dependencies:** `weasyprint` (HTML → PDF for synthetic pages), the OFL fonts Sarabun, Noto Sans Thai and Kanit added to the dev container beside `fonts-thai-tlwg`; `pdfplumber` is already present for table extraction.
- **Data:** new corpora live under `data/<corpus>/` (gitignored, DVC remote). Committed files hold ids, URLs, hashes and licences only. Open structure sets are downloaded by `prepare` from their public hosts and never committed. Synthetic pages are regenerated from seeds, so only the templates are committed.
- **Compute:** baselines and docparse on the new corpora, the per-class TrOCR gate, and the open-set runs are batched into H100 sessions (runbook §0b style). Everything else runs on the dev box with recorded replies and synthetic fixtures.
- **Sequencing:** `docparse/classify.py`, `verify.py` and `extract/` are still stubs. Groups 1–2 of this change define the generic contracts they will be built against, so landing them before `add-docparse-system` tasks 3.1, 5.1 and 6.1 avoids a rewrite. The statement eval list, gate and success criterion are untouched.
