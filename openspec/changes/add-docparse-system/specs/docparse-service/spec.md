# Spec Delta

## Purpose

Exposes parsing and extraction through a CLI and an HTTP API over a per-document artifact directory, with explicit status and no silent drops. Binds each model role of the pipeline to an entry of the model registry, so a role's model is switched in configuration.

## ADDED Requirements

### Requirement: CLI commands
The system SHALL provide these commands:
- `docparse parse <path>... [--config]`: writes each document's artifacts and HTML;
- `docparse extract --doc <id> --query <text>`: writes the answer JSON;
- `docparse gate`: runs the TrOCR promotion gate.

A command SHALL exit non-zero if any input ended with `error` status.

#### Scenario: Batch parse
- **WHEN** `docparse parse` is run on 3 PDFs, and one of them cannot be opened
- **THEN** the two readable PDFs are parsed, the third has an `error` status record, and the command exits non-zero

### Requirement: HTTP API
The system SHALL provide an HTTP API with these endpoints:
- `POST /parse`: accepts an uploaded PDF or image and returns the document id, class, status, field report and HTML;
- `POST /extract`: accepts a document id and a query and returns the extraction answer;
- `GET /documents/{id}`: returns the stored result.

Invalid uploads (an unsupported type, or a file over the configured size limit) SHALL be rejected with a 4xx response and a reason.

#### Scenario: Upload and parse
- **WHEN** a statement PDF is posted to `/parse`
- **THEN** the response contains a document id, `bank_statement`, `status: complete|partial`, the field report and the HTML

#### Scenario: Unsupported upload
- **WHEN** a `.docx` file is posted to `/parse`
- **THEN** the response is 415 with a reason, and no document is created

### Requirement: Per-document artifacts
Each document SHALL have its own artifact directory, with one typed artifact per stage:
- `classify`, `layout`, `segments`, `readings`, `reconcile`;
- `verify` audit;
- `parsed.html`;
- `fields.json`;
- `status.json`.

Re-running a stage SHALL reuse the upstream artifacts. Document status SHALL be one of:
- `complete`: all required fields found or recovered;
- `partial`: a page error, or a required field `unverified` or `absent`;
- `error`.

#### Scenario: Re-run verification only
- **WHEN** the verifier config changes, and the verify stage is re-run for a document
- **THEN** classify, layout, segment, recognize and reconcile are not called again

### Requirement: Model roles bound to the model registry
Every model that docparse calls over HTTP SHALL be named by a role in `configs/docparse/pipeline.yaml`. The roles are `layout`, `classifier`, `verifier_reader` and `llm`. Each role SHALL name an entry of the `ocr_bench` model registry (`configs/models/<name>.yaml`), built with `build_model`. A role's prompts, sampling and chat-template settings SHALL come from that entry's `roles.<role>` section, not from docparse code.

At startup, the system SHALL refuse a binding, naming the role and the model, when:
- the entry does not exist;
- the entry has no `roles.<role>` section;
- the role is `layout` and the model has no docparse layout adapter.

Each stage artifact SHALL record the bound model name, its `served_model_name`, the role prompt version and the vLLM version. The resolved entry SHALL be part of the stage's config hash.

#### Scenario: Swapping the classifier
- **WHEN** `roles.classifier` is changed from `qwen3vl` to another registry entry that has a `roles.classifier` section
- **THEN** classify re-runs for each document with the new model, its artifact records the new model name and prompt version, and layout is not re-run

#### Scenario: Entry lacks the role
- **WHEN** `roles.llm` names `dotsocr`, which has no `roles.llm` section
- **THEN** `docparse parse` exits non-zero before any document is processed, and the error names `llm` and `dotsocr`
