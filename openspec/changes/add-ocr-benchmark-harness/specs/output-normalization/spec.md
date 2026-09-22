# Spec Delta

## Purpose

Turns each model's raw output (markdown, HTML or JSON) into one normalized page structure, and canonicalizes text and field values identically on the prediction and ground-truth sides so metrics compare like with like.

## ADDED Requirements

### Requirement: Normalized page structure
Every prediction SHALL be parsed into a normalized page with these parts:

- Reading-order plain text.
- Blocks, each with a type in {text, table, figure, header, footer}, an optional bbox `[x0,y0,x1,y1]` and a page number.
- Tables as rows × cells.
- For KIE and statement tasks, fields, each with a normalized value, the raw string, an optional confidence and an optional bbox.
- For classification, a label.

Metrics SHALL consume only normalized pages, never raw model text.

#### Scenario: Unparseable output
- **WHEN** a model returns output that its parser cannot interpret for the task (e.g. invalid JSON for KIE)
- **THEN** the normalized page keeps the plain text, has empty fields, and records `parse_error`, and the prediction still reaches scoring

#### Scenario: Table extraction from HTML
- **WHEN** a model emits an HTML table
- **THEN** the normalized page contains that table as a rows × cells list, preserving the cell text

### Requirement: Text canonicalization
Before scoring, text on both sides SHALL be canonicalized the same way: Unicode NFC, Thai digits converted to Arabic digits, zero-width characters removed, and whitespace runs collapsed. The raw string SHALL be kept alongside the normalized one.

#### Scenario: Thai digits and zero-width
- **WHEN** the text "๑๒๓​ 45" is canonicalized
- **THEN** the result is "123 45"

### Requirement: Field value normalization
Field values SHALL be normalized by type, identically for prediction and ground truth:

- **Amounts:** currency markers `฿` and `บาท` and thousands separators removed, then parsed as decimals.
- **Dates:** parsed to ISO `YYYY-MM-DD`, with Buddhist-era years (year > 2400) converted by subtracting 543.
- **Account numbers:** reduced to digits only.
- **Names:** text-canonicalized.

Unparseable values SHALL be kept as canonical text and flagged. They SHALL NOT be dropped.

#### Scenario: Buddhist-era date
- **WHEN** the date "15/03/2568" is normalized
- **THEN** the value is "2025-03-15"

#### Scenario: Thai amount string
- **WHEN** the amount "฿152,340.75 บาท" is normalized
- **THEN** the value is the decimal 152340.75 and the raw string is preserved

#### Scenario: Account number
- **WHEN** the account number "123-4-56789-0" is normalized
- **THEN** the value is "1234567890"

### Requirement: Table format conversion
Tables SHALL be represented as HTML before scoring, whatever form the model emitted them in. OTSL output (TeleOCR) SHALL be converted to HTML, preserving row and column spans. Markdown pipe tables SHALL be converted to HTML.

#### Scenario: OTSL with a merged cell
- **WHEN** TeleOCR emits `<fcel>A<lcel><nl><fcel>1<fcel>2<nl>`
- **THEN** the normalized table is HTML whose first row has one cell `A` with `colspan="2"`, followed by a row with cells `1` and `2`

### Requirement: Statement table construction
For the `statement` task, no model SHALL be asked for a statement schema. The statement page SHALL be derived from the model's full-page layout parse, using its table blocks (as HTML) and text blocks, by one rule-based mapper shared by all models. The derived statement page has these fields: bank, account number, account name, period start/end, opening/closing balance, page number, and rows. Each row has date, description, debit, credit, balance, and optional channel and bbox. Pages of one file SHALL be merged into a statement file in page order.

#### Scenario: Merge pages
- **WHEN** a 4-page statement is normalized
- **THEN** a single statement file is produced whose rows are the concatenation of page rows in page order, each row tagged with its source page

#### Scenario: Model-independent mapping
- **WHEN** two models produce the same table HTML and text blocks for a page
- **THEN** the derived statement pages are identical
