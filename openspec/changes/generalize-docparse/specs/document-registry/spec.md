# Spec Delta

## ADDED Requirements

### Requirement: Class variants replace banks
A class MAY declare a variant list (`variants_from`), with optional per-variant overrides. A field's `required_when` SHALL be one of `always`, `never`, `variant_in: [...]` or `variant_not_in: [...]`. Variant names in a condition SHALL be members of the class's variant list, and the registry SHALL refuse a class file that names an unknown variant, the file and the field. A field with `required_when: never` is optional: it SHALL be looked up and reported when found, and SHALL NOT be reported missing or recovered.

The `bank_statement` class's variants SHALL be its banks, and the bank-keyed overrides it needs SHALL be reachable only through its variant file. The registry SHALL refuse the former keys `banks_from`, `bank_in` and `bank_not_in` with a message naming their replacements.

This supersedes "Conditional required fields per bank" of `add-docparse-system`.

#### Scenario: Statement opening balance by variant
- **WHEN** `bank_statement.yaml` marks `opening_balance` with `variant_not_in: [kbank]`, and a document's variant is `kbank`
- **THEN** the field report lists `opening_balance` as `not_required`

#### Scenario: Unknown variant in a condition
- **WHEN** a class file uses `variant_in: [acme]` and `acme` is not in its variant file
- **THEN** loading fails with an error naming the class file, the field and `acme`

#### Scenario: Old bank keys refused
- **WHEN** a class file still uses `banks_from`
- **THEN** loading fails with an error naming the file and `variants_from`

### Requirement: Field type and validator catalogue
Field types SHALL be `text`, `amount`, `number`, `percent`, `date`, `identifier`, `account` and `table`. The registry SHALL provide the field validators `date`, `amount`, `number`, `percent`, `account_number`, `thai_tax_id`, `non_empty`, `known_variant` and `regex:<pattern>`. A class file MAY use any validator on any field; an unknown validator name SHALL be refused naming the file and the field.

#### Scenario: Thai tax id validator
- **WHEN** an `invoice` field `vendor_tax_id` has validators `[thai_tax_id]` and the parsed value is 13 digits with a valid checksum
- **THEN** the value passes; a 13-digit value with a wrong check digit fails

#### Scenario: Parameterized regex validator
- **WHEN** a field declares `validators: ["regex:^[A-Z]{2}-\\d{6}$"]`
- **THEN** `AB-123456` passes and `AB-12345` fails

### Requirement: Alias providers and column specs
A field's aliases SHALL come from an inline `aliases` list, from a named provider (`aliases_from: <provider>:<key>`), or both. The provider `statement_header:<field>` SHALL resolve to the benchmark statement mapper's labels, so that the statement keeps a single copy of them. A `table` field's `columns` SHALL be either a mapping of column name to a provider reference (`statement_column:<role>`) or an inline `{type, aliases}` spec, or the literal `infer`, which takes the header row as read. An unknown provider or role SHALL be refused naming the file and the field.

#### Scenario: Inline columns for an invoice
- **WHEN** `invoice.yaml` declares `line_items` with columns `amount: {type: amount, aliases: [จำนวนเงิน, amount]}`
- **THEN** the registry exposes the column with type `amount` and both aliases, and no statement mapper table is consulted

#### Scenario: Inferred columns
- **WHEN** `tabular_report.yaml` declares `tables` with `columns: infer`
- **THEN** the class loads, and the field reports that its columns are taken from each table's header row

### Requirement: Cross validators over a document view
A cross validator SHALL take a class-neutral document view (typed fields, typed tables, variant, overrides) and SHALL return a pass or fail with a reason. The registry SHALL provide `running_balance`, `line_items_sum`, `date_order:<earlier>,<later>` and `fields_consistent:<a>=<b>`. A class SHALL declare the cross validators it uses, and the verifier and the extractor SHALL run only those.

#### Scenario: Invoice totals reconcile
- **WHEN** an invoice's line-item amounts sum to the subtotal and subtotal plus VAT equals the total within 0.01
- **THEN** `line_items_sum` passes

#### Scenario: Statement unchanged
- **WHEN** a `bank_statement` document view is validated
- **THEN** `running_balance` produces the same result as the benchmark's Check B on the same rows

### Requirement: Built-in `unknown` class
The registry SHALL provide a built-in class `unknown` that no class file may define: no fields, no variants, no cross validators, and the query types `find`, `table_filter_aggregate` and `free_form`. A document classified `unknown` SHALL still receive layout, full text and tables, and SHALL be queryable.

#### Scenario: Unknown document parsed and queried
- **WHEN** a document matches no class and is parsed
- **THEN** `parsed.html` contains its blocks and tables, `fields.json` records `verification_skipped`, and a `find` query returns matching block ids

#### Scenario: A file named unknown.yaml
- **WHEN** the class directory contains `unknown.yaml`
- **THEN** loading fails naming the file and stating that `unknown` is built in

### Requirement: Initial document classes
The registry SHALL ship `invoice`, `government_form` and `tabular_report` beside `bank_statement`:
- `invoice`: fields `vendor_name`, `vendor_tax_id` (optional), `document_no`, `issue_date`, `buyer_name` (optional), `subtotal`, `vat_amount` (optional), `total`; table `line_items` (`description`, `quantity`, `unit_price`, `amount`); variants `invoice`, `tax_invoice`, `receipt`; cross validator `line_items_sum`.
- `government_form`: fields `document_no`, `issue_date`, `issuing_agency`, `subject`, `reference_no` (optional), `valid_until` (optional); no required table.
- `tabular_report`: optional field `title`; table `tables` with inferred columns.

Each field SHALL have Thai and English aliases. Each class SHALL have a description usable by the classifier.

#### Scenario: Registry lists four classes
- **WHEN** the shipped class directory is loaded
- **THEN** the registry exposes `bank_statement`, `invoice`, `government_form` and `tabular_report`, and `unknown` as built in

#### Scenario: Invoice required fields
- **WHEN** `invoice` is loaded and the variant is `receipt`
- **THEN** its required fields are `vendor_name`, `document_no`, `issue_date`, `subtotal`, `total` and `line_items`

### Requirement: Class-declared query types and reader override
A class file SHALL declare `query_types`: the subset of the extraction template catalogue (`field`, `table_filter_aggregate`, `lookup_by_key`, `cross_check`, `find`, `free_form`) it supports, with the bindings each template needs. A class file MAY declare `line_reader` to override the pipeline's default reader for its documents. Unknown template names or bindings to undeclared fields or tables SHALL be refused naming the file.

#### Scenario: Statement query types through the catalogue
- **WHEN** `bank_statement.yaml` declares `lookup_by_key: {table: transactions, key: date, value: balance}` and `cross_check: [running_balance]`
- **THEN** the extractor answers "balance on 2026-03-15" and "does the statement reconcile?" with those templates

#### Scenario: Binding to a missing table
- **WHEN** a class declares `lookup_by_key: {table: payments, ...}` and has no `payments` field
- **THEN** loading fails naming the file and `payments`
