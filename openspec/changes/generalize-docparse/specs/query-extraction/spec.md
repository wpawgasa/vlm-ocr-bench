# Spec Delta

## ADDED Requirements

### Requirement: Generic typed table view
The extractor's `get_table` and `table_query` tools SHALL operate on a typed table view built from the parsed document's tables for any class. For a `table` field with declared columns, columns SHALL be matched to the header row by the field's column aliases. For a field with inferred columns, or for a document without a declared table, every header cell SHALL become a column whose type is inferred from its cells. Every row SHALL carry a row id and a block id, and every cell SHALL be normalized by its column type. An answer that aggregates over inferred columns SHALL carry the flag `inferred_columns`.

#### Scenario: Invoice line items aggregated
- **WHEN** the query is "Total of line items over 1,000 baht" on an invoice
- **THEN** `table_query` filters the `amount` column, sums it, and the answer cites the contributing row ids

#### Scenario: Unknown document table
- **WHEN** an `unknown` document contains a table whose header reads "ชื่อ | จำนวน | ราคา"
- **THEN** `get_table` returns three columns with inferred types, and an aggregate over "ราคา" carries `inferred_columns`

### Requirement: Query types from the class catalogue
The extractor SHALL route each query only into the query types the document's class declares, each from the shared catalogue: `field`, `table_filter_aggregate`, `lookup_by_key`, `cross_check`, `find` and `free_form`. Template parameters SHALL be validated against the class's bindings. `free_form` SHALL remain the flagged fallback. For `bank_statement`, the former `balance_on_date` and `reconciliation` types SHALL be served by `lookup_by_key` and `cross_check` with the same answers and provenance.

This supersedes "Query-type plans" of `add-docparse-system` in its list of types.

#### Scenario: Lookup by key on a statement
- **WHEN** the query is "What was the balance on 15 March 2026?" on a parsed statement
- **THEN** the plan is `lookup_by_key` over `transactions` with key `date`, and the answer is the balance cell with its row id, status `verified`

#### Scenario: Query type not declared by the class
- **WHEN** the query is "Does it reconcile?" on a `government_form`, whose class declares no `cross_check`
- **THEN** the query is routed to `free_form`, the answer carries the `free_form` flag, and no cross validator is run

#### Scenario: Field query on an unknown document
- **WHEN** the query is "What is the invoice number?" on an `unknown` document
- **THEN** `get_field` is not available, the plan uses `find`, and the answer is `unverified` unless the cited block contains the value
