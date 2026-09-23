# Spec Delta

## Purpose

Answers a natural-language query over a parsed document with a small LLM, returning values with provenance that have been verified against the parse, or an explicit abstention.

## ADDED Requirements

### Requirement: Query input and answer contract
The extractor SHALL accept a query string and a parsed-document id. It SHALL return:
- the answer value or values, each typed as the field type;
- for each value, its provenance (block ids, page, bbox and, for tables, row ids);
- a status: `verified`, `unverified` or `absent`;
- the plan that was executed;
- a trace of tool calls.

#### Scenario: Header field query
- **WHEN** the query is "What is the closing balance?" on a parsed statement
- **THEN** the answer is an amount with its block id and bbox, and the status is `verified`

### Requirement: Tools over the parse
The extractor SHALL expose only these tools to the LLM. A tool call with invalid arguments SHALL return an error result to the LLM and SHALL NOT raise.
- `outline`: document structure;
- `find`: text or regex search returning block ids;
- `get_block`;
- `get_field`: the verified field value and its state;
- `get_table`: rows and columns;
- `table_query`: filter and aggregate over typed statement rows;
- `compute`: exact decimal arithmetic over cited values.

#### Scenario: Aggregate over transactions
- **WHEN** the query is "Total deposits in March 2026"
- **THEN** the plan uses `table_query` with a date filter and a sum over credits, and the answer cites the contributing row ids

### Requirement: Query-type plans
The extractor SHALL classify each query into a supported type:
- header field;
- table filter or aggregate;
- balance on a date;
- reconciliation check;
- free-form.

For every type except free-form, it SHALL execute that type's plan template, with parameters filled in by the LLM. A free-form plan SHALL be allowed only as a fallback, and SHALL be flagged `free_form` in the answer.

#### Scenario: Unsupported phrasing
- **WHEN** a query matches no template type
- **THEN** a free-form plan is executed, and the answer carries the `free_form` flag

### Requirement: Extraction verification
The extractor SHALL verify each value before returning it:
- **A read value** SHALL match its cited block's text after field normalization.
- **A computed value** SHALL equal the result of recomputing it, in exact decimal arithmetic, from the cited rows or fields.

A failed step SHALL be retried up to `max_step_retries` (default 2). After that, the value SHALL be returned as `unverified`. A query whose required field is `absent` in the parse SHALL be answered `absent`. It SHALL NOT be answered with a guess.

#### Scenario: Hallucinated citation
- **WHEN** the LLM returns 9,999.00 citing a block whose text does not contain that amount
- **THEN** the step is retried, and if it still fails the answer status is `unverified`

#### Scenario: Unanswerable query
- **WHEN** the query asks for the opening balance of a KBank statement that prints none
- **THEN** the status is `absent`, and no value is returned
