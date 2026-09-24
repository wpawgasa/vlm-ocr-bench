# Spec Delta

## Purpose

Defines the document classes docparse knows and, for each, the fields a complete parse must contain, so that classification, verification and extraction share one source of truth.

## ADDED Requirements

### Requirement: Class definitions from configuration
The system SHALL load document classes from configuration files. Each class SHALL have:
- a unique id;
- a description used for classification;
- a list of field definitions.

Each field definition SHALL have a name, a value type (`text`, `amount`, `date`, `account`, `table`), label aliases in Thai and English, an optional region hint, a list of validators and a `required_when` condition.

The system SHALL refuse to start when a class file is invalid, and SHALL name the file and the offending key.

#### Scenario: Valid registry loads
- **WHEN** the configured class directory contains a valid `bank_statement` definition
- **THEN** the registry exposes `bank_statement` with its fields, aliases, validators and region hints

#### Scenario: Invalid field type rejected
- **WHEN** a class file declares a field with type `currency_code`
- **THEN** loading fails with an error naming the file and the field

#### Scenario: New class without code changes
- **WHEN** a new class file `invoice.yaml` is added to the class directory
- **THEN** classification and verification consider `invoice` without any code change

### Requirement: Bank statement class in v1
The registry SHALL ship a fully specified `bank_statement` class with:
- **header fields:** bank, account number, account name, statement period start and end, opening balance, closing balance;
- **a transaction table field:** date, description, debit, credit, balance;
- **validators:** date and amount parsing, and running-balance arithmetic consistency.

The label aliases for these fields SHALL be the same ones the `ocr_bench` statement mapper uses. They SHALL NOT be maintained as a second copy.

#### Scenario: Aliases shared with the benchmark mapper
- **WHEN** a new Thai header alias is added to the statement mapper's label table
- **THEN** the `bank_statement` class recognizes that alias without editing the class file

### Requirement: Conditional required fields per bank
A field's `required_when` SHALL be able to depend on the detected bank. A field whose condition is false for a document SHALL NOT be reported missing.

#### Scenario: Bank that prints no opening balance
- **WHEN** a KBank statement is parsed and the class marks opening balance as not required for `kbank`
- **THEN** the field report does not list opening balance as missing, and no verifier retries are spent on it
