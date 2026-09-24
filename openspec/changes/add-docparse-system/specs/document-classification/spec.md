# Spec Delta

## Purpose

Assigns each uploaded document a registry class with a confidence, so the verifier knows which fields to require and extraction knows which plan templates apply.

## ADDED Requirements

### Requirement: Document class assignment
The system SHALL assign each document exactly one registry class id, or `unknown`, with a confidence in [0, 1]. The system SHALL classify from the first two pages. A single-page document SHALL be classified from its one page. The `classifier` role model SHALL be offered only the registry's class ids and descriptions.

#### Scenario: Statement classified
- **WHEN** a three-page bank statement is uploaded
- **THEN** the classification artifact records `bank_statement`, a confidence, the page indices used and the classifier model

#### Scenario: Out-of-registry document
- **WHEN** a document matches no registry class
- **THEN** the class is `unknown`, parsing continues without required-field verification, and the field report states that verification was skipped

### Requirement: Invalid classifier output handling
The system SHALL treat classifier output that is not a registry class id as `unknown`, SHALL NOT raise, and SHALL keep the raw reply in the artifact.

#### Scenario: Classifier replies with prose
- **WHEN** the classifier returns "This looks like a bank document"
- **THEN** the class is `unknown`, and the raw reply is stored with an `invalid_reply` flag

### Requirement: Bank detection for statements
For `bank_statement` documents, the system SHALL record the detected bank, drawn from the banks known to the registry, or `unknown`. Conditional required fields SHALL use this value.

#### Scenario: Bank detected
- **WHEN** a KBank statement is classified
- **THEN** the artifact records `bank: kbank`
