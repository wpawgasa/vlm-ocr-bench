# Spec Delta

## Purpose

Gives every new document class a small, human-labelled set of real pages, with one label format, a pre-fill rule that limits model bias, and validation, so that classes without text-layer ground truth still have a trustworthy anchor.

## ADDED Requirements

### Requirement: Anchor set per class
Every class other than `bank_statement` SHALL have a manual anchor set of 30 to 50 real pages, drawn to cover its variants and, where available, scanned and photographed pages. The anchor files SHALL be listed in the class's frozen eval list, and the labels SHALL be stored outside git with the corpus. The statement anchor set SHALL keep the existing review-queue format.

#### Scenario: Anchor files in the eval list
- **WHEN** the `government_form` anchor set is labelled
- **THEN** every anchor file id appears in the current `government_form` eval list with `source` and hash, and no label content is committed

### Requirement: Page label format
A label SHALL be one JSON document per page containing: the class and variant, the full text in reading order, every table as HTML, the class's fields with values (null when absent), optional bounding boxes, the labeller, and the pre-fill record. Typed field values SHALL be stored as written on the page; normalization is the scorer's job. A label SHALL validate against the label schema and the class definition.

#### Scenario: Field unknown to the class
- **WHEN** an invoice label contains a field `po_number` that the class does not define
- **THEN** validation fails naming the file and `po_number`

#### Scenario: Valid label accepted
- **WHEN** a government form label has all class fields, text, tables and a pre-fill record
- **THEN** `ocrbench anchor validate` reports it valid and the scorer can load it as `gt_kind=json`

### Requirement: Pre-fill rule
A label MAY be pre-filled by one model. That model SHALL be recorded in the label, and SHALL be none of: docparse, a model bound to a docparse role, or a model in the comparison set of that class. Pages from client sources SHALL NOT be pre-filled by an external service. A label with a pre-fill record SHALL carry the labeller's confirmation of every field and every table.

#### Scenario: Pre-filled by a comparison model
- **WHEN** a label records pre-fill by a model that is in the class's comparison set
- **THEN** validation fails naming the model and the class

#### Scenario: Pre-filled by a docparse role model
- **WHEN** a label records pre-fill by the model bound to the `verifier_reader` role
- **THEN** validation fails naming the model and the role

#### Scenario: Unconfirmed field
- **WHEN** a pre-filled label has a field without the labeller's confirmation
- **THEN** validation fails naming the field
