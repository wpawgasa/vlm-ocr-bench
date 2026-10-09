# Spec Delta

## ADDED Requirements

### Requirement: Class-generic cross validation on patch
After every patch, the verifier SHALL re-run the cross validators declared by the document's class on the class-neutral document view, and SHALL revert the patch if any now fails, recording the validator name and reason in the audit. A class with no cross validators SHALL accept the patch when the field validators pass. The verifier SHALL NOT call any statement-specific check for a non-statement class.

#### Scenario: Invoice total patched
- **WHEN** a recovered `total` on an invoice makes `line_items_sum` fail
- **THEN** the patch is reverted, the field becomes `unverified`, and the audit names `line_items_sum`

#### Scenario: Class without cross validators
- **WHEN** a recovered `document_no` on a `government_form` passes its field validators
- **THEN** the patch stands, and the audit records that no cross validator applies

### Requirement: Optional and variant-conditional fields
A field whose `required_when` is false for the document, including `never`, SHALL be reported `not_required` when absent and `found` when present and valid. No recovery attempt and no reader call SHALL be spent on it.

#### Scenario: Optional VAT amount absent
- **WHEN** an invoice prints no VAT line and `vat_amount` is `required_when: never`
- **THEN** the field is `not_required`, and the verifier makes no crop read for it

#### Scenario: Optional field present
- **WHEN** the same invoice prints a VAT line that parses as an amount
- **THEN** the field is `found` with its value and block id
