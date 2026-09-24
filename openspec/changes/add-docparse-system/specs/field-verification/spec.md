# Spec Delta

## Purpose

Checks a parsed document against its class's required fields and, within a bounded budget, recovers missing or invalid values by cropping and re-reading the relevant region, never inventing a value.

## ADDED Requirements

### Requirement: Field states
The verifier SHALL assign each field of the document's class exactly one state:
- `found`: present in the parse and valid;
- `recovered`: filled by the verifier under the recovery guards;
- `absent`: the region shows no text for it;
- `unverified`: a candidate exists but could not be confirmed;
- `not_required`: its `required_when` condition is false.

The field report SHALL be derived from the final HTML. It SHALL NOT be kept as separate state.

#### Scenario: Complete statement
- **WHEN** every required field is present and passes its validators
- **THEN** all fields are `found`, and no crop reads are made

### Requirement: Recovery guard
A value SHALL become `recovered` only if both of these hold:
- the focused reading of the crop by the `verifier_reader` role model and the line-reader reading of the crop agree after field normalization;
- the field's validators pass.

A candidate from one reader only SHALL leave the field `unverified`, SHALL be recorded in the audit trail, and SHALL NOT be written into the field.

#### Scenario: Single-reader candidate
- **WHEN** the VLM reads "45,210.00" for the closing balance but the line reader reads "45,21O.00"
- **THEN** the field stays `unverified`, and the HTML is not patched

#### Scenario: Agreed recovery
- **WHEN** both readers read "45,210.00", and the running-balance check passes with that value
- **THEN** the field becomes `recovered`, and the HTML is patched

### Requirement: Absent detection
Before any VLM call on a crop, the verifier SHALL run text detection on the crop. If the crop contains no text, the field SHALL be marked `absent` for that region.

#### Scenario: Empty region
- **WHEN** the region hint for opening balance contains no detected text
- **THEN** no VLM call is made for that region, and the verifier moves to the next region

### Requirement: Region ladder and budgets
For each field, every retry SHALL use a new region, taken in this order:
1. the class's region hint;
2. the block with the nearest matching label;
3. the whole page at higher resolution.

When the ladder is exhausted, the verifier SHALL stop for that field. The verifier SHALL stop for the whole document when either limit below is reached, and SHALL report which limit ended the loop:
- at most `max_retries` per field (default 3);
- at most `max_vlm_calls` per document (default 12).

#### Scenario: Budget exhausted
- **WHEN** 12 VLM calls have been made and two fields are still missing
- **THEN** both fields are reported with their current state, and the report says `budget: max_vlm_calls`

### Requirement: Consistent patching and audit
Patches SHALL modify elements by block id only. A patched element SHALL keep its previous text in `data-prev` and SHALL set `data-src="verify"`. After every patch, the verifier SHALL re-run the class's cross-field validators. It SHALL revert the patch if they now fail. Every crop, prompt, reply and decision SHALL be persisted in the document's artifacts.

#### Scenario: Patch breaks arithmetic
- **WHEN** a recovered opening balance makes the running-balance check fail
- **THEN** the patch is reverted, the field becomes `unverified`, and the audit records the reason
