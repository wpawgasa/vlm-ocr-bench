# Spec Delta

## ADDED Requirements

### Requirement: No leakage from any class's eval list
The system SHALL refuse to build a training set containing any crop or page from: a file in any class's current frozen eval list (including the statement list), a page of an open structure set's evaluation split, or a synthetic page whose seed lies in an evaluation seed range. It SHALL report every offending id and its list or split. The union of eval ids SHALL come from one loader shared with the evaluation, not from a copy.

This supersedes "No eval leakage" of `add-docparse-system`, which covered the statement list only.

#### Scenario: Government eval file in real crops
- **WHEN** the real-crop source includes a file listed in `configs/docparse/eval/government_form.v1.yaml`
- **THEN** the build fails naming that file and the list, and no training set is written

#### Scenario: Synthetic eval seed in training
- **WHEN** a synthetic training page's seed is inside the frozen evaluation seed range of its template
- **THEN** the build fails naming the template, the seed and the range

#### Scenario: Open-set evaluation split
- **WHEN** a PubTabNet validation image is included in the training crops
- **THEN** the build fails naming the set and split
