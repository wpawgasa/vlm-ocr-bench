# Spec Delta

## Purpose

Renders class-shaped synthetic Thai documents from templates with ground truth and full provenance, for training, test fixtures and rare-case coverage, while keeping them out of every headline result.

## ADDED Requirements

### Requirement: Template rendering with provenance
The system SHALL render synthetic pages from per-class templates filled by seeded generators, using the configured Thai fonts, to PDF and then to page images under the three benchmark conditions. Every page SHALL record its template id, seed, font, class, variant and condition. Given the same template, seed and font, rendering SHALL be deterministic. A template edited after being used for a frozen evaluation slice SHALL get a new template id, and the renderer SHALL refuse a template whose content hash differs from the one recorded under its id.

#### Scenario: Deterministic render
- **WHEN** template `invoice/tax_invoice_a` is rendered twice with seed 17 and font Sarabun
- **THEN** the two PDFs' text layers and field JSON are identical

#### Scenario: Edited template
- **WHEN** a template file changes but keeps its id
- **THEN** rendering fails naming the template id and the recorded hash

### Requirement: Ground truth from the generator
Every synthetic page SHALL carry ground truth produced by the generator itself, not by any reader: the full text in reading order, tables as HTML with cell values, the class's fields with typed values, and word boxes from the rendered PDF's text layer. Field values SHALL be consistent with the class's cross validators by construction.

#### Scenario: Invoice fields reconcile
- **WHEN** a synthetic invoice is generated
- **THEN** its line items sum to its subtotal and `line_items_sum` passes on its ground truth

### Requirement: Seed ranges and the synthetic slice
Each template SHALL declare disjoint seed ranges for training and evaluation. Evaluation seeds SHALL be frozen with the class's eval list. Synthetic evaluation pages SHALL be reported only in the synthetic slice and SHALL never enter a headline aggregate or a per-class verdict.

#### Scenario: Overlapping ranges refused
- **WHEN** a template declares training seeds 0–999 and evaluation seeds 900–1099
- **THEN** loading the template fails naming the overlap

#### Scenario: Synthetic verdict excluded
- **WHEN** the invoice evaluation includes synthetic pages
- **THEN** the invoice verdict is computed from real sources only

### Requirement: Synthetic pages as committed fixtures
Synthetic pages SHALL be the only document images that tests may commit. A committed fixture SHALL record its template id and seed, and SHALL contain no text taken from a real document.

#### Scenario: Fixture provenance
- **WHEN** a fixture page is added under the test fixtures
- **THEN** a sidecar names its template id and seed, and a test regenerates and compares it
