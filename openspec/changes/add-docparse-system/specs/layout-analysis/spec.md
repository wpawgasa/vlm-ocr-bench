# Spec Delta

## Purpose

Produces the page layout, meaning blocks with bounding boxes, categories, reading order and a first text reading, that all later stages anchor to.

## ADDED Requirements

### Requirement: Layout blocks with a text reading
For each page, the system SHALL produce layout blocks. Each block SHALL have:
- a stable block id;
- a page index;
- a bbox in page pixel coordinates;
- a category (`text`, `title`, `table`, `figure`, `header`, `footer`, `other`);
- a reading-order index;
- the backend's text reading for the block, with table blocks read as HTML.

A backend mode that returns boxes without text SHALL NOT be used.

#### Scenario: Default backend produces blocks and text
- **WHEN** a statement page is parsed with the default layout backend
- **THEN** every block has a bbox, a category, an order index, and non-empty text or table HTML, unless the block is a figure

### Requirement: Pluggable layout backends
The layout backend SHALL be the model bound to the `layout` role (docparse-service, "Model roles bound to the model registry"). `paddleocr_vl` (the PaddleOCR-VL pipeline) SHALL be the default and `dotsocr` (dots.ocr full layout) the alternative. Only a registry model with a docparse layout adapter SHALL be accepted for this role. Every adapter SHALL produce the same block schema.

#### Scenario: Switching backend
- **WHEN** `roles.layout` is set to `dotsocr`
- **THEN** the layout artifact has the same schema, and records `model: dotsocr` with its prompt version

### Requirement: Truncated backend output
When the backend reply was truncated at the token limit, the system SHALL mark the affected page's block text `truncated`. Reconciliation SHALL treat truncated text as absent. The blocks' boxes SHALL be kept.

#### Scenario: Repetition loop hits max tokens
- **WHEN** the dots.ocr reply for a page ends with `finish_reason=length`
- **THEN** the blocks keep their bboxes, and their text carries the `truncated` flag, so that downstream stages rely on the line reader

### Requirement: Layout failure is explicit
If the layout backend fails after its retry, the system SHALL record a page-level error and SHALL continue with the other pages. The document status SHALL be `partial`.

#### Scenario: One page fails
- **WHEN** layout fails for page 2 of 3
- **THEN** pages 1 and 3 are parsed, page 2 has an error record, and the document status is `partial`
