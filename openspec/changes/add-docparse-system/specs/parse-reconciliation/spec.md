# Spec Delta

## Purpose

Merges the layout text and line-reader text into one reading per segment and emits the parsed document as HTML that keeps geometry, confidence and provenance.

## ADDED Requirements

### Requirement: Alignment voting
For each segment, the system SHALL align the line-reader text with the corresponding span of the layout block text at character level:
- Agreed spans SHALL be accepted.
- For a disputed span, the system SHALL first try the field-type validator on each candidate. If exactly one candidate passes, it wins.
- If the dispute remains, the system SHALL request a third reading of that segment only, and SHALL take a 2-of-3 majority.
- If there is still no majority, the system SHALL keep the line-reader text and mark the segment disputed.

Raw confidences from different models SHALL NOT be compared directly.

#### Scenario: Readers agree
- **WHEN** the layout text and the line reader both read "12,500.00"
- **THEN** the segment text is "12,500.00" and is not disputed

#### Scenario: Validator breaks the tie
- **WHEN** an amount segment reads "12,5O0.00" from one source and "12,500.00" from the other
- **THEN** "12,500.00" wins, because only it parses as an amount

#### Scenario: Unresolved dispute
- **WHEN** three readings disagree pairwise
- **THEN** the line-reader text is kept, and the element carries `data-disputed` and `data-alt` with the other readings

#### Scenario: Truncated layout text
- **WHEN** the block's layout text is flagged `truncated`
- **THEN** the line-reader text is used without voting, and the segment is marked `single-source`

### Requirement: Parsed HTML contract
The system SHALL emit one HTML document per input document. Content SHALL be in reading order and grouped by page. Every content element SHALL carry `data-page`, `data-bbox` (x0,y0,x1,y1 in page pixels), `data-block-id`, `data-src` (the reader or readers that produced it) and `data-conf`. Tables SHALL be `<table>` elements, with row and column spans where the layout gives them. The same input and configuration SHALL produce byte-identical HTML.

#### Scenario: Element provenance
- **WHEN** the HTML for a statement is inspected
- **THEN** every `<p>`, heading and `<td>` has page, bbox, block id, source and confidence attributes

#### Scenario: Deterministic output
- **WHEN** the same document is parsed twice from cached stage artifacts
- **THEN** the two HTML files are byte-identical
