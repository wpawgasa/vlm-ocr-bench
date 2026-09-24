# Spec Delta

## Purpose

Splits layout blocks into line segments that a line reader can read accurately, preserving each segment's link to its block and table cell.

## ADDED Requirements

### Requirement: Segments within blocks
The system SHALL split each text-bearing block into line segments. Each segment SHALL have:
- a segment id;
- a parent block id;
- a bbox in page coordinates, contained in the parent block's bbox (with a small tolerance);
- an order index within the block.

For table blocks, each segment SHALL also carry its row and column indices.

#### Scenario: Paragraph block
- **WHEN** a text block contains three printed lines
- **THEN** three segments are produced, in top-to-bottom order, each inside the block bbox

#### Scenario: Table block
- **WHEN** a table block with 10 rows and 5 columns is segmented
- **THEN** each segment carries its row and column indices, taken from the block's table structure

### Requirement: Aspect-ratio limit
No segment SHALL exceed the configured maximum width-to-height ratio. By default, this is the active line reader's input aspect ratio: 6:1 for the muocr checkpoint's 64×384 input, and 10:1 for a reader with no fixed input size. A longer line SHALL be split at its widest internal whitespace gaps. A table cell SHALL be split only if the cell itself exceeds the limit.

#### Scenario: Long description line
- **WHEN** a detected line is 1500×30 px and the active reader is muocr (limit 6:1)
- **THEN** it is split at whitespace gaps into segments that are each at most 180 px wide, and together they cover the original line

### Requirement: Detector fallback
If the text-line detector finds no lines in a non-figure block that has text, the system SHALL fall back to a projection-profile split. The system SHALL flag the segments it produced with `fallback`.

#### Scenario: Detector misses faint print
- **WHEN** the detector returns no lines for a block whose layout text is non-empty
- **THEN** projection-profile segments are produced with the `fallback` flag
