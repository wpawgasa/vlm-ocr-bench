# Spike B: reclaiming statement text layers

**Task:** docparse 1.2 (OpenSpec `add-docparse-system`). **Date:** 2026-09-24.
**Result:** 70 of the 103 unusable digital pages are recovered. **27 → 97** digital pages now have a usable text layer. The other 33 cannot be mapped from the PDF.

This note gives counts and file ids only, with no statement content.

## Before

On run `2026-09-22-a`, 103 of the 130 digital pages failed `text_layer_usable` (`ocr_bench/data/bankstmt.py`). All of them come from four files:

| File | Bank | Pages | Font | ToUnicode | Embedded font |
|---|---|---|---|---|---|
| `ktb-en-03` | ktb | 49 | subset DBAdmanX, CID TrueType, Identity-H | identity: every code maps to itself | has `cmap` and `post` glyph names |
| `ktb-th-01` | ktb | 21 | the same | identity | has `cmap` and `post` glyph names |
| `bay-en-02-note` | krungsri | 3 | anonymous subset, CID TrueType, Identity-H | arbitrary codes ≤ U+0077 | no `cmap`, no glyph names |
| `bay-th-02-note` | krungsri | 30 | the same | the same | the same |

None of the four has PUA glyphs, which the task had suggested as a possible cause. Each one extracts only ASCII: glyph ids, plus control characters in the KTB files.

## Cause 1, KTB (fixed): identity ToUnicode over CID == GID

The CIDs are glyph ids (`CIDToGIDMap` absent, meaning Identity), and the ToUnicode map sends each CID to the same number. So the extracted text is the glyph id read as a code point. The embedded font's own `cmap` maps each used glyph to exactly one character: 88 of 88 glyphs in `ktb-en-03` and 98 of 98 in `ktb-th-01`, with no PUA code points.

**Fix:** `ocr_bench/data/pdf_repair.py` (`repair_tounicode`). For each Type0/Identity-H font whose ToUnicode map is identity, it resolves every code through `CIDToGIDMap` and the font `cmap`. It rewrites the map in a temporary copy only when that result differs from the identity map. `prepare_statements` extracts text and words from the copy and rasterizes from the original. It reports the number of pages it repaired as `statement_text_layer_repaired`.

matplotlib-style PDFs also have an identity ToUnicode map, but there it is correct, because their CIDs are code points mapped to glyphs by a `CIDToGIDMap` stream. The resolved map equals the identity map, so they are left alone (unit test).

**Verification on the corpus:**
- All 70 KTB pages pass `text_layer_usable` and have word boxes.
- The repaired header text matches the rendered page: title, statement period and dates.
- Running balances reconcile on every consecutive transaction line: 1018 of 1018 (`ktb-en-03`) and 419 of 419 (`ktb-th-01`). KTB lists rows newest first. A wrong digit mapping could not reconcile.
- No other digital file is changed: 0 fonts repaired in the BBL, KBank and TTB files.

## Cause 2, Krungsri (not mappable)

The ToUnicode map assigns arbitrary low codes to glyphs, and the subset font has no `cmap` and no glyph names. Even the English file extracts as a substitution cipher. Nothing in the PDF links a glyph to its character. Recovering the text would mean recognizing glyph shapes, which is OCR, and OCR output cannot serve as ground truth for an OCR benchmark. These 33 pages stay excluded (`NoGT`). The fixture `_write_identity_tounicode_pdf(keep_cmap=False)` checks that such fonts are not "repaired".

## After

| Bank | Usable digital pages before | After |
|---|---|---|
| bbl | 1 | 1 |
| kbank | 22 | 22 |
| ktb | 0 | **70** |
| krungsri | 0 | 0 (33 unmappable) |
| ttb | 4 | 4 |
| **Total** | **27** | **97** |

The usable pages now meet `CorpusTargets.min_digital_pages` (40), but KTB supplies 70 of the 97. Evaluation results still have to be reported per bank (design D12).

## Follow-ups

- The prepared run `2026-09-22-a` still has the old ground truth. Its text-layer GT reflects the fix only after `prepare` is re-run.
- Task 1.3 builds the frozen eval list from these usable pages.
