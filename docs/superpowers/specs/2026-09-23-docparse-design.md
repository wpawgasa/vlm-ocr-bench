# docparse: multi-model document parsing and extraction

**Date:** 2026-09-23. **Status:** design approved; implementation not started.
**Source of truth:** the OpenSpec change [`add-docparse-system`](../../../openspec/changes/add-docparse-system/). This page is the one-screen overview. The decisions (D1–D12), requirements and tasks live there.

## Problem

No single Thai-capable VLM parses bank statements reliably. On run `2026-09-22-a`:
- dots.ocr loops on 9.9% of rows;
- typhoon-ocr1.5 and PaddleOCR-VL do not follow instructions;
- none of them gives trustworthy field confidence.

docparse orchestrates several models so that every field and answer is complete, or else explicitly `unverified`/`absent`, and always traceable.

## Architecture

```
upload ─► classify ─► layout ─► segment ─► recognize ─► reconcile ─► verify ─► parsed.html + fields.json
          Qwen3-VL    Paddle     PP-OCRv5   paddle_crop   alignment    agent:
                      pipeline   DB det     | Thai TrOCR  voting,      crop, re-read,
                      (dots alt)            (if gated in) validators   2 readers agree
                                                                       + validators

query + parsed doc ─► route query type ─► plan template ─► tools ─► verify ─► answer + provenance
                      Qwen3-8B            (free-form       over      re-match /
                                           fallback)       parse     recompute
```

- **Pipeline 1, parsing.** Each stage writes a typed, cached artifact per document (`runs/docparse/<doc_id>/`). Only the verifier is agentic, with at most 3 retries per field and 12 VLM calls per document.
- **Pipeline 2, extraction.** A small LLM routes the query to a plan template, fills its parameters, and calls tools over the parse. Read values are re-matched in the cited block; computed values are recomputed in `Decimal`. If verification fails, the answer is `unverified`/`absent` rather than a guess.
- **Thai TrOCR.** A TrOCR printed encoder at 96×768 plus a character-level Thai decoder, trained on synthetic lines and non-eval statement crops. It replaces `paddle_crop` only if it passes the gate: CER ≤ 2% clean, ≤ 6% degraded, and no worse than `paddle_crop`.
- **Service.** `docparse parse | extract | gate` CLI, and a FastAPI app (`POST /parse`, `POST /extract`).

## Key decisions and why

| Decision | Why |
|---|---|
| Staged pipeline, agents only in verify and extract | Reproducible, scoreable per stage; rejected: a single orchestrator agent, or a workflow engine |
| Separate instruction-following VLM (Qwen3-VL-8B) for classify and verifier reads | The OCR VLMs cannot follow free-form prompts (457 of 1,149 KIE replies were not JSON) |
| Paddle layout by default; never dots `layout_only` | No loops, table HTML; reconciliation needs a second text reading |
| Alignment voting instead of comparing confidences | Model confidences are not on a common scale; Paddle returns none |
| A recovery needs 2 agreeing readers and a passing validator | Prevents the verifier from inventing values |
| TrOCR gated against `paddle_crop` | Its value is unproven; the reader interface keeps the fallback cheap |
| Frozen, file-level eval list; paired bootstrap per condition and bank | Only 27 text-layer pages exist (21 from one file), so leakage would make the result meaningless |

These decisions were reviewed adversarially by Fable 5.1 on 2026-09-23. The user accepted all of its recommendations and kept the FastAPI service.

## Success criterion

docparse is declared better than the best single VLM for a given condition only if both hold:
- the lower bound of the 95% CI of the paired field-F1 difference is above 0;
- the upper bound of the CER-difference CI is below 0.

Results are reported per bank, with the Check B consistency rate over all 458 statement pages.
