"""`ocrbench probe`: 5 fixed statement pages through each model -> capability matrix.

Each page gets the model's `statement` plan plus one `question` request asking for JSON
fields. Every (capability, model) cell is `yes` (all pages), `partial` (some) or `no`.
"""

import asyncio
from pathlib import Path

from ocr_bench.config import RequestPlan
from ocr_bench.models.base import PlannedModel
from ocr_bench.models.runner import load_image
from ocr_bench.schemas import Condition, ManifestRow, PredictionRow, Source, Task

N_PROBE_PAGES = 5
PROBE_JSON_QUESTION = (
    "Extract the account number, account name and closing balance from this bank statement "
    "as JSON with keys account_no, account_name, closing_balance."
)
PROBE_JSON_KEYS = ("account_no", "account_name", "closing_balance")
PROBE_JSON_PLAN = RequestPlan(kind="question", version="probe-json-v1", parser="json_fields")
CAPABILITIES = [
    "reading_order_text",
    "table_rows_cells",
    "block_bboxes",
    "page_index",
    "json_fields_on_request",
    "token_logprobs",
]


def select_probe_pages(rows: list[ManifestRow], n: int = N_PROBE_PAGES) -> list[ManifestRow]:
    """Clean statement pages: the first page of each bank's first file (sorted by bank),
    topped up in sample_id order when there are fewer than `n` banks."""
    candidates = sorted(
        (r for r in rows if r.source == Source.bankstmt and r.condition == Condition.clean),
        key=lambda r: r.sample_id,
    )
    picks: list[ManifestRow] = []
    for bank in sorted({r.bank or "" for r in candidates}):
        of_bank = [r for r in candidates if (r.bank or "") == bank]
        first_pages = [r for r in of_bank if r.page_no == 1]
        picks.append((first_pages or of_bank)[0])
        if len(picks) == n:
            return picks
    chosen = {r.sample_id for r in picks}
    for row in candidates:
        if len(picks) == n:
            break
        if row.sample_id not in chosen:
            picks.append(row)
            chosen.add(row.sample_id)
    return picks


async def probe_model(
    model: PlannedModel, pages: list[ManifestRow], run_root: Path
) -> tuple[list[PredictionRow], list[PredictionRow]]:
    """(statement predictions, JSON-on-request predictions), in page order."""

    async def one(row: ManifestRow) -> tuple[PredictionRow, PredictionRow]:
        image = await asyncio.to_thread(load_image, run_root / row.image_path)
        statement = await model.predict(row, image)
        json_row = row.model_copy(update={"task": Task.kie, "question": PROBE_JSON_QUESTION})
        asked = await model.predict_with_plan(json_row, image, PROBE_JSON_PLAN)
        return statement, asked

    results = await asyncio.gather(*(one(row) for row in pages))
    return [s for s, _ in results], [j for _, j in results]


def _level(flags: list[bool]) -> str:
    if flags and all(flags):
        return "yes"
    return "partial" if any(flags) else "no"


def _json_ok(pred: PredictionRow) -> bool:
    if pred.raw.error is not None or pred.normalized.parse_error is not None:
        return False
    return sum(1 for key in PROBE_JSON_KEYS if key in pred.normalized.fields) >= 2


def capability_row(statement: list[PredictionRow], asked: list[PredictionRow]) -> dict[str, str]:
    """One model's matrix column. Error rows have an empty normalized page, so a failed
    page counts as a miss."""
    calls = [c for p in statement + asked for c in p.raw.calls if c.error is None]
    # No model emits a document page index; typhoon's <page_number> tag carries the page
    # number printed on the page, which counts as partial.
    has_page_tag = any("<page_number>" in c.text for p in statement for c in p.raw.calls)
    return {
        "reading_order_text": _level([bool(p.normalized.text.strip()) for p in statement]),
        "table_rows_cells": _level([bool(p.normalized.tables) for p in statement]),
        "block_bboxes": _level(
            [any(b.bbox is not None for b in p.normalized.blocks) for p in statement]
        ),
        "page_index": "partial" if has_page_tag else "no",
        "json_fields_on_request": _level([_json_ok(p) for p in asked]),
        "token_logprobs": _level([bool(c.token_logprobs) for c in calls]),
    }


def build_probe_report(
    per_model: dict[str, tuple[list[PredictionRow], list[PredictionRow]]],
    pages: list[ManifestRow],
    statement_versions: dict[str, str],
) -> dict:
    matrix: dict[str, dict[str, str]] = {cap: {} for cap in CAPABILITIES}
    for model, (statement, asked) in per_model.items():
        for cap, level in capability_row(statement, asked).items():
            matrix[cap][model] = level
    return {
        "matrix": matrix,
        "evidence": {model: [p.sample_id for p in pages] for model in per_model},
        "prompt_versions": {
            model: {
                "statement": statement_versions[model],
                "json_fields_on_request": PROBE_JSON_PLAN.version,
            }
            for model in per_model
        },
    }
