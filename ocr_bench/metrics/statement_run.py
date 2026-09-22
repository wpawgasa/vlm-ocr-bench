"""The statement part of `ocrbench score`: per-file Check B, per-page Check C and the
precision of both checks' flags against the manual labels (tasks 5.3, 5.4, 5.6).

Predictions are mapped page by page with the shared rule-based mapper, merged into
statement files, and checked. Everything here is deterministic: the same predictions
always give byte-identical `statements.jsonl` and score rows.
"""

import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from ocr_bench.config import BankOverrides
from ocr_bench.metrics.agreement import agreement_rate, pair_agreements
from ocr_bench.metrics.arithmetic import ArithmeticResult, check_file
from ocr_bench.metrics.dispatch import is_full_miss, overrides_for
from ocr_bench.metrics.statement_gt import ROW_CELLS, cell_disagreements, cell_map, normalize_cell
from ocr_bench.normalize.statement import map_statement_page, merge_file
from ocr_bench.schemas import (
    Condition,
    ErrorFlag,
    ManifestRow,
    PredictionRow,
    ScoreRow,
    StatementFile,
    StatementPage,
    Task,
)

PAIR_SEPARATOR = "|"
_PAGE_SUFFIX_RE = re.compile(r"-p\d+$")

PageKey = tuple[str, str]  # (sample_id, condition)
PredictionKey = tuple[str, str, str]  # (sample_id, condition, model)


class StatementRecord(BaseModel):
    """One file's mapped statement and Check B result, for M5/M6 (`statements.jsonl`)."""

    model_config = ConfigDict(extra="forbid")

    file_id: str
    condition: Condition
    model: str
    bank: str | None = None
    doc_type: str | None = None
    sample_ids: list[str] = []
    statement: StatementFile
    arithmetic: ArithmeticResult


def file_id_for(sample_id: str) -> str:
    """The statement file a page belongs to: its sample id without the `-p<n>` suffix."""
    return _PAGE_SUFFIX_RE.sub("", sample_id)


def map_pages(
    manifest: list[ManifestRow],
    latest: dict[PredictionKey, PredictionRow],
    overrides: dict[str, BankOverrides] | None,
    models: list[str],
) -> dict[tuple[str, str, str], StatementPage]:
    """Every statement page of every model, mapped to a `StatementPage`.

    A missing or failed prediction still yields an (empty) page, so a file never loses a
    page and the page count stays comparable across models.
    """
    pages: dict[tuple[str, str, str], StatementPage] = {}
    for row in manifest:
        if row.task != Task.statement:
            continue
        for model in models:
            pred = latest.get((row.sample_id, row.condition.value, model))
            if pred is None or is_full_miss(pred):
                page = StatementPage(bank=row.bank, page_no=row.page_no)
            else:
                page = map_statement_page(
                    pred.normalized,
                    bank=row.bank,
                    page_no=row.page_no,
                    overrides=overrides_for(row, overrides),
                )
            pages[(row.sample_id, row.condition.value, model)] = page
    return pages


def _slice_keys(row: ManifestRow) -> dict:
    return dict(
        condition=row.condition,
        task=Task.statement,
        subtask=row.subtask,
        domain=row.domain,
        bank=row.bank,
        doc_type=row.doc_type,
    )


def _flag_precision(flagged: list[bool | None]) -> float | None:
    """Share of raised flags that were true errors; None when nothing was flagged."""
    raised = [f for f in flagged if f is not None]
    return sum(raised) / len(raised) if raised else None


def _row_cell_differs(
    pred_cells: dict[str, str | None], label_cells: dict[str, str | None], index: int
) -> bool:
    """Whether a labelled row disagrees with the extraction on any of its cells."""
    return any(
        normalize_cell(key, pred_cells.get(key)) != normalize_cell(key, label_cells.get(key))
        for key in (f"row[{index}].{cell}" for cell in ROW_CELLS)
    )


def run_statement_checks(
    manifest: list[ManifestRow],
    latest: dict[PredictionKey, PredictionRow],
    overrides: dict[str, BankOverrides] | None = None,
    manual: dict[str, StatementPage] | None = None,
) -> tuple[list[ScoreRow], list[StatementRecord]]:
    """Checks B and C over every statement file, plus flag precision where labels exist.

    File-level metrics (`row_consistency_rate`, `file_reconciles`) are written once per
    file, on its first page's `sample_id`, so a long file does not outweigh a short one
    in an aggregate. Agreement is written per page with the model pair as the `model`
    key ("teleocr|dotsocr"), which is how the report shows it as its own column.
    """
    manual = manual or {}
    models = sorted({model for (_, _, model) in latest})
    statement_rows = [r for r in manifest if r.task == Task.statement]
    pages = map_pages(manifest, latest, overrides, models)

    scores: list[ScoreRow] = []
    records: list[StatementRecord] = []

    files: dict[tuple[str, str, str], list[ManifestRow]] = defaultdict(list)
    for row in statement_rows:
        for model in models:
            files[(file_id_for(row.sample_id), row.condition.value, model)].append(row)

    for (file_id, condition, model), rows in sorted(files.items()):
        ordered = sorted(rows, key=lambda r: r.page_no)
        first = ordered[0]
        statement = merge_file(
            [pages[(r.sample_id, condition, model)] for r in ordered],
            file_id=file_id,
            bank=first.bank,
        )
        result = check_file(statement, model=model)
        records.append(
            StatementRecord(
                file_id=file_id,
                condition=first.condition,
                model=model,
                bank=first.bank,
                doc_type=first.doc_type,
                sample_ids=[r.sample_id for r in ordered],
                statement=statement,
                arithmetic=result,
            )
        )
        extra = {
            "file_id": file_id,
            "n_rows": result.n_rows,
            "first_break_row": result.first_break_row,
            "newest_first": result.newest_first,
            "flags": [f.value for f in result.flags],
        }
        for metric, value in (
            ("row_consistency_rate", result.row_consistency_rate),
            ("file_reconciles", float(result.file_reconciles)),
        ):
            scores.append(
                ScoreRow(
                    sample_id=first.sample_id,
                    model=model,
                    metric=metric,
                    value=None if value is None else float(value),
                    extra=dict(extra),
                    **_slice_keys(first),
                )
            )

        # Check B flag precision: a flagged row whose labelled cells differ was a true error.
        for row in ordered:
            label = manual.get(row.sample_id)
            if label is None:
                continue
            page = pages[(row.sample_id, condition, model)]
            pred_cells, label_cells = cell_map(page), cell_map(label)
            verdicts = [
                _row_cell_differs(pred_cells, label_cells, check.page_row_index)
                for check in result.rows
                if check.page_no == row.page_no and ErrorFlag.balance_mismatch in check.flags
            ]
            scores.append(
                ScoreRow(
                    sample_id=row.sample_id,
                    model=model,
                    metric="arith_flag_precision",
                    value=_flag_precision(verdicts),
                    extra={"n_flags": len(verdicts)},
                    **_slice_keys(row),
                )
            )

    # Check C: agreement per page and per model pair, plus its flag precision.
    for row in statement_rows:
        page_by_model = {m: pages[(row.sample_id, row.condition.value, m)] for m in models}
        label = manual.get(row.sample_id)
        for (model_a, model_b), agreement in sorted(pair_agreements(page_by_model).items()):
            pair = f"{model_a}{PAIR_SEPARATOR}{model_b}"
            rate = agreement_rate(agreement)
            scores.append(
                ScoreRow(
                    sample_id=row.sample_id,
                    model=pair,
                    metric="agreement_rate",
                    value=rate,
                    extra={"n_compared": sum(v is not None for v in agreement.values())},
                    **_slice_keys(row),
                )
            )
            if label is None:
                continue
            cells_a, cells_b = cell_map(page_by_model[model_a]), cell_map(page_by_model[model_b])
            label_cells = cell_map(label)
            verdicts = [
                normalize_cell(key, cells_a.get(key)) != normalize_cell(key, label_cells.get(key))
                or normalize_cell(key, cells_b.get(key))
                != normalize_cell(key, label_cells.get(key))
                for key in cell_disagreements(cells_a, cells_b)
            ]
            scores.append(
                ScoreRow(
                    sample_id=row.sample_id,
                    model=pair,
                    metric="agreement_flag_precision",
                    value=_flag_precision(verdicts),
                    extra={"n_flags": len(verdicts)},
                    **_slice_keys(row),
                )
            )

    scores.sort(key=lambda s: (s.sample_id, s.condition.value, s.model, s.metric))
    records.sort(key=lambda r: (r.file_id, r.condition.value, r.model))
    return scores, records
