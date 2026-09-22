"""The statement review queue and the manual label set (task 5.5).

`ocrbench review export` writes `runs/<id>/review/queue.csv`: one line per header field
and per row cell that at least one model produced, with every cross-model disagreement
plus a seeded 10% sample of the agreements. A reviewer fills the `label` column, and
`ocrbench review load` validates every labelled line strictly (a bad amount or date
fails the whole file, naming the line) and writes the labels as statement ground truth
under `runs/<id>/gt_manual/`, which `ocrbench score` then prefers.
"""

import csv
import random
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from ocr_bench.metrics.statement_gt import ROW_CELLS, SCORED_FIELDS, cell_map, normalize_cell
from ocr_bench.normalize.statement import parse_date, parse_decimal
from ocr_bench.normalize.text import canonical_text
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import JsonGT, StatementPage, StatementRow

#: Fixed columns; the models' value columns sit between `row_index` and `label`.
QUEUE_COLUMNS = ("sample_id", "page_no", "file_id", "field", "row_index", "label")
AGREEMENT_SAMPLE = 0.10

_DATE_FIELDS = frozenset({"date", "period_start", "period_end"})
_AMOUNT_FIELDS = frozenset(
    {"debit", "credit", "amount", "balance", "opening_balance", "closing_balance"}
)


class LabelError(ValueError):
    """An invalid label CSV: the message names the file and the offending line."""


def _cell_sort_key(field: str, row_index: int | None) -> tuple:
    """Header fields first in their declared order, then rows by index and cell order."""
    if row_index is None:
        return (0, SCORED_FIELDS.index(field), 0)
    return (1, row_index, ROW_CELLS.index(field))


def _split_key(key: str) -> tuple[str, int | None]:
    """`"row[3].debit"` -> `("debit", 3)`; `"account_no"` -> `("account_no", None)`."""
    if key.startswith("row["):
        index, _, cell = key[4:].partition("].")
        return cell, int(index)
    return key, None


def queue_rows(
    pages: dict[tuple[str, str], StatementPage | None],
    models: Sequence[str],
    *,
    file_ids: dict[str, str],
    page_numbers: dict[str, int],
    seed: int,
    agreement_sample: float = AGREEMENT_SAMPLE,
) -> list[dict[str, str]]:
    """Queue lines for `{(sample_id, model): page}`: every disagreement, plus a seeded
    sample of the agreements.

    A cell no model produced is left out; a cell one model produced and another did not
    counts as a disagreement, since that missing value is exactly what review must settle.
    """
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for sample_id in sorted({sample for sample, _ in pages}):
        cells = {model: cell_map(pages.get((sample_id, model))) for model in models}
        keys = sorted(
            {key for values in cells.values() for key in values},
            key=lambda k: _cell_sort_key(*_split_key(k)),
        )
        for key in keys:
            values = {model: cells[model].get(key) for model in models}
            if all(value is None for value in values.values()):
                continue
            normalized = {normalize_cell(key, value) for value in values.values()}
            agreed = len(normalized) == 1
            if agreed and rng.random() >= agreement_sample:
                continue
            field, row_index = _split_key(key)
            row = {
                "sample_id": sample_id,
                "page_no": str(page_numbers.get(sample_id, 1)),
                "file_id": file_ids.get(sample_id, sample_id),
                "field": field,
                "row_index": "" if row_index is None else str(row_index),
                "label": "",
            }
            for model in models:
                row[f"{model}_value"] = values[model] or ""
            rows.append(row)
    return rows


def queue_fieldnames(models: Sequence[str]) -> list[str]:
    return [*QUEUE_COLUMNS[:5], *(f"{m}_value" for m in models), QUEUE_COLUMNS[5]]


def write_queue(path: Path, rows: list[dict[str, str]], models: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=queue_fieldnames(models))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# --- loading ------------------------------------------------------------------------------------


def _fail(path: Path, lineno: int, message: str) -> None:
    raise LabelError(f"{path}: line {lineno}: {message}")


def _parse_index(path: Path, lineno: int, field: str, raw: str) -> int:
    if not raw.strip().lstrip("-").isdigit() or int(raw) < 0:
        _fail(path, lineno, f"field {field!r} needs a non-negative row_index, got {raw!r}")
    return int(raw)


def _parse_value(path: Path, lineno: int, field: str, raw: str) -> str | Decimal:
    if field in _DATE_FIELDS:
        value = parse_date(raw)
        if value is None:
            _fail(path, lineno, f"field {field!r}: {raw!r} is not a date")
        return value
    if field in _AMOUNT_FIELDS:
        amount = parse_decimal(raw)
        if amount is None:
            _fail(path, lineno, f"field {field!r}: {raw!r} is not an amount")
        return amount
    return canonical_text(raw)


def load_labels(path: Path) -> dict[str, StatementPage]:
    """Parse a filled queue CSV into `{sample_id: StatementPage}`.

    Every labelled line is validated; the first bad one raises `LabelError` naming the
    line, so a file is never half-loaded. Lines with an empty `label` are unlabelled and
    contribute nothing. Row cells are keyed by the `row_index` the queue exported, and a
    row index no line labels is still created (empty), so positions never shift.
    """
    pages: dict[str, StatementPage] = {}
    rows: dict[str, dict[int, StatementRow]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"sample_id", "field", "row_index", "label"} - set(reader.fieldnames or [])
        if missing:
            raise LabelError(f"{path}: missing column(s) {sorted(missing)}")
        for line in reader:
            lineno = reader.line_num
            label = (line.get("label") or "").strip()
            if not label:
                continue
            sample_id = (line.get("sample_id") or "").strip()
            if not sample_id:
                _fail(path, lineno, "sample_id is empty")
            field = (line.get("field") or "").strip()
            raw_index = (line.get("row_index") or "").strip()
            page = pages.setdefault(sample_id, StatementPage())
            page_no = (line.get("page_no") or "").strip()
            if page_no:
                if not page_no.isdigit() or int(page_no) < 1:
                    _fail(path, lineno, f"page_no {page_no!r} is not a page number")
                page.page_no = int(page_no)

            if field in SCORED_FIELDS:
                if raw_index:
                    _fail(path, lineno, f"header field {field!r} must have an empty row_index")
                setattr(page, field, _parse_value(path, lineno, field, label))
            elif field in ROW_CELLS:
                index = _parse_index(path, lineno, field, raw_index)
                row = rows.setdefault(sample_id, {}).setdefault(index, StatementRow())
                setattr(row, field, _parse_value(path, lineno, field, label))
            else:
                _fail(path, lineno, f"unknown field {field!r}")

    for sample_id, by_index in rows.items():
        pages[sample_id].rows = [
            by_index.get(index, StatementRow()) for index in range(max(by_index) + 1)
        ]
    return pages


def write_manual_gt(paths: RunPaths, pages: dict[str, StatementPage]) -> list[Path]:
    """Write each labelled page as `gt_manual/<sample_id>.json` (`JsonGT.statement`)."""
    paths.gt_manual.mkdir(parents=True, exist_ok=True)
    written = []
    for sample_id, page in sorted(pages.items()):
        target = paths.gt_manual / f"{sample_id}.json"
        target.write_text(
            JsonGT(gt_kind="json", statement=page).model_dump_json(), encoding="utf-8"
        )
        written.append(target)
    return written


def load_manual_pages(paths: RunPaths) -> dict[str, StatementPage]:
    """`{sample_id: StatementPage}` for every manual label stored in the run."""
    if not paths.gt_manual.exists():
        return {}
    pages: dict[str, StatementPage] = {}
    for path in sorted(paths.gt_manual.glob("*.json")):
        gt = JsonGT.model_validate_json(path.read_text(encoding="utf-8"))
        if gt.statement is not None:
            pages[path.stem] = gt.statement
    return pages
