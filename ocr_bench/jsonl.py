"""Typed JSON Lines I/O for row models.

Writes are append-and-flush so `infer` can resume; final artifacts from
`score`, `calibrate` and `report` are written to a temp file and renamed,
which makes them atomic.
"""

import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from ocr_bench.schemas import PredictionRow

T = TypeVar("T", bound=BaseModel)
PredictionKey = tuple[str, str, str]  # (sample_id, condition, model)


def read_rows(path: Path, model: type[T]) -> Iterator[T]:
    """Yield `model` instances parsed from each non-blank line of `path`.

    Raises `ValueError(f"{path}:{lineno}: ...")` on an invalid row.
    """
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield model.model_validate_json(stripped)
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc


def append_rows(path: Path, rows: Iterable[BaseModel]) -> int:
    """Append `rows` to `path`, one JSON object per line. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json())
            fh.write("\n")
            count += 1
        fh.flush()
        os.fsync(fh.fileno())
    return count


def write_rows_atomic(path: Path, rows: Iterable[BaseModel]) -> int:
    """Write `rows` to `path` atomically (temp file + fsync + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    count = 0
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json())
            fh.write("\n")
            count += 1
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)
    return count


def prediction_key(row: PredictionRow) -> PredictionKey:
    return (row.sample_id, row.condition.value, row.model)


def latest_predictions(path: Path) -> dict[PredictionKey, PredictionRow]:
    """The last row per `(sample_id, condition, model)` in `path` (empty if it is missing).

    `infer` appends a retried row after the errored one it supersedes, so readers must
    take the last row per key.
    """
    latest: dict[PredictionKey, PredictionRow] = {}
    if not path.exists():
        return latest
    for row in read_rows(path, PredictionRow):
        latest[prediction_key(row)] = row
    return latest
