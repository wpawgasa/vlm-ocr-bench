"""`ocrbench infer` orchestration: resumable, bounded-concurrency inference of one model
over the manifest, plus the per-model summary.

Pages in flight are bounded by an `asyncio.Semaphore(concurrency)` per model; each
finished page is appended (and fsynced) to `predictions.jsonl` at once, so an interrupted
run loses at most the pages in flight. A re-run skips keys whose latest row has no error.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ocr_bench.jsonl import PredictionKey, append_rows
from ocr_bench.models.base import OcrModel
from ocr_bench.schemas import ManifestRow, NormalizedPage, PredictionRow, RawPrediction


def manifest_key(row: ManifestRow, model: str) -> PredictionKey:
    return (row.sample_id, row.condition.value, model)


def pending_rows(
    rows: list[ManifestRow], model: str, latest: dict[PredictionKey, PredictionRow]
) -> list[ManifestRow]:
    """Manifest rows still to request: no prediction yet, or the latest one has an error."""
    pending = []
    for row in rows:
        done = latest.get(manifest_key(row, model))
        if done is None or done.raw.error is not None:
            pending.append(row)
    return pending


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def error_row(row: ManifestRow, model: str, error: str) -> PredictionRow:
    return PredictionRow(
        sample_id=row.sample_id,
        condition=row.condition,
        model=model,
        prompt_version="unknown",
        raw=RawPrediction(error=error),
        normalized=NormalizedPage(),
        latency_ms=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        n_requests=0,
    )


async def predict_row(model: OcrModel, row: ManifestRow, run_root: Path) -> PredictionRow:
    """One page's prediction; any exception becomes an error row (no silent drops)."""
    try:
        image = await asyncio.to_thread(load_image, run_root / row.image_path)
        return await model.predict(row, image)
    except Exception as exc:
        return error_row(row, model.name, f"{type(exc).__name__}: {exc}")


async def infer_model(
    model: OcrModel,
    rows: list[ManifestRow],
    run_root: Path,
    predictions: Path,
    concurrency: int,
    on_row: Callable[[PredictionRow], None] | None = None,
) -> int:
    """Predict `rows` with `model`, appending each row as it completes. Returns the count."""
    semaphore = asyncio.Semaphore(concurrency)
    written = 0

    async def one(row: ManifestRow) -> None:
        nonlocal written
        async with semaphore:
            prediction = await predict_row(model, row, run_root)
            append_rows(predictions, [prediction])
            written += 1
            if on_row is not None:
                on_row(prediction)

    async with asyncio.TaskGroup() as group:
        for row in rows:
            group.create_task(one(row))
    return written


@dataclass
class ModelSummary:
    model: str
    pages: int
    predicted: int
    errors: int
    mean_latency_ms: float | None
    p95_latency_ms: float | None

    @property
    def error_rate(self) -> float:
        return self.errors / self.pages if self.pages else 0.0

    def line(self) -> str:
        def ms(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.0f}"

        return (
            f"{self.model}: pages={self.pages} errors={self.errors} "
            f"error_rate={self.error_rate:.2%} predicted={self.predicted} "
            f"latency_mean_ms={ms(self.mean_latency_ms)} latency_p95_ms={ms(self.p95_latency_ms)}"
        )


def summarize(
    model: str, rows: list[ManifestRow], latest: dict[PredictionKey, PredictionRow]
) -> ModelSummary:
    """Summary over the manifest's keys; latency stats use rows without error."""
    found = [latest[k] for k in (manifest_key(r, model) for r in rows) if k in latest]
    latencies = [p.latency_ms for p in found if p.raw.error is None]
    return ModelSummary(
        model=model,
        pages=len(rows),
        predicted=len(found),
        errors=sum(1 for p in found if p.raw.error is not None),
        mean_latency_ms=float(np.mean(latencies)) if latencies else None,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else None,
    )
