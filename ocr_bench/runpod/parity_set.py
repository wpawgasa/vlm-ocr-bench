"""The parity sample list (`configs/runpod/parity_samples.v1.yaml`), design D6.

The file holds sample ids only; images and ground truth stay under `data/runs/<run_id>/`.
"""

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from ocr_bench.schemas import Condition, ManifestRow, Source, Task

# Minimum number of samples per task (D6: 60 pages in total).
MINIMUMS: dict[Task, int] = {
    Task.ocr_fullpage: 20,
    Task.ocr_line: 10,
    Task.table: 10,
    Task.handwriting: 10,
    Task.kie: 10,
}


class ParitySet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    run_id: str
    samples: dict[Task, list[str]]

    @property
    def sample_ids(self) -> list[str]:
        return [i for ids in self.samples.values() for i in ids]


def load_parity_set(path: Path, manifest_rows: Iterable[ManifestRow]) -> ParitySet:
    """Load and validate the parity list against the manifest rows of its run."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ps = ParitySet.model_validate(raw)

    by_id: dict[str, list[ManifestRow]] = {}
    for row in manifest_rows:
        by_id.setdefault(row.sample_id, []).append(row)

    for task, ids in ps.samples.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"parity set lists a duplicate id under {task.value}")
        for sample_id in ids:
            rows = by_id.get(sample_id)
            if not rows:
                raise ValueError(f"parity sample {sample_id!r} is not in the manifest")
            if any(r.source == Source.bankstmt for r in rows):
                raise ValueError(
                    f"parity sample {sample_id!r} is a bankstmt page; client data is never allowed"
                )
            if not any(r.condition == Condition.clean for r in rows):
                raise ValueError(f"parity sample {sample_id!r} has no clean row in the manifest")
            if any(r.task != task for r in rows):
                raise ValueError(
                    f"parity sample {sample_id!r} is listed under {task.value} "
                    f"but the manifest task differs"
                )

    for task, minimum in MINIMUMS.items():
        have = len(ps.samples.get(task, []))
        if have < minimum:
            raise ValueError(f"parity set has {have} {task.value} samples, needs {minimum}")
    return ps
