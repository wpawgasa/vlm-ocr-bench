"""Run-directory path helpers honoring `$OCRBENCH_DATA_DIR`."""

import os
from dataclasses import dataclass
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("OCRBENCH_DATA_DIR", "data"))


class MissingArtifactError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @classmethod
    def for_run(cls, run_id: str) -> "RunPaths":
        return cls(root=data_dir() / "runs" / run_id)

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions.jsonl"

    @property
    def scores(self) -> Path:
        return self.root / "scores.jsonl"

    @property
    def fields(self) -> Path:
        return self.root / "fields.jsonl"

    @property
    def statements(self) -> Path:
        return self.root / "statements.jsonl"

    @property
    def review_queue(self) -> Path:
        return self.root / "review" / "queue.csv"

    @property
    def gt_manual(self) -> Path:
        return self.root / "gt_manual"

    @property
    def dupset(self) -> Path:
        return self.root / "dupset.json"

    @property
    def duplicates(self) -> Path:
        return self.root / "duplicates.jsonl"

    @property
    def duplicate_pairs(self) -> Path:
        return self.root / "duplicate_pairs.jsonl"

    @property
    def aggregates(self) -> Path:
        return self.root / "aggregates.jsonl"

    @property
    def tob_parity(self) -> Path:
        return self.root / "tob_parity.json"

    @property
    def calibration(self) -> Path:
        return self.root / "calibration.jsonl"

    @property
    def calibration_fields(self) -> Path:
        return self.root / "calibration_fields.jsonl"

    @property
    def reliability(self) -> Path:
        return self.root / "reliability.json"

    @property
    def latency(self) -> Path:
        return self.root / "latency.jsonl"

    @property
    def thresholds(self) -> Path:
        return self.root / "thresholds.json"

    @property
    def probe(self) -> Path:
        return self.root / "probe.json"

    @property
    def probe_predictions(self) -> Path:
        return self.root / "probe_predictions.jsonl"

    @property
    def config_infer(self) -> Path:
        return self.root / "config.infer.yaml"

    @property
    def config_resolved(self) -> Path:
        return self.root / "config.resolved.yaml"

    @property
    def prepare_summary(self) -> Path:
        return self.root / "prepare_summary.json"

    @property
    def report_dir(self) -> Path:
        return self.root / "report"

    @property
    def images(self) -> Path:
        return self.root / "img"

    @property
    def gt(self) -> Path:
        return self.root / "gt"


def require(path: Path, producer: str) -> Path:
    """Return `path` if it exists, else raise `MissingArtifactError`."""
    if not path.exists():
        raise MissingArtifactError(f"missing {path} — produced by `ocrbench {producer}`")
    return path
