"""Per-document artifact store: hashes and skip-if-unchanged (docparse task 2.3).

Each document gets `runs/docparse/<doc_id>/` with one JSON artifact per stage plus a
few non-stage output files (`parsed.html`, `fields.json`, `status.json`). Every artifact
records the config hash used to produce it and the hashes of its upstream artifact files;
a stage is skipped when both still match. All writes go through `ocr_bench.jsonl` so they
are atomic.
"""

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from docparse.schemas import (
    DOC_ID_RE,
    ArtifactMeta,
    ClassifyArtifact,
    LayoutArtifact,
    ReadingsArtifact,
    ReconcileArtifact,
    SegmentsArtifact,
    VerifyArtifact,
)
from ocr_bench.jsonl import write_rows_atomic

STAGE_FILES: dict[str, str] = {  # stage -> file name in the document directory
    "classify": "classify.json",
    "layout": "layout.json",
    "segments": "segments.json",
    "readings": "readings.json",
    "reconcile": "reconcile.json",
    "verify": "verify.json",
}

STAGE_MODELS: dict[str, type[BaseModel]] = {  # stage -> artifact model from docparse.schemas
    "classify": ClassifyArtifact,
    "layout": LayoutArtifact,
    "segments": SegmentsArtifact,
    "readings": ReadingsArtifact,
    "reconcile": ReconcileArtifact,
    "verify": VerifyArtifact,
}

# The upstream artifacts each stage reads. The document itself is covered by doc_id.
STAGE_INPUTS: dict[str, tuple[str, ...]] = {
    "classify": (),
    "layout": (),
    "segments": ("layout",),
    "readings": ("segments",),
    "reconcile": ("layout", "segments", "readings"),
    "verify": ("classify", "reconcile"),
}


def _sha16(data: bytes) -> str:
    """sha256 hexdigest of `data`, first 16 hex chars."""
    return hashlib.sha256(data).hexdigest()[:16]


def doc_id_for_bytes(data: bytes) -> str:
    return _sha16(data)


def doc_id_for_file(path: Path) -> str:
    return doc_id_for_bytes(path.read_bytes())


def config_hash(config: BaseModel | Mapping[str, Any]) -> str:
    """_sha16 of the config's canonical JSON representation."""
    payload = config.model_dump(mode="json") if isinstance(config, BaseModel) else config
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha16(canonical.encode())


class MissingInputError(RuntimeError):
    """A stage's upstream artifact does not exist."""

    def __init__(self, stage: str, input_stage: str) -> None:
        super().__init__(f"stage {stage!r}: missing input artifact {input_stage!r}")
        self.stage = stage
        self.input_stage = input_stage


class ArtifactStore:
    """Reads and writes the artifacts of one document under `root/<doc_id>/`."""

    def __init__(self, root: Path, doc_id: str) -> None:
        if not re.fullmatch(DOC_ID_RE, doc_id):
            raise ValueError(f"invalid doc_id: {doc_id!r}")
        self.doc_id = doc_id
        self.dir = root / doc_id

    def path(self, stage: str) -> Path:
        try:
            name = STAGE_FILES[stage]
        except KeyError:
            raise ValueError(f"unknown stage: {stage!r}") from None
        return self.dir / name

    def read(self, stage: str) -> BaseModel | None:
        """The stored artifact, or None if missing or it fails to parse/validate."""
        path = self.path(stage)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            return STAGE_MODELS[stage].model_validate_json(text)
        except ValueError:
            return None

    def input_hashes(self, stage: str) -> dict[str, str]:
        """{input_stage: hash of its artifact file bytes} for stage's declared inputs."""
        hashes: dict[str, str] = {}
        for input_stage in STAGE_INPUTS[stage]:
            input_path = self.path(input_stage)
            try:
                data = input_path.read_bytes()
            except FileNotFoundError:
                raise MissingInputError(stage, input_stage) from None
            hashes[input_stage] = _sha16(data)
        return hashes

    def is_fresh(self, stage: str, cfg_hash: str) -> bool:
        artifact = self.read(stage)
        if artifact is None:
            return False
        meta: ArtifactMeta = artifact.meta
        if meta.config_hash != cfg_hash:
            return False
        return meta.input_hashes == self.input_hashes(stage)

    def write(self, stage: str, artifact: BaseModel) -> Path:
        model = STAGE_MODELS[stage]
        if not isinstance(artifact, model):
            raise ValueError(f"stage {stage!r} expects {model.__name__}, got {type(artifact)!r}")
        meta: ArtifactMeta = artifact.meta
        if meta.stage != stage:
            raise ValueError(f"artifact meta.stage {meta.stage!r} != stage {stage!r}")
        if meta.doc_id != self.doc_id:
            raise ValueError(f"artifact meta.doc_id {meta.doc_id!r} != {self.doc_id!r}")
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path(stage)
        write_rows_atomic(path, [artifact])
        return path

    def run_stage(
        self,
        stage: str,
        config: BaseModel | Mapping[str, Any],
        compute: Callable[[ArtifactMeta], BaseModel],
    ) -> tuple[BaseModel, bool]:
        cfg_hash = config_hash(config)
        inputs = self.input_hashes(stage)
        if self.is_fresh(stage, cfg_hash):
            artifact = self.read(stage)
            assert artifact is not None  # is_fresh already confirmed it reads and matches
            return artifact, False

        meta = ArtifactMeta(
            stage=stage,
            doc_id=self.doc_id,
            config_hash=cfg_hash,
            input_hashes=inputs,
        )
        artifact = compute(meta)
        result_meta: ArtifactMeta = artifact.meta
        if (
            result_meta.stage != stage
            or result_meta.doc_id != self.doc_id
            or result_meta.config_hash != cfg_hash
            or result_meta.input_hashes != inputs
        ):
            raise ValueError(f"compute() for stage {stage!r} changed the artifact's meta")
        self.write(stage, artifact)
        return artifact, True

    def _validated_name(self, name: str) -> Path:
        if "/" in name or name in ("..", ".") or ".." in Path(name).parts:
            raise ValueError(f"invalid file name: {name!r}")
        return self.dir / name

    def write_text(self, name: str, text: str) -> Path:
        """Atomically write a non-stage text file (e.g. parsed.html) in self.dir."""
        path = self._validated_name(name)
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        return path

    def write_model(self, name: str, model: BaseModel) -> Path:
        """Atomically write a non-stage JSON file (fields.json, status.json)."""
        path = self._validated_name(name)
        write_rows_atomic(path, [model])
        return path
