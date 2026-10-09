"""Frozen eval file list (`configs/docparse/eval_files.v<N>.yaml`, docparse task 1.3)."""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_EVAL_FILES = Path("configs/docparse/eval_files.v1.yaml")

_VERSIONED_NAME = re.compile(r"eval_files\.v(\d+)\.yaml")


class EvalFile(BaseModel):
    """One statement file: its stem under data/statements and its harness file id."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    harness_id: str = Field(pattern=r"^BS-[a-z]+-\d{4}$")
    bank: str = Field(pattern=r"^[a-z]+$")

    @model_validator(mode="after")
    def _bank_matches_harness_id(self) -> "EvalFile":
        if not self.harness_id.startswith(f"BS-{self.bank}-"):
            raise ValueError(f"harness_id {self.harness_id!r} does not match bank {self.bank!r}")
        return self


class EvalFileList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    text_layer: list[EvalFile]
    anchor: list[EvalFile] = []

    @model_validator(mode="after")
    def _no_duplicates_within_group(self) -> "EvalFileList":
        for name in ("text_layer", "anchor"):
            stems = [f.file for f in getattr(self, name)]
            dupes = sorted({s for s in stems if stems.count(s) > 1})
            if dupes:
                raise ValueError(f"duplicate file in {name}: {dupes}")
        return self

    @property
    def files(self) -> list[EvalFile]:
        """text_layer + anchor, deduplicated by `file`, first-seen order."""
        seen: dict[str, EvalFile] = {}
        for entry in [*self.text_layer, *self.anchor]:
            seen.setdefault(entry.file, entry)
        return list(seen.values())

    @property
    def file_ids(self) -> list[str]:
        return [f.file for f in self.files]

    @property
    def harness_ids(self) -> list[str]:
        return [f.harness_id for f in self.files]

    @property
    def file_count(self) -> int:
        return len(self.files)


class EvalFilesError(ValueError):
    pass


def load_eval_files(path: Path) -> EvalFileList:
    """Load and validate `path`; raises `EvalFilesError` naming the file."""
    if not path.exists():
        raise EvalFilesError(f"{path}: no such file")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    try:
        result = EvalFileList.model_validate(data)
    except ValidationError as exc:
        raise EvalFilesError(f"{path}: {exc}") from exc
    match = _VERSIONED_NAME.fullmatch(path.name)
    if match and int(match.group(1)) != result.version:
        raise EvalFilesError(
            f"{path}: filename says version {match.group(1)} but file says {result.version}"
        )
    return result
