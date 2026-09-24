"""Configuration models and loader for `configs/docparse/pipeline.yaml`."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Roles(BaseModel):
    """Registry entry name bound to each role (docparse task 2.5, design D3)."""

    model_config = ConfigDict(extra="forbid")

    layout: str
    classifier: str
    verifier_reader: str
    llm: str


class ClassifyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_pages: int = Field(default=2, ge=1)
    long_side_px: int = Field(default=1024, gt=0)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SegmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_aspect: float | None = Field(default=None, gt=0)


class TrocrConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: str
    version: str | None = None


class VerifyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=3, ge=0)
    max_vlm_calls: int = Field(default=12, ge=0)
    min_conf: float = Field(default=0.9, ge=0.0, le=1.0)


class ExtractConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_step_retries: int = Field(default=2, ge=0)
    free_form_max_tool_calls: int = Field(default=8, ge=1)


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_upload_mb: int = Field(default=50, gt=0)
    page_concurrency: int = Field(default=4, ge=1)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles: Roles
    line_reader: Literal["paddle_crop", "trocr"] = "paddle_crop"
    classes_dir: str
    artifacts_root: str
    classify: ClassifyConfig = Field(default_factory=ClassifyConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    trocr: TrocrConfig
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)


class PipelineConfigError(ValueError):
    pass


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Load and validate `path`; raises `PipelineConfigError` naming the file."""
    if not path.exists():
        raise PipelineConfigError(f"{path}: no such file")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    try:
        return PipelineConfig.model_validate(data)
    except ValidationError as exc:
        raise PipelineConfigError(f"{path}: {exc}") from exc
