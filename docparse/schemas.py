"""Pydantic models for every per-document artifact under `runs/docparse/<doc_id>/`.

One model per artifact file (design D2). All models are `extra="forbid"`. Artifacts are
deterministic: no timestamps appear anywhere in this module.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ocr_bench.schemas import BBox

DOC_ID_RE = r"^[0-9a-f]{16}$"  # sha256 of the file bytes, first 16 hex chars
BLOCK_ID_RE = r"^p\d+-b\d+$"  # p{page}-b{order}
SEGMENT_ID_RE = r"^p\d+-b\d+-s\d+$"  # {block_id}-s{order}

Stage = Literal["classify", "layout", "segments", "readings", "reconcile", "verify"]


class ModelIdentity(BaseModel):
    """Which registry model produced an artifact (docparse-service spec)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["layout", "classifier", "verifier_reader", "llm"]
    model: str  # registry entry name, e.g. "qwen3vl"
    served_model_name: str
    prompt_version: str
    vllm_version: str = "unknown"


class ArtifactMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Stage
    doc_id: str = Field(pattern=DOC_ID_RE)
    config_hash: str  # hash of the stage's resolved config (task 2.3 computes it)
    input_hashes: dict[str, str] = Field(default_factory=dict)  # upstream artifact -> hash
    models: list[ModelIdentity] = Field(default_factory=list)


class ClassifyArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ArtifactMeta
    class_id: str  # a registry class id, or "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    bank: str | None = None  # bank_statement only; a registry bank or "unknown"
    pages_used: list[int]  # 1-based page numbers sent to the classifier
    invalid_reply: bool = False
    raw_reply: str | None = None


BlockCategory = Literal["text", "title", "table", "figure", "header", "footer", "other"]


class LayoutBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(pattern=BLOCK_ID_RE)
    page: int = Field(ge=1)  # 1-based
    bbox: BBox
    category: BlockCategory
    order: int = Field(ge=0)  # reading order within the page
    text: str | None = None  # backend's reading; None for figures
    table_html: str | None = None  # table blocks only
    truncated: bool = False  # backend reply hit the token limit; text treated as absent


class LayoutPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    blocks: list[LayoutBlock] = Field(default_factory=list)
    error: str | None = None  # page-level failure after retry; blocks empty


class LayoutArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ArtifactMeta
    model: str  # registry entry bound to the layout role
    prompt_version: str
    pages: list[LayoutPage]


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(pattern=SEGMENT_ID_RE)
    block_id: str = Field(pattern=BLOCK_ID_RE)
    page: int = Field(ge=1)
    bbox: BBox
    order: int = Field(ge=0)  # within the block
    row: int | None = None  # table blocks only
    col: int | None = None
    fallback: bool = False  # produced by the projection-profile fallback


class SegmentsArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ArtifactMeta
    max_aspect: float = Field(gt=0)
    segments: list[Segment]


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(pattern=SEGMENT_ID_RE)
    text: str | None = None  # NFC; None iff error
    char_conf: list[float] | None = None  # one per char of text, in [0, 1]; None if unavailable
    reader: str  # "paddle_crop" | "trocr" | ...
    reader_version: str
    error: str | None = None

    @model_validator(mode="after")
    def _check_text_xor_error(self) -> "Reading":
        if (self.text is None) == (self.error is None):
            raise ValueError("exactly one of `text` / `error` must be set")
        if self.char_conf is not None:
            if self.text is None:
                raise ValueError("`char_conf` requires `text` to be set")
            if len(self.char_conf) != len(self.text):
                raise ValueError("`char_conf` must have one entry per character of `text`")
            if any(not (0.0 <= c <= 1.0) for c in self.char_conf):
                raise ValueError("every `char_conf` value must be in [0, 1]")
        return self


class ReadingsArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ArtifactMeta
    readings: list[Reading]


class ReconciledSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(pattern=SEGMENT_ID_RE)
    block_id: str = Field(pattern=BLOCK_ID_RE)
    text: str
    sources: list[str]  # readers whose reading was accepted
    conf: float | None = Field(default=None, ge=0.0, le=1.0)  # calibrated; None if uncalibrated
    disputed: bool = False
    alt: list[str] = Field(default_factory=list)  # the other readings of a disputed segment
    single_source: bool = False  # layout text was truncated/absent, no vote possible


class ReconcileArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ArtifactMeta
    segments: list[ReconciledSegment]


FieldState = Literal["found", "recovered", "absent", "unverified", "not_required"]
Region = Literal["region_hint", "label_block", "page"]


class VerifyAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    region: Region
    crop_path: str | None = None  # relative to the document dir, under crops/
    prompt: str | None = None
    reply: str | None = None
    readings: dict[str, str] = Field(default_factory=dict)  # reader name -> normalized reading
    decision: str  # e.g. "recovered", "absent", "disagree", "reverted: <reason>"


class VerifyArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ArtifactMeta
    attempts: list[VerifyAttempt] = Field(default_factory=list)
    vlm_calls: int = Field(default=0, ge=0)
    budget_stop: Literal["max_vlm_calls", "max_retries"] | None = None


class FieldReport(BaseModel):
    """One entry of fields.json."""

    model_config = ConfigDict(extra="forbid")

    name: str
    state: FieldState
    value: str | None = None
    block_ids: list[str] = Field(default_factory=list)
    page: int | None = None
    bbox: BBox | None = None


class FieldsReport(BaseModel):
    """fields.json."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=DOC_ID_RE)
    class_id: str
    verification_skipped: bool = False  # class "unknown": no required fields to check
    fields: list[FieldReport] = Field(default_factory=list)


class PageError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1)
    stage: Stage
    message: str


class DocStatus(BaseModel):
    """status.json."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=DOC_ID_RE)
    status: Literal["complete", "partial", "error"]
    class_id: str | None = None
    stages: dict[str, Literal["done", "skipped", "error"]] = Field(default_factory=dict)
    page_errors: list[PageError] = Field(default_factory=list)
    error: str | None = None  # document-level error message (status "error")
