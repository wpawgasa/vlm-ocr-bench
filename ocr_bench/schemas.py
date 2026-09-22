"""Pydantic v2 row schemas shared by every stage of the benchmark harness.

`jsonl.py` reads and writes these models one per line. Nothing outside this
module should define a row shape.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

BBox = tuple[float, float, float, float]


class Task(StrEnum):
    ocr_fullpage = "ocr_fullpage"
    ocr_line = "ocr_line"
    handwriting = "handwriting"
    table = "table"
    docparse = "docparse"
    kie = "kie"
    kie_map = "kie_map"
    classify = "classify"
    statement = "statement"


class Condition(StrEnum):
    clean = "clean"
    scan_low = "scan_low"
    photo = "photo"


class GtKind(StrEnum):
    json = "json"
    text = "text"
    html = "html"
    text_layer = "text_layer"
    none = "none"


class Source(StrEnum):
    thaiocrbench = "thaiocrbench"
    bankstmt = "bankstmt"


class BlockType(StrEnum):
    text = "text"
    table = "table"
    figure = "figure"
    header = "header"
    footer = "footer"


class ErrorFlag(StrEnum):
    balance_mismatch = "balance_mismatch"
    total_mismatch = "total_mismatch"
    model_disagreement = "model_disagreement"
    low_confidence = "low_confidence"
    missing_required = "missing_required"
    page_gap = "page_gap"


class ManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source: Source
    task: Task
    subtask: str | None = None
    domain: str | None = None
    bank: str | None = None
    doc_type: str | None = None
    page_no: int = Field(1, ge=1)
    n_pages: int = Field(1, ge=1)
    condition: Condition
    image_path: str
    gt_kind: GtKind
    gt_path: str | None = None
    critical_fields: list[str] = Field(default_factory=list)
    question: str = ""

    @model_validator(mode="after")
    def _check_page_no(self) -> "ManifestRow":
        if self.page_no > self.n_pages:
            raise ValueError(f"page_no ({self.page_no}) must be <= n_pages ({self.n_pages})")
        return self


class Block(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: BlockType
    text: str = ""
    bbox: BBox | None = None
    page: int = 1


class FieldValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | None
    raw: str | None = None
    confidence: float | None = None
    bbox: BBox | None = None
    flags: list[str] = Field(default_factory=list)


class NormalizedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""
    blocks: list[Block] = Field(default_factory=list)
    tables: list[list[list[str]]] = Field(default_factory=list)
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    label: str | None = None
    parse_error: str | None = None


class RawCall(BaseModel):
    """One request of a page's pipeline, verbatim (TeleOCR pages have one layout call plus
    one call per block; every other plan has a single call)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["single", "layout", "block"]
    block_index: int | None = None
    block_type: str | None = None
    text: str = ""
    tokens: list[str] = Field(default_factory=list)
    token_logprobs: list[float] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    error: str | None = None


class RawPrediction(BaseModel):
    """Page-level aggregate of the page's calls.

    `text` is the model's page-level text (two-stage: block texts joined in reading order);
    `tokens`/`token_logprobs` are all calls concatenated in call order; `error` is the
    page-level error or None.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    tokens: list[str] = Field(default_factory=list)
    token_logprobs: list[float] = Field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    calls: list[RawCall] = Field(default_factory=list)


class PredictionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    condition: Condition
    model: str
    prompt_version: str
    raw: RawPrediction
    normalized: NormalizedPage
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    n_requests: int = 1


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    bbox: BBox


class StatementRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str | None = None
    description: str | None = None
    debit: Decimal | None = None
    credit: Decimal | None = None
    # An unsigned amount from a combined "Withdrawal / Deposit" column, whose side the
    # extraction could not tell; Check B infers the side from the balance delta and sets
    # `side_inferred` (such rows are excluded from `row_consistency_rate`).
    amount: Decimal | None = None
    side_inferred: bool = False
    balance: Decimal | None = None
    channel: str | None = None
    bbox: BBox | None = None
    page_no: int = 1


class StatementPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bank: str | None = None
    account_no: str | None = None
    account_name: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    opening_balance: Decimal | None = None
    closing_balance: Decimal | None = None
    # Summary totals printed on the page ("TOTAL WITHDRAWAL(S)"/"TOTAL DEPOSIT(S)"),
    # checked against the row sums by Check B.
    total_debit: Decimal | None = None
    total_credit: Decimal | None = None
    rows: list[StatementRow] = Field(default_factory=list)
    page_no: int = 1


class StatementFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str
    bank: str | None = None
    pages: list[StatementPage] = Field(default_factory=list)

    @property
    def rows(self) -> list[StatementRow]:
        ordered_pages = sorted(self.pages, key=lambda p: p.page_no)
        result: list[StatementRow] = []
        for page in ordered_pages:
            result.extend(page.rows)
        return result


class TextGT(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gt_kind: Literal["text"]
    text: str


class HtmlGT(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gt_kind: Literal["html"]
    html: str
    # The benchmark's original answer (markdown or HTML) as the official scorer sees it.
    raw: str | None = None


class JsonGT(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gt_kind: Literal["json"]
    fields: dict[str, str | None] = Field(default_factory=dict)
    label: str | None = None
    statement: StatementPage | None = None
    raw: str | None = None


class TextLayerGT(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gt_kind: Literal["text_layer"]
    gt_text: str
    gt_words: list[Word] = Field(default_factory=list)
    statement: StatementPage | None = None
    homography: list[list[float]] | None = None


class NoGT(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gt_kind: Literal["none"]


GroundTruth = Annotated[
    TextGT | HtmlGT | JsonGT | TextLayerGT | NoGT,
    Field(discriminator="gt_kind"),
]
GROUND_TRUTH: TypeAdapter[GroundTruth] = TypeAdapter(GroundTruth)


class ScoreRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    condition: Condition
    model: str
    task: Task
    metric: str
    value: float | None
    subtask: str | None = None
    domain: str | None = None
    bank: str | None = None
    doc_type: str | None = None
    is_critical: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class FieldResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    condition: Condition
    model: str
    task: Task
    field: str
    is_critical: bool
    subtask: str | None = None
    pred: str | None
    gt: str | None
    field_exact: int
    field_fuzzy: int
    false_accept: bool = False
    false_reject: bool = False
    domain: str | None = None
    bank: str | None = None
    doc_type: str | None = None


class AggregateRow(BaseModel):
    """One cell of a slice table in `aggregates.jsonl`: a metric's mean over the samples
    matching `keys`, with a 95% bootstrap CI over samples."""

    model_config = ConfigDict(extra="forbid")

    table: str
    keys: dict[str, str | bool | None]
    metric: str
    mean: float | None
    n: int
    ci_low: float | None
    ci_high: float | None


class CalibrationRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    variant: Literal["full", "logprob_only"]
    band: int = Field(ge=1, le=5)
    lo: float
    hi: float
    n: int
    accuracy: float | None
    ci_low: float | None
    ci_high: float | None
    critical_only: bool = False


class LatencyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    gpu: str
    concurrency: int
    sample_id: str
    latency_ms: float
    ttft_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    image_px: int = 0
    error: str | None = None
    warmup: bool = False


class OutputSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    page: int
    bbox: BBox | None = None


class PrepareSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    thaiocrbench: dict[str, int] = Field(default_factory=dict)
    thaiocrbench_domains: dict[str, int] = Field(default_factory=dict)
    domain_slice_available: bool = False
    statement_pages_by_doc_type: dict[str, int] = Field(default_factory=dict)
    statement_banks: list[str] = Field(default_factory=list)
    statement_files: int = 0
    statement_multipage_files: int = 0
    statement_text_layer_unusable: int = 0
    n_samples: int = 0
    n_manifest_rows: int = 0
    warnings: list[str] = Field(default_factory=list)


class OutputField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    value: str | float | int | None
    raw: str | None
    type: str
    unit: str | None = None
    definition: str
    source: OutputSource
    normalization: list[str] = Field(default_factory=list)
    confidence: float | None = None
    error_flags: list[ErrorFlag] = Field(default_factory=list)
    model: str
    prompt_version: str
    run_id: str
