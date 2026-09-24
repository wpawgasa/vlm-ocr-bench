"""Round-trip and validation tests for docparse.schemas."""

import inspect

import pytest
from pydantic import BaseModel, ValidationError

from docparse import schemas
from docparse.schemas import (
    ArtifactMeta,
    ClassifyArtifact,
    DocStatus,
    FieldReport,
    FieldsReport,
    LayoutArtifact,
    LayoutBlock,
    LayoutPage,
    ModelIdentity,
    PageError,
    Reading,
    ReadingsArtifact,
    ReconcileArtifact,
    ReconciledSegment,
    Segment,
    SegmentsArtifact,
    VerifyArtifact,
    VerifyAttempt,
)

DOC_ID = "0123456789abcdef"

_MODEL_IDENTITY = ModelIdentity(
    role="layout",
    model="paddleocr_vl",
    served_model_name="PaddleOCR-VL-0.9B",
    prompt_version="v1",
)

_ARTIFACT_META = ArtifactMeta(
    stage="layout",
    doc_id=DOC_ID,
    config_hash="abc123",
    input_hashes={"classify": "def456"},
    models=[_MODEL_IDENTITY],
)

_CLASSIFY_ARTIFACT = ClassifyArtifact(
    meta=ArtifactMeta(stage="classify", doc_id=DOC_ID, config_hash="c1"),
    class_id="bank_statement",
    confidence=0.87,
    bank="ktb",
    pages_used=[1, 2],
    invalid_reply=False,
    raw_reply='{"class_id": "bank_statement"}',
)

_THAI_TEXT = "ยอดยกมา 1,000.00"

_LAYOUT_BLOCK_TEXT = LayoutBlock(
    block_id="p1-b0",
    page=1,
    bbox=(10.0, 20.0, 200.0, 60.0),
    category="text",
    order=0,
    text=_THAI_TEXT,
)

_LAYOUT_BLOCK_TABLE = LayoutBlock(
    block_id="p1-b1",
    page=1,
    bbox=(10.0, 70.0, 400.0, 300.0),
    category="table",
    order=1,
    text=None,
    table_html="<table><tr><td>1,000.00</td></tr></table>",
)

_LAYOUT_PAGE = LayoutPage(
    page=1,
    width=1240,
    height=1754,
    blocks=[_LAYOUT_BLOCK_TEXT, _LAYOUT_BLOCK_TABLE],
)

_LAYOUT_ARTIFACT = LayoutArtifact(
    meta=ArtifactMeta(stage="layout", doc_id=DOC_ID, config_hash="c2"),
    model="paddleocr_vl",
    prompt_version="v1",
    pages=[_LAYOUT_PAGE],
)

_SEGMENT = Segment(
    segment_id="p1-b0-s0",
    block_id="p1-b0",
    page=1,
    bbox=(10.0, 20.0, 200.0, 35.0),
    order=0,
    row=None,
    col=None,
    fallback=False,
)

_SEGMENTS_ARTIFACT = SegmentsArtifact(
    meta=ArtifactMeta(stage="segments", doc_id=DOC_ID, config_hash="c3"),
    max_aspect=10.0,
    segments=[_SEGMENT],
)

_READING = Reading(
    segment_id="p1-b0-s0",
    text=_THAI_TEXT,
    char_conf=[0.9] * len(_THAI_TEXT),
    reader="paddle_crop",
    reader_version="v1",
)

_READINGS_ARTIFACT = ReadingsArtifact(
    meta=ArtifactMeta(stage="readings", doc_id=DOC_ID, config_hash="c4"),
    readings=[_READING],
)

_RECONCILED_SEGMENT = ReconciledSegment(
    segment_id="p1-b0-s0",
    block_id="p1-b0",
    text=_THAI_TEXT,
    sources=["layout", "paddle_crop"],
    conf=0.95,
    disputed=False,
    alt=[],
    single_source=False,
)

_RECONCILE_ARTIFACT = ReconcileArtifact(
    meta=ArtifactMeta(stage="reconcile", doc_id=DOC_ID, config_hash="c5"),
    segments=[_RECONCILED_SEGMENT],
)

_VERIFY_ATTEMPT = VerifyAttempt(
    field="account_no",
    region="region_hint",
    crop_path="crops/p1-b0.png",
    prompt="What is the account number?",
    reply="123-4-56789-0",
    readings={"paddle_crop": "123-4-56789-0"},
    decision="recovered",
)

_VERIFY_ARTIFACT = VerifyArtifact(
    meta=ArtifactMeta(stage="verify", doc_id=DOC_ID, config_hash="c6"),
    attempts=[_VERIFY_ATTEMPT],
    vlm_calls=3,
    budget_stop=None,
)

_FIELD_REPORT = FieldReport(
    name="account_no",
    state="found",
    value="123-4-56789-0",
    block_ids=["p1-b0"],
    page=1,
    bbox=(10.0, 20.0, 200.0, 35.0),
)

_FIELDS_REPORT = FieldsReport(
    doc_id=DOC_ID,
    class_id="bank_statement",
    verification_skipped=False,
    fields=[_FIELD_REPORT],
)

_PAGE_ERROR = PageError(page=2, stage="layout", message="timeout after retry")

_DOC_STATUS = DocStatus(
    doc_id=DOC_ID,
    status="partial",
    class_id="bank_statement",
    stages={"classify": "done", "layout": "done", "verify": "error"},
    page_errors=[_PAGE_ERROR],
    error=None,
)

ROUNDTRIP_INSTANCES = [
    _MODEL_IDENTITY,
    _ARTIFACT_META,
    _CLASSIFY_ARTIFACT,
    _LAYOUT_BLOCK_TEXT,
    _LAYOUT_PAGE,
    _LAYOUT_ARTIFACT,
    _SEGMENT,
    _SEGMENTS_ARTIFACT,
    _READING,
    _READINGS_ARTIFACT,
    _RECONCILED_SEGMENT,
    _RECONCILE_ARTIFACT,
    _VERIFY_ATTEMPT,
    _VERIFY_ARTIFACT,
    _FIELD_REPORT,
    _FIELDS_REPORT,
    _PAGE_ERROR,
    _DOC_STATUS,
]


@pytest.mark.parametrize("instance", ROUNDTRIP_INSTANCES, ids=lambda m: type(m).__name__)
def test_roundtrip(instance: BaseModel):
    cls = type(instance)
    restored = cls.model_validate_json(instance.model_dump_json())
    assert restored == instance


def test_every_schema_model_has_a_roundtrip_instance():
    covered = {type(m) for m in ROUNDTRIP_INSTANCES}
    defined = {
        obj
        for _, obj in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__
    }
    missing = defined - covered
    assert not missing, f"schema models without a round-trip test: {missing}"


def test_reading_requires_text_xor_error():
    with pytest.raises(ValidationError):
        Reading(
            segment_id="p1-b0-s0",
            text="hello",
            error="boom",
            reader="paddle_crop",
            reader_version="v1",
        )
    with pytest.raises(ValidationError):
        Reading(segment_id="p1-b0-s0", reader="paddle_crop", reader_version="v1")


def test_reading_char_conf_length_must_match_text():
    with pytest.raises(ValidationError):
        Reading(
            segment_id="p1-b0-s0",
            text="hello",
            char_conf=[0.9, 0.9],
            reader="paddle_crop",
            reader_version="v1",
        )


def test_reading_char_conf_values_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        Reading(
            segment_id="p1-b0-s0",
            text="hi",
            char_conf=[0.9, 1.5],
            reader="paddle_crop",
            reader_version="v1",
        )


def test_classify_artifact_confidence_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        ClassifyArtifact(
            meta=ArtifactMeta(stage="classify", doc_id=DOC_ID, config_hash="c1"),
            class_id="bank_statement",
            confidence=1.2,
            pages_used=[1],
        )


def test_layout_block_id_must_match_pattern():
    with pytest.raises(ValidationError):
        LayoutBlock(
            block_id="b3",
            page=1,
            bbox=(0.0, 0.0, 10.0, 10.0),
            category="text",
            order=0,
        )


def test_artifact_meta_doc_id_must_match_pattern():
    with pytest.raises(ValidationError):
        ArtifactMeta(stage="classify", doc_id="XYZ", config_hash="c1")


def test_doc_status_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        DocStatus.model_validate(
            {
                "doc_id": DOC_ID,
                "status": "complete",
                "unexpected_field": "nope",
            }
        )
