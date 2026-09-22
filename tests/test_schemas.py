"""Round-trip and validation tests for ocr_bench.schemas."""

from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from ocr_bench.schemas import (
    GROUND_TRUTH,
    Block,
    BlockType,
    CalibrationRow,
    Condition,
    ErrorFlag,
    FieldResult,
    FieldValue,
    GtKind,
    HtmlGT,
    JsonGT,
    LatencyRecord,
    ManifestRow,
    NoGT,
    NormalizedPage,
    OutputField,
    OutputSource,
    PredictionRow,
    RawPrediction,
    ScoreRow,
    Source,
    StatementFile,
    StatementPage,
    StatementRow,
    Task,
    TextGT,
    TextLayerGT,
    Word,
)


def _roundtrip(model: BaseModel) -> None:
    cls = type(model)
    dumped = model.model_dump_json()
    restored = cls.model_validate_json(dumped)
    assert restored == model


def test_manifest_row_roundtrip():
    row = ManifestRow(
        sample_id="s1",
        source=Source.thaiocrbench,
        task=Task.ocr_fullpage,
        domain="Government",
        bank=None,
        doc_type=None,
        page_no=1,
        n_pages=1,
        condition=Condition.clean,
        image_path="img/s1.png",
        gt_kind=GtKind.text,
        gt_path="gt/s1.txt",
        critical_fields=["id_number"],
    )
    _roundtrip(row)


def test_manifest_row_page_no_gt_n_pages_raises():
    with pytest.raises(ValidationError):
        ManifestRow(
            sample_id="s1",
            source=Source.thaiocrbench,
            task=Task.ocr_fullpage,
            page_no=2,
            n_pages=1,
            condition=Condition.clean,
            image_path="img/s1.png",
            gt_kind=GtKind.none,
        )


def test_manifest_row_unknown_gt_kind_raises():
    with pytest.raises(ValidationError):
        ManifestRow(
            sample_id="s1",
            source=Source.thaiocrbench,
            task=Task.ocr_fullpage,
            page_no=1,
            n_pages=1,
            condition=Condition.clean,
            image_path="img/s1.png",
            gt_kind="not-a-kind",
        )


def test_manifest_row_forbids_unknown_key():
    with pytest.raises(ValidationError):
        ManifestRow(
            sample_id="s1",
            source=Source.thaiocrbench,
            task=Task.ocr_fullpage,
            page_no=1,
            n_pages=1,
            condition=Condition.clean,
            image_path="img/s1.png",
            gt_kind=GtKind.none,
            bogus_field="nope",
        )


def test_block_roundtrip():
    _roundtrip(Block(type=BlockType.text, text="hello", bbox=(0.0, 0.0, 1.0, 1.0), page=1))


def test_field_value_roundtrip():
    _roundtrip(
        FieldValue(
            value="152340.75",
            raw="฿152,340.75 บาท",
            confidence=0.98,
            bbox=(1.0, 2.0, 3.0, 4.0),
            flags=["unparseable"],
        )
    )


def test_normalized_page_roundtrip():
    page = NormalizedPage(
        text="hello world",
        blocks=[Block(type=BlockType.text, text="hi")],
        tables=[[["a", "b"], ["c", "d"]]],
        fields={"total": FieldValue(value="100")},
        label="invoice",
        parse_error=None,
    )
    _roundtrip(page)


def test_raw_prediction_roundtrip():
    _roundtrip(
        RawPrediction(
            text="hello",
            tokens=["he", "llo"],
            token_logprobs=[-0.1, -0.2],
            latency_ms=123.4,
            prompt_tokens=10,
            completion_tokens=2,
            error=None,
        )
    )


def test_prediction_row_roundtrip():
    row = PredictionRow(
        sample_id="s1",
        condition=Condition.clean,
        model="teleocr",
        prompt_version="ocr_fullpage-v0",
        raw=RawPrediction(text="hi"),
        normalized=NormalizedPage(text="hi"),
        latency_ms=100.0,
        prompt_tokens=5,
        completion_tokens=1,
    )
    _roundtrip(row)


def test_word_roundtrip():
    _roundtrip(Word(text="hello", bbox=(0.0, 0.0, 10.0, 10.0)))


def test_statement_row_roundtrip():
    _roundtrip(
        StatementRow(
            date="2025-03-15",
            description="transfer",
            debit=Decimal("100.50"),
            credit=None,
            balance=Decimal("1000.00"),
            channel="ATM",
            bbox=(0.0, 0.0, 1.0, 1.0),
            page_no=1,
        )
    )


def test_statement_page_roundtrip():
    page = StatementPage(
        bank="kbank",
        account_no="1234567890",
        account_name="Somchai",
        period_start="2025-01-01",
        period_end="2025-01-31",
        opening_balance=Decimal("1000.00"),
        closing_balance=Decimal("900.00"),
        rows=[StatementRow(date="2025-01-02", balance=Decimal("900.00"))],
        page_no=1,
    )
    _roundtrip(page)


def test_statement_file_rows_in_page_order():
    page2 = StatementPage(page_no=2, rows=[StatementRow(date="2025-01-02")])
    page1 = StatementPage(page_no=1, rows=[StatementRow(date="2025-01-01")])
    stmt = StatementFile(file_id="f1", bank="kbank", pages=[page2, page1])
    assert [r.date for r in stmt.rows] == ["2025-01-01", "2025-01-02"]


def test_statement_file_roundtrip():
    stmt = StatementFile(
        file_id="f1",
        bank="kbank",
        pages=[StatementPage(page_no=1, rows=[StatementRow(date="2025-01-01")])],
    )
    _roundtrip(stmt)


def test_decimal_survives_roundtrip_exactly():
    row = StatementRow(balance=Decimal("152340.75"))
    dumped = row.model_dump_json()
    restored = StatementRow.model_validate_json(dumped)
    assert restored.balance == Decimal("152340.75")
    assert str(restored.balance) == "152340.75"


def test_ground_truth_text():
    gt = GROUND_TRUTH.validate_python({"gt_kind": "text", "text": "hello"})
    assert isinstance(gt, TextGT)
    dumped = GROUND_TRUTH.dump_json(gt)
    restored = GROUND_TRUTH.validate_json(dumped)
    assert restored == gt


def test_ground_truth_html():
    gt = HtmlGT(gt_kind="html", html="<table></table>")
    dumped = GROUND_TRUTH.dump_json(gt)
    restored = GROUND_TRUTH.validate_json(dumped)
    assert restored == gt


def test_ground_truth_json():
    gt = JsonGT(gt_kind="json", fields={"id_number": "123"}, label=None, statement=None)
    dumped = GROUND_TRUTH.dump_json(gt)
    restored = GROUND_TRUTH.validate_json(dumped)
    assert restored == gt


def test_ground_truth_text_layer_roundtrip_via_validate_json():
    gt = TextLayerGT(
        gt_kind="text_layer",
        gt_text="hello world",
        gt_words=[Word(text="hello", bbox=(0.0, 0.0, 5.0, 5.0))],
        statement=None,
    )
    dumped = GROUND_TRUTH.dump_json(gt)
    restored = GROUND_TRUTH.validate_json(dumped)
    assert isinstance(restored, TextLayerGT)
    assert restored == gt


def test_ground_truth_none():
    gt = NoGT(gt_kind="none")
    dumped = GROUND_TRUTH.dump_json(gt)
    restored = GROUND_TRUTH.validate_json(dumped)
    assert restored == gt


def test_ground_truth_unknown_gt_kind_raises():
    with pytest.raises(ValidationError):
        GROUND_TRUTH.validate_python({"gt_kind": "bogus"})


def test_score_row_roundtrip():
    row = ScoreRow(
        sample_id="s1",
        condition=Condition.clean,
        model="teleocr",
        task=Task.ocr_fullpage,
        metric="cer",
        value=0.05,
        domain="Government",
        bank=None,
        doc_type=None,
        is_critical=True,
        extra={"n_tokens": 42},
    )
    _roundtrip(row)


def test_field_result_roundtrip():
    row = FieldResult(
        sample_id="s1",
        condition=Condition.clean,
        model="teleocr",
        task=Task.kie,
        field="id_number",
        is_critical=True,
        pred="123",
        gt="123",
        field_exact=1,
        field_fuzzy=1,
        false_accept=False,
        false_reject=False,
        domain=None,
        bank=None,
        doc_type=None,
    )
    _roundtrip(row)


def test_calibration_row_roundtrip():
    row = CalibrationRow(
        model="teleocr",
        variant="full",
        band=1,
        lo=0.0,
        hi=0.2,
        n=50,
        accuracy=0.9,
        ci_low=0.85,
        ci_high=0.95,
        critical_only=False,
    )
    _roundtrip(row)


def test_latency_record_roundtrip():
    row = LatencyRecord(
        model="teleocr",
        gpu="h100",
        concurrency=8,
        sample_id="s1",
        latency_ms=250.0,
        ttft_ms=50.0,
        prompt_tokens=100,
        completion_tokens=20,
        image_px=1000000,
        error=None,
        warmup=False,
    )
    _roundtrip(row)


def test_output_field_roundtrip():
    field = OutputField(
        field="total_amount",
        value=152340.75,
        raw="฿152,340.75 บาท",
        type="amount",
        unit="THB",
        definition="Total statement amount",
        source=OutputSource(file="stmt.pdf", page=1, bbox=(0.0, 0.0, 1.0, 1.0)),
        normalization=["strip_currency", "parse_decimal"],
        confidence=0.95,
        error_flags=[ErrorFlag.low_confidence],
        model="teleocr",
        prompt_version="statement-v0",
        run_id="run-1",
    )
    _roundtrip(field)


def test_output_field_unknown_error_flag_raises():
    with pytest.raises(ValidationError):
        OutputField(
            field="total_amount",
            value=1.0,
            raw="1.0",
            type="amount",
            definition="x",
            source=OutputSource(file="a.pdf", page=1),
            error_flags=["not_a_flag"],
            model="teleocr",
            prompt_version="v0",
            run_id="run-1",
        )
