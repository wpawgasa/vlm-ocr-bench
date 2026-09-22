"""Tests for scorer dispatch and `ocrbench score` (task 4.7)."""

import json

import pytest
from typer.testing import CliRunner

from ocr_bench.cli import app
from ocr_bench.jsonl import read_rows, write_rows_atomic
from ocr_bench.metrics import tob_official
from ocr_bench.metrics.dispatch import answer_text, is_full_miss
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import (
    FieldResult,
    FieldValue,
    HtmlGT,
    JsonGT,
    ManifestRow,
    NoGT,
    NormalizedPage,
    PredictionRow,
    RawPrediction,
    ScoreRow,
    TextGT,
    TextLayerGT,
)

runner = CliRunner()

needs_wordnet = pytest.mark.skipif(
    not tob_official.wordnet_available(),
    reason="NLTK WordNet missing: run `uv run python -m nltk.downloader wordnet omw-1.4`",
)

TABLE = "<table><tr><th>ก</th><th>ข</th></tr><tr><td>1</td><td>2</td></tr></table>"
KIE_RAW = '{"ชื่อ": "สมชาย", "ยอดรวม": "1,000.00 บาท", "หมายเหตุ": ""}'
FAKE_PARITY = {
    t: {"n": 24, "max_abs_diff": 0.0, "comparable": True} for t in tob_official.KEPT_TASKS
}


def _manifest(sample_id, task, subtask, gt_kind, condition="clean", **kw):
    return ManifestRow(
        sample_id=sample_id,
        source=kw.pop("source", "thaiocrbench"),
        task=task,
        subtask=subtask,
        domain=kw.pop("domain", "Government"),
        condition=condition,
        image_path=f"img/{sample_id}.png",
        gt_kind=gt_kind,
        gt_path=None if gt_kind == "none" else f"gt/{sample_id}.json",
        **kw,
    )


def _pred(sample_id, text, condition="clean", *, version="dots-ocr-v1", error=None, **norm):
    return PredictionRow(
        sample_id=sample_id,
        condition=condition,
        model="m1",
        prompt_version=version,
        raw=RawPrediction(text=text, error=error),
        normalized=NormalizedPage(text=text if error is None or text else "", **norm),
        latency_ms=1.0,
        prompt_tokens=1,
        completion_tokens=1,
    )


def _build_run(root):
    rp = RunPaths(root=root)
    (root / "gt").mkdir(parents=True)
    gts = {
        "TOB-fullpage-1": TextGT(gt_kind="text", text="สวัสดี ครับ วันนี้ อากาศ ดี"),
        "TOB-textrec-1": TextGT(gt_kind="text", text="ใบเสร็จรับเงิน"),
        "TOB-table-1": HtmlGT(gt_kind="html", html=TABLE, raw=TABLE),
        "TOB-docparse-1": TextGT(gt_kind="text", text="# หัวข้อ\nเนื้อหา"),
        "TOB-kie-1": JsonGT(
            gt_kind="json",
            fields={"ชื่อ": "สมชาย", "ยอดรวม": "1,000.00 บาท", "หมายเหตุ": ""},
            raw=KIE_RAW,
        ),
        "TOB-classify-1": JsonGT(gt_kind="json", label="ใบเสร็จ", raw="ใบเสร็จ"),
        "BS-kbank-0001-p1": TextLayerGT(gt_kind="text_layer", gt_text="ยอดยกมา 1,000.00"),
        "TOB-odd-1": TextGT(gt_kind="text", text="x"),
    }
    for sid, gt in gts.items():
        (root / "gt" / f"{sid}.json").write_text(gt.model_dump_json(), encoding="utf-8")
    assert isinstance(NoGT(gt_kind="none"), NoGT)

    manifest = [
        _manifest("TOB-fullpage-1", "ocr_fullpage", "Full-page OCR", "text"),
        _manifest("TOB-fullpage-1", "ocr_fullpage", "Full-page OCR", "text", "scan_low"),
        _manifest("TOB-textrec-1", "ocr_line", "Text recognition", "text"),
        _manifest("TOB-table-1", "table", "Table parsing", "html"),
        _manifest("TOB-docparse-1", "docparse", "Document parsing", "text"),
        _manifest(
            "TOB-kie-1",
            "kie",
            "Key information extraction",
            "json",
            domain="Finance",
            critical_fields=["ยอดรวม"],
        ),
        _manifest(
            "TOB-kie-1",
            "kie",
            "Key information extraction",
            "json",
            "scan_low",
            domain="Finance",
            critical_fields=["ยอดรวม"],
        ),
        _manifest("TOB-classify-1", "classify", "Document classification", "json"),
        _manifest(
            "BS-kbank-0001-p1",
            "statement",
            None,
            "text_layer",
            source="bankstmt",
            domain="Finance",
            bank="kbank",
            doc_type="digital",
        ),
        _manifest("BS-scan-0001-p1", "statement", None, "none", source="bankstmt", bank="scb"),
        _manifest("TOB-odd-1", "table", "Table parsing", "text"),
    ]
    write_rows_atomic(rp.manifest, manifest)

    kie_fields = {
        "ชื่อ": FieldValue(value="สมชาย", raw="สมชาย"),
        "ยอดรวม": FieldValue(value="1000", raw="1000"),
        "เพิ่ม": FieldValue(value="x", raw="x"),
    }
    preds = [
        _pred("TOB-fullpage-1", "สวัสดี ครับ วันนี้ อากาศ ดี"),
        # superseded by the retried row below: last row per key wins
        _pred("TOB-fullpage-1", "", "scan_low", error="APIError: boom"),
        _pred("TOB-fullpage-1", "สวัสดี ครับ", "scan_low", error="1 of 3 blocks failed"),
        _pred("TOB-textrec-1", "ใบเสร็จรับเงิน", version="question-v1"),
        _pred("TOB-table-1", "หัว\n" + TABLE, tables=[[["ก", "ข"], ["1", "2"]]]),
        _pred("TOB-docparse-1", "# หัวข้อ\nเนื้อหา"),
        _pred("TOB-kie-1", json.dumps({"ชื่อ": "สมชาย"}), version="question-v1", fields=kie_fields),
        _pred("TOB-kie-1", "", "scan_low", version="question-v1", error="TimeoutError: slow"),
        _pred("TOB-classify-1", "ใบเสร็จ", version="question-v1", label="ใบเสร็จ"),
        _pred("BS-kbank-0001-p1", "ยอดยกมา 1,000.00"),
        _pred("BS-scan-0001-p1", "whatever"),
        _pred("TOB-odd-1", "x"),
    ]
    write_rows_atomic(rp.predictions, preds)
    return rp


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(tob_official, "run_parity", lambda: FAKE_PARITY)
    return _build_run(tmp_path / "runs" / "r1")


def _scores(rp):
    return {(r.sample_id, r.condition.value, r.metric): r for r in read_rows(rp.scores, ScoreRow)}


def test_full_miss_and_answer_text_rules():
    assert is_full_miss(_pred("s", "", error="APIError: x"))
    assert is_full_miss(_pred("s", "text", error="APIError: x"))
    assert not is_full_miss(_pred("s", "text", error="2 of 5 blocks failed"))
    assert is_full_miss(_pred("s", "", error="2 of 5 blocks failed"))
    assert not is_full_miss(_pred("s", "ok"))
    p = _pred("s", "raw reply", version="question-v1")
    p.normalized.text = "normalized"
    assert answer_text(p) == "raw reply"
    p.prompt_version = "dots-layout-all-v1"
    assert answer_text(p) == "normalized"
    assert answer_text(_pred("s", "", error="APIError: x")) is None


@needs_wordnet
def test_score_writes_every_branch(run):
    result = runner.invoke(app, ["score", "--run-id", "r1"])
    assert result.exit_code == 0, result.output
    s = _scores(run)

    # text tasks
    for m in ("cer", "wer", "bmfl", "tob_score"):
        assert ("TOB-fullpage-1", "clean", m) in s
    assert s[("TOB-fullpage-1", "clean", "cer")].value == 0.0
    assert s[("TOB-fullpage-1", "clean", "tob_score")].value == pytest.approx(
        s[("TOB-fullpage-1", "clean", "bmfl")].value
    )
    assert s[("TOB-textrec-1", "clean", "tob_score")].value == 1.0
    assert s[("TOB-textrec-1", "clean", "tob_score")].subtask == "Text recognition"

    # partial page is scored, not worst-cased
    partial = s[("TOB-fullpage-1", "scan_low", "cer")]
    assert 0.0 < partial.value < 1.0
    assert partial.extra.get("partial") is True

    # table / docparse / classify
    assert s[("TOB-table-1", "clean", "ted")].value == 1.0
    assert s[("TOB-table-1", "clean", "tob_score")].value == 1.0
    assert ("TOB-table-1", "clean", "cer") in s
    assert s[("TOB-docparse-1", "clean", "ted")].value == 1.0
    assert s[("TOB-docparse-1", "clean", "tob_score")].value == 1.0
    assert s[("TOB-classify-1", "clean", "anls")].value == 1.0
    assert s[("TOB-classify-1", "clean", "tob_score")].value == 1.0

    # kie
    kie_f1 = s[("TOB-kie-1", "clean", "kie_f1")]
    assert kie_f1.domain == "Finance" and kie_f1.is_critical is None
    assert s[("TOB-kie-1", "clean", "kie_precision")].value == pytest.approx(2 / 3)
    assert s[("TOB-kie-1", "clean", "kie_recall")].value == 1.0

    # statement / no gt / not applicable
    stmt = s[("BS-kbank-0001-p1", "clean", "cer")]
    assert stmt.value == 0.0 and stmt.bank == "kbank" and stmt.doc_type == "digital"
    assert s[("BS-scan-0001-p1", "clean", "no_gt")].value is None
    assert s[("TOB-odd-1", "clean", "not_applicable")].value is None

    fields = [r for r in read_rows(run.fields, FieldResult) if r.condition.value == "clean"]
    by_field = {r.field: r for r in fields}
    assert by_field["ยอดรวม"].is_critical is True and by_field["ยอดรวม"].field_exact == 1
    assert by_field["ชื่อ"].is_critical is False
    assert by_field["เพิ่ม"].gt is None and by_field["เพิ่ม"].false_accept is True
    assert by_field["หมายเหตุ"].field_exact == 1  # blank == blank
    assert all(r.subtask == "Key information extraction" for r in fields)

    parity = json.loads((run.root / "tob_parity.json").read_text(encoding="utf-8"))
    assert parity == FAKE_PARITY
    assert (run.root / "aggregates.jsonl").exists()


@needs_wordnet
def test_errored_rows_are_worst_case_and_never_omitted(run):
    assert runner.invoke(app, ["score", "--run-id", "r1"]).exit_code == 0
    s = _scores(run)
    for m in ("kie_precision", "kie_recall", "kie_f1", "tob_score"):
        row = s[("TOB-kie-1", "scan_low", m)]
        assert row.value == 0.0
        assert row.extra["full_miss"] is True
    errored = [r for r in read_rows(run.fields, FieldResult) if r.condition.value == "scan_low"]
    assert {r.field for r in errored} == {"ชื่อ", "ยอดรวม", "หมายเหตุ"}
    assert all(r.field_exact == 0 and r.field_fuzzy == 0 for r in errored)
    fr = {r.field: r.false_reject for r in errored}
    assert fr == {"ชื่อ": True, "ยอดรวม": True, "หมายเหตุ": False}

    # every manifest key has score rows
    keys = {(r.sample_id, r.condition.value) for r in read_rows(run.scores, ScoreRow)}
    manifest = {(r.sample_id, r.condition.value) for r in read_rows(run.manifest, ManifestRow)}
    assert keys == manifest


@needs_wordnet
def test_missing_prediction_is_worst_case(run):
    preds = [
        p for p in read_rows(run.predictions, PredictionRow) if p.sample_id != "TOB-classify-1"
    ]
    write_rows_atomic(run.predictions, preds)
    assert runner.invoke(app, ["score", "--run-id", "r1"]).exit_code == 0
    s = _scores(run)
    row = s[("TOB-classify-1", "clean", "anls")]
    assert row.value == 0.0 and row.extra["missing_prediction"] is True


@needs_wordnet
def test_score_twice_is_byte_identical(run):
    assert runner.invoke(app, ["score", "--run-id", "r1"]).exit_code == 0
    names = ["scores.jsonl", "fields.jsonl", "aggregates.jsonl", "tob_parity.json"]
    first = {n: (run.root / n).read_bytes() for n in names}
    assert runner.invoke(app, ["score", "--run-id", "r1"]).exit_code == 0
    second = {n: (run.root / n).read_bytes() for n in names}
    assert first == second


def test_score_missing_predictions_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    rp = RunPaths.for_run("r2")
    write_rows_atomic(rp.manifest, [])
    result = runner.invoke(app, ["score", "--run-id", "r2"])
    assert result.exit_code == 2
    assert "ocrbench infer" in result.output
