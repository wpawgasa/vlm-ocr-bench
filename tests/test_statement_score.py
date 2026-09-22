"""TDD tests for statement scoring inside `score`: Check A metrics in the dispatcher and
the file-level Check B/C run (tasks 5.2, 5.3, 5.4, 5.6). Synthetic pages only."""

import json
from decimal import Decimal

import pytest
from typer.testing import CliRunner

from ocr_bench.cli import app
from ocr_bench.jsonl import read_rows, write_rows_atomic
from ocr_bench.metrics import tob_official
from ocr_bench.metrics.dispatch import score_sample
from ocr_bench.metrics.statement_run import StatementRecord, file_id_for, run_statement_checks
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import (
    Block,
    BlockType,
    FieldResult,
    JsonGT,
    ManifestRow,
    NormalizedPage,
    PredictionRow,
    RawPrediction,
    ScoreRow,
    StatementPage,
    StatementRow,
    TextLayerGT,
    Word,
)

runner = CliRunner()
FAKE_PARITY = {
    t: {"n": 24, "max_abs_diff": 0.0, "comparable": True} for t in tob_official.KEPT_TASKS
}

TABLE = [
    ["Date", "Description", "Withdrawal", "Deposit", "Balance"],
    ["01/03/24", "Brought Forward", "", "", "100.00"],
    ["04/03/24", "Fee", "20.00", "", "80.00"],
    ["07/03/24", "Salary", "", "50.00", "130.00"],
]
HEADER_BLOCK = Block(
    type=BlockType.text,
    text="Account Number 111-2-33333-4\nPeriod 01/03/2024 - 31/03/2024\nClosing Balance 130.00",
)


def _line(y, items):
    return [Word(text=t, bbox=(x0, float(y), x1, float(y + 18))) for t, x0, x1 in items]


def _gt_words():
    return [
        *_line(100, [("Account", 100, 200), ("Number", 210, 300), ("111-2-33333-4", 320, 470)]),
        *_line(
            140,
            [
                ("Period", 100, 170),
                ("01/03/2024", 200, 320),
                ("-", 330, 340),
                ("31/03/2024", 350, 470),
            ],
        ),
        *_line(180, [("Closing", 100, 180), ("Balance", 190, 280), ("130.00", 400, 470)]),
        *_line(
            300,
            [
                ("Date", 100, 160),
                ("Description", 260, 400),
                ("Withdrawal", 600, 720),
                ("Deposit", 820, 910),
                ("Balance", 1020, 1110),
            ],
        ),
        *_line(340, [("01/03/24", 100, 190), ("Brought", 260, 340), ("Forward", 345, 420)]),
        *_line(340, [("100.00", 1040, 1110)]),
        *_line(
            380,
            [("04/03/24", 100, 190), ("Fee", 260, 300), ("20.00", 660, 720), ("80.00", 1050, 1110)],
        ),
        *_line(
            420,
            [
                ("07/03/24", 100, 190),
                ("Salary", 260, 340),
                ("50.00", 850, 910),
                ("130.00", 1040, 1110),
            ],
        ),
    ]


def _text_layer_gt():
    return TextLayerGT(
        gt_kind="text_layer",
        gt_text="Account Number 111-2-33333-4 Period 01/03/2024 - 31/03/2024",
        gt_words=_gt_words(),
    )


def _manifest_row(sample_id="BS-synth-0001-p1", condition="clean", gt_kind="text_layer", **kw):
    return ManifestRow(
        sample_id=sample_id,
        source="bankstmt",
        task="statement",
        domain="Finance",
        bank=kw.pop("bank", "synth"),
        doc_type=kw.pop("doc_type", "digital"),
        condition=condition,
        image_path=f"img/{sample_id}.png",
        gt_kind=gt_kind,
        gt_path=f"gt/{sample_id}.json",
        critical_fields=["account_no", "opening_balance", "closing_balance"],
        **kw,
    )


def _prediction(
    sample_id="BS-synth-0001-p1", model="m1", condition="clean", tables=None, blocks=None
):
    normalized = NormalizedPage(
        text="statement page",
        blocks=[HEADER_BLOCK] if blocks is None else blocks,
        tables=[TABLE] if tables is None else tables,
    )
    return PredictionRow(
        sample_id=sample_id,
        condition=condition,
        model=model,
        prompt_version="dots-layout-all-v1",
        raw=RawPrediction(text="statement page"),
        normalized=normalized,
        latency_ms=1.0,
        prompt_tokens=1,
        completion_tokens=1,
    )


# --- Check A through the dispatcher -----------------------------------------------------------


class TestDispatchTextLayer:
    def _scores(self, **kw):
        result = score_sample(_manifest_row(), _text_layer_gt(), _prediction(**kw), "m1")
        return {s.metric: s for s in result.scores}, result.fields

    def test_row_metrics_and_cer(self):
        scores, fields = self._scores()
        assert scores["row_f1"].value == 1.0
        assert scores["row_precision"].value == 1.0
        assert scores["row_recall"].value == 1.0
        assert "cer" in scores
        assert scores["row_f1"].bank == "synth" and scores["row_f1"].doc_type == "digital"

    def test_header_fields_are_scored_with_criticality(self):
        _, fields = self._scores()
        by_field = {f.field: f for f in fields}
        assert by_field["account_no"].field_exact == 1
        assert by_field["account_no"].is_critical is True
        assert by_field["closing_balance"].field_exact == 1
        assert by_field["account_name"].is_critical is False
        assert by_field["period_start"].gt == "2024-03-01"

    def test_wrong_rows_lower_the_f1(self):
        bad = [row[:] for row in TABLE]
        bad[2][2] = "999.00"
        scores, _ = self._scores(tables=[bad])
        assert scores["row_f1"].value == pytest.approx(0.5)

    def test_missing_header_marks_rows_not_applicable(self):
        gt = TextLayerGT(
            gt_kind="text_layer",
            gt_text="Dear customer",
            gt_words=_line(100, [("Dear", 100, 160), ("customer", 170, 280)]),
        )
        result = score_sample(_manifest_row(), gt, _prediction(), "m1")
        scores = {s.metric: s for s in result.scores}
        assert scores["row_f1"].value is None
        assert scores["row_f1"].extra["reason"] == "no_header"

    def test_full_miss_still_lists_the_statement_metrics(self):
        result = score_sample(_manifest_row(), _text_layer_gt(), None, "m1")
        scores = {s.metric: s for s in result.scores}
        assert scores["cer"].value == 1.0
        assert scores["row_f1"].value == 0.0
        assert all(f.field_exact == 0 for f in result.fields)
        assert {f.field for f in result.fields} >= {"account_no", "closing_balance"}


class TestDispatchManualLabels:
    def test_manual_json_statement_is_scored(self):
        gt = JsonGT(
            gt_kind="json",
            statement=StatementPage(
                account_no="111-2-33333-4",
                closing_balance=Decimal("130.00"),
                rows=[
                    StatementRow(date="2024-03-04", debit=Decimal("20.00")),
                    StatementRow(date="2024-03-07", credit=Decimal("50.00")),
                ],
            ),
        )
        result = score_sample(_manifest_row(gt_kind="json"), gt, _prediction(), "m1")
        scores = {s.metric: s for s in result.scores}
        assert scores["row_f1"].value == 1.0
        by_field = {f.field: f for f in result.fields}
        assert by_field["account_no"].field_exact == 1
        assert by_field["account_name"].gt is None


# --- Checks B and C across a file --------------------------------------------------------------


def _run_checks(predictions, manifest=None):
    manifest = manifest or [_manifest_row()]
    latest = {(p.sample_id, p.condition.value, p.model): p for p in predictions}
    return run_statement_checks(manifest, latest, {}, {})


def test_file_id_strips_the_page_suffix():
    assert file_id_for("BS-kbank-0007-p12") == "BS-kbank-0007"
    assert file_id_for("BS-kbank-0007") == "BS-kbank-0007"


class TestStatementRun:
    def test_arithmetic_score_rows_and_record(self):
        scores, records = _run_checks([_prediction()])
        by_metric = {s.metric: s for s in scores}
        assert by_metric["file_reconciles"].value == 1.0
        assert by_metric["row_consistency_rate"].value == 1.0
        assert by_metric["file_reconciles"].extra["file_id"] == "BS-synth-0001"
        assert len(records) == 1
        record = records[0]
        assert isinstance(record, StatementRecord)
        assert record.arithmetic.file_reconciles is True
        assert record.statement.pages[0].opening_balance == Decimal("100.00")

    def test_broken_file_reports_the_first_break(self):
        bad = [row[:] for row in TABLE]
        bad[3][4] = "999.00"
        scores, records = _run_checks([_prediction(tables=[bad])])
        by_metric = {s.metric: s for s in scores}
        assert by_metric["file_reconciles"].value == 0.0
        assert by_metric["row_consistency_rate"].value == 0.5
        assert records[0].arithmetic.first_break_row == 2

    def test_agreement_rate_row_per_model_pair(self):
        other = [row[:] for row in TABLE]
        other[2][2] = "21.00"
        scores, _ = _run_checks([_prediction(model="m1"), _prediction(model="m2", tables=[other])])
        agreement = [s for s in scores if s.metric == "agreement_rate"]
        assert len(agreement) == 1
        assert agreement[0].model == "m1|m2"
        assert 0.0 < agreement[0].value < 1.0

    def test_multipage_file_is_merged(self):
        manifest = [_manifest_row(), _manifest_row("BS-synth-0001-p2", page_no=2, n_pages=2)]
        manifest[0].n_pages = 2
        page2 = [
            ["Date", "Description", "Withdrawal", "Deposit", "Balance"],
            ["09/03/24", "Fee", "30.00", "", "100.00"],
        ]
        scores, records = _run_checks(
            # neither page carries a summary block: the file's closing balance is unknown
            [
                _prediction(blocks=[]),
                _prediction("BS-synth-0001-p2", tables=[page2], blocks=[]),
            ],
            manifest,
        )
        assert len(records) == 1
        assert [p.page_no for p in records[0].statement.pages] == [1, 2]
        assert records[0].arithmetic.n_rows == 3
        assert records[0].arithmetic.file_reconciles is True


class TestFlagPrecision:
    """A labelled page tells whether Check B's and Check C's flags were true errors."""

    def _labels(self, debit="20.00"):
        return {
            "BS-synth-0001-p1": StatementPage(
                account_no="111-2-33333-4",
                closing_balance=Decimal("130.00"),
                rows=[
                    StatementRow(date="2024-03-04", debit=Decimal(debit), balance=Decimal("80.00")),
                    StatementRow(
                        date="2024-03-07", credit=Decimal("50.00"), balance=Decimal("130.00")
                    ),
                ],
            )
        }

    def _metric(self, scores, metric, model="m1"):
        return next(s for s in scores if s.metric == metric and s.model == model)

    def test_check_b_flag_is_true_when_the_label_differs(self):
        bad = [row[:] for row in TABLE]
        bad[2][2] = "21.00"  # a misread debit: the balance equation breaks and it is wrong
        latest = {("BS-synth-0001-p1", "clean", "m1"): _prediction(tables=[bad])}
        scores, _ = run_statement_checks([_manifest_row()], latest, {}, self._labels())
        assert self._metric(scores, "arith_flag_precision").value == 1.0

    def test_check_b_flag_is_false_when_the_label_agrees(self):
        # the extraction matches the label, but the printed balance chain does not add up
        labels = self._labels()
        labels["BS-synth-0001-p1"].rows[0].balance = Decimal("80.00")
        bad = [row[:] for row in TABLE]
        bad[1][4] = "500.00"  # a wrong opening balance breaks row 1's equation only
        latest = {("BS-synth-0001-p1", "clean", "m1"): _prediction(tables=[bad])}
        scores, _ = run_statement_checks([_manifest_row()], latest, {}, labels)
        assert self._metric(scores, "arith_flag_precision").value == 0.0

    def test_check_c_disagreement_precision(self):
        other = [row[:] for row in TABLE]
        other[2][2] = "21.00"  # m2 is wrong here, so the disagreement is a true error
        latest = {
            ("BS-synth-0001-p1", "clean", "m1"): _prediction(model="m1"),
            ("BS-synth-0001-p1", "clean", "m2"): _prediction(model="m2", tables=[other]),
        }
        scores, _ = run_statement_checks([_manifest_row()], latest, {}, self._labels())
        precision = self._metric(scores, "agreement_flag_precision", model="m1|m2")
        assert precision.value == 1.0


# --- end to end ---------------------------------------------------------------------------------


@pytest.fixture
def statement_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(tob_official, "run_parity", lambda: FAKE_PARITY)
    rp = RunPaths.for_run("r1")
    (rp.root / "gt").mkdir(parents=True)
    (rp.root / "gt" / "BS-synth-0001-p1.json").write_text(
        _text_layer_gt().model_dump_json(), encoding="utf-8"
    )
    write_rows_atomic(rp.manifest, [_manifest_row()])
    write_rows_atomic(rp.predictions, [_prediction(model="m1"), _prediction(model="m2")])
    return rp


def test_score_writes_statements_and_statement_aggregates(statement_run):
    result = runner.invoke(app, ["score", "--run-id", "r1"])
    assert result.exit_code == 0, result.output

    records = [json.loads(line) for line in statement_run.statements.read_text().splitlines()]
    assert {r["model"] for r in records} == {"m1", "m2"}
    assert records[0]["arithmetic"]["file_reconciles"] is True

    metrics = {(s.model, s.metric) for s in read_rows(statement_run.scores, ScoreRow)}
    assert ("m1", "row_f1") in metrics
    assert ("m1", "file_reconciles") in metrics
    assert ("m1|m2", "agreement_rate") in metrics
    fields = list(read_rows(statement_run.fields, FieldResult))
    assert {f.field for f in fields} >= {"account_no", "closing_balance"}


def test_score_is_byte_identical_on_a_rerun(statement_run):
    assert runner.invoke(app, ["score", "--run-id", "r1"]).exit_code == 0
    first = [p.read_bytes() for p in (statement_run.scores, statement_run.statements)]
    assert runner.invoke(app, ["score", "--run-id", "r1"]).exit_code == 0
    assert [p.read_bytes() for p in (statement_run.scores, statement_run.statements)] == first
