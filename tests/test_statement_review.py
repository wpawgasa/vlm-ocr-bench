"""TDD tests for the review queue export and the manual label loader (task 5.5)."""

import csv
from decimal import Decimal

import pytest
from typer.testing import CliRunner

from ocr_bench.cli import app
from ocr_bench.jsonl import read_rows, write_rows_atomic
from ocr_bench.metrics import tob_official
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import FieldResult, JsonGT, ManifestRow, TextLayerGT
from ocr_bench.statement_review import (
    QUEUE_COLUMNS,
    LabelError,
    load_labels,
    load_manual_pages,
)
from tests.test_statement_score import _prediction, _text_layer_gt

runner = CliRunner()
FAKE_PARITY = {
    t: {"n": 24, "max_abs_diff": 0.0, "comparable": True} for t in tob_official.KEPT_TASKS
}
SAMPLE = "BS-synth-0001-p1"

TABLE_M2 = [
    ["Date", "Description", "Withdrawal", "Deposit", "Balance"],
    ["01/03/24", "Brought Forward", "", "", "100.00"],
    ["04/03/24", "Fee", "21.00", "", "80.00"],
    ["07/03/24", "Salary", "", "50.00", "130.00"],
]


def _manifest_row():
    return ManifestRow(
        sample_id=SAMPLE,
        source="bankstmt",
        task="statement",
        domain="Finance",
        bank="synth",
        doc_type="digital",
        condition="clean",
        image_path=f"img/{SAMPLE}.png",
        gt_kind="text_layer",
        gt_path=f"gt/{SAMPLE}.json",
        critical_fields=["account_no", "opening_balance", "closing_balance"],
    )


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(tob_official, "run_parity", lambda: FAKE_PARITY)
    rp = RunPaths.for_run("r1")
    (rp.root / "gt").mkdir(parents=True)
    (rp.root / "gt" / f"{SAMPLE}.json").write_text(
        _text_layer_gt().model_dump_json(), encoding="utf-8"
    )
    write_rows_atomic(rp.manifest, [_manifest_row()])
    write_rows_atomic(
        rp.predictions,
        [_prediction(model="m1"), _prediction(model="m2", tables=[TABLE_M2])],
    )
    return rp


def _export(sample=None):
    args = ["review", "export", "--run-id", "r1"]
    if sample is not None:
        args += ["--agreement-sample", str(sample)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def _queue(rp):
    with open(rp.review_queue, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class TestExport:
    def test_export_writes_the_queue_with_one_column_per_model(self, run):
        _export()
        rows = _queue(run)
        assert list(rows[0]) == [*QUEUE_COLUMNS[:5], "m1_value", "m2_value", "label"]
        assert all(r["sample_id"] == SAMPLE and r["file_id"] == "BS-synth-0001" for r in rows)

    def test_every_disagreement_is_queued(self, run):
        _export()
        rows = {(r["field"], r["row_index"]): r for r in _queue(run)}
        debit = rows[("debit", "0")]
        assert debit["m1_value"] == "20.00" and debit["m2_value"] == "21.00"
        assert debit["label"] == ""

    def test_cells_no_model_saw_are_not_queued(self, run):
        _export(sample=1.0)
        rows = _queue(run)
        assert ("credit", "0") not in {(r["field"], r["row_index"]) for r in rows}
        assert ("account_name", "") not in {(r["field"], r["row_index"]) for r in rows}

    def test_export_is_deterministic(self, run):
        _export()
        first = run.review_queue.read_bytes()
        _export()
        assert run.review_queue.read_bytes() == first

    def test_agreements_are_sampled_not_all_exported(self, run):
        _export()
        rows = _queue(run)
        agreements = [r for r in rows if r["m1_value"] == r["m2_value"]]
        # 1 disagreement out of ~14 comparable cells: sampling must drop most agreements
        assert len(agreements) < 6


def _fill(rp, values):
    """Fill the exported queue's label column from `{(field, row_index): label}`."""
    rows = _queue(rp)
    known = {(r["field"], r["row_index"]) for r in rows}
    for field_row in values:
        assert field_row in known, f"{field_row} not in the queue: {sorted(known)}"
    for row in rows:
        row["label"] = values.get((row["field"], row["row_index"]), "")
    path = rp.root / "labels.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestLoad:
    def test_round_trip_writes_manual_ground_truth(self, run):
        _export(sample=1.0)
        path = _fill(run, {("debit", "0"): "22.00", ("account_no", ""): "111-2-33333-4"})
        result = runner.invoke(app, ["review", "load", "--run-id", "r1", "--labels", str(path)])
        assert result.exit_code == 0, result.output

        gt_path = run.gt_manual / f"{SAMPLE}.json"
        gt = JsonGT.model_validate_json(gt_path.read_text(encoding="utf-8"))
        assert gt.statement.account_no == "111-2-33333-4"
        assert gt.statement.rows[0].debit == Decimal("22.00")
        assert load_manual_pages(run)[SAMPLE].rows[0].debit == Decimal("22.00")

    def test_score_prefers_the_manual_ground_truth(self, run):
        _export(sample=1.0)
        path = _fill(run, {("account_no", ""): "999-9-99999-9"})
        assert (
            runner.invoke(app, ["review", "load", "--run-id", "r1", "--labels", str(path)])
        ).exit_code == 0
        assert runner.invoke(app, ["score", "--run-id", "r1"]).exit_code == 0
        fields = {
            (f.model, f.field): f
            for f in read_rows(run.fields, FieldResult)
            if f.sample_id == SAMPLE
        }
        # the text layer says 111-2-33333-4, the label says otherwise: the label wins
        assert fields[("m1", "account_no")].gt == "9999999999"  # digits only
        assert fields[("m1", "account_no")].field_exact == 0

    def test_unlabelled_rows_are_skipped(self, run):
        _export(sample=1.0)
        path = _fill(run, {("debit", "0"): "22.00"})
        pages = load_labels(path)
        assert pages[SAMPLE].account_no is None

    def test_invalid_amount_names_the_line(self, run):
        _export(sample=1.0)
        path = _fill(run, {("debit", "0"): "twenty-two baht"})
        with pytest.raises(LabelError) as exc:
            load_labels(path)
        assert "line" in str(exc.value) and "debit" in str(exc.value)

    def test_invalid_date_is_rejected_not_skipped(self, run):
        _export(sample=1.0)
        path = _fill(run, {("date", "0"): "32/13/2024"})
        with pytest.raises(LabelError):
            load_labels(path)

    def test_unknown_field_is_rejected(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text(
            "sample_id,page_no,file_id,field,row_index,m1_value,label\n"
            "BS-x-0001-p1,1,BS-x-0001,nonsense,,1,2\n",
            encoding="utf-8",
        )
        with pytest.raises(LabelError) as exc:
            load_labels(path)
        assert "nonsense" in str(exc.value)

    def test_cli_reports_the_line_and_exits_2(self, run):
        _export(sample=1.0)
        path = _fill(run, {("balance", "0"): "not a number"})
        result = runner.invoke(app, ["review", "load", "--run-id", "r1", "--labels", str(path)])
        assert result.exit_code == 2
        assert "line" in result.output
        assert not (run.gt_manual / f"{SAMPLE}.json").exists()


def test_gt_for_a_sample_without_labels_is_untouched(run):
    _export()
    assert not run.gt_manual.exists()
    gt = TextLayerGT.model_validate_json(
        (run.root / "gt" / f"{SAMPLE}.json").read_text(encoding="utf-8")
    )
    assert gt.gt_kind == "text_layer"
