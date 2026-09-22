"""TDD tests for ocr_bench.data.manifest.materialize and the `ocrbench prepare` CLI.

No network / Hugging Face hub access: ThaiOCRBench rows come from monkeypatching
`ocr_bench.data.thaiocrbench.THAIOCRBENCH_ROWS_PROVIDER`.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from ocr_bench.cli import app
from ocr_bench.data.manifest import PreparedSample, materialize
from ocr_bench.jsonl import read_rows
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import (
    GROUND_TRUTH,
    Condition,
    GtKind,
    ManifestRow,
    NoGT,
    Source,
    Task,
    TextGT,
    TextLayerGT,
    Word,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _checkerboard(h=32, w=48):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[::4, ::4] = (200, 100, 50)
    img[1::4, 2::4] = (10, 220, 30)
    return img


def _text_sample(sample_id="TOB-fullpage-1", text="hello world"):
    return PreparedSample(
        sample_id=sample_id,
        source=Source.thaiocrbench,
        task=Task.ocr_fullpage,
        subtask="Full-page OCR",
        domain="Government",
        bank=None,
        doc_type=None,
        page_no=1,
        n_pages=1,
        question="Read the page.",
        image=_checkerboard(),
        gt=TextGT(gt_kind="text", text=text),
        critical_fields=[],
    )


def _no_gt_sample(sample_id="BS-kbank-0001-p1"):
    return PreparedSample(
        sample_id=sample_id,
        source=Source.bankstmt,
        task=Task.statement,
        subtask=None,
        domain="Finance",
        bank="kbank",
        doc_type="scanned",
        page_no=1,
        n_pages=1,
        question="",
        image=_checkerboard(h=40, w=40),
        gt=NoGT(gt_kind="none"),
        critical_fields=["account_no"],
    )


def _text_layer_sample(sample_id="BS-kbank-0001-p1"):
    return PreparedSample(
        sample_id=sample_id,
        source=Source.bankstmt,
        task=Task.statement,
        subtask=None,
        domain="Finance",
        bank="kbank",
        doc_type="digital",
        page_no=1,
        n_pages=1,
        question="",
        image=_checkerboard(h=60, w=80),
        gt=TextLayerGT(
            gt_kind="text_layer",
            gt_text="Balance 1,000.00",
            gt_words=[
                Word(text="Balance", bbox=(5.0, 5.0, 25.0, 15.0)),
                Word(text="1,000.00", bbox=(30.0, 5.0, 60.0, 15.0)),
            ],
        ),
        critical_fields=["account_no"],
    )


class TestMaterialize:
    def test_row_count_is_conditions_times_samples(self, tmp_path):
        rp = RunPaths(root=tmp_path / "r1")
        samples = [_text_sample("s1"), _no_gt_sample("s2")]
        rows = materialize(samples, [Condition.clean, Condition.scan_low, Condition.photo], rp)
        assert len(rows) == 6

    def test_image_files_exist(self, tmp_path):
        rp = RunPaths(root=tmp_path / "r1")
        samples = [_text_sample("s1")]
        rows = materialize(samples, [Condition.clean, Condition.scan_low, Condition.photo], rp)
        for row in rows:
            assert (rp.root / row.image_path).exists()

    def test_gt_json_validates(self, tmp_path):
        rp = RunPaths(root=tmp_path / "r1")
        samples = [_text_sample("s1")]
        rows = materialize(samples, [Condition.clean], rp)
        row = rows[0]
        assert row.gt_path is not None
        data = (rp.root / row.gt_path).read_text(encoding="utf-8")
        gt = GROUND_TRUTH.validate_json(data)
        assert gt.text == "hello world"

    def test_no_gt_has_none_path_and_kind(self, tmp_path):
        rp = RunPaths(root=tmp_path / "r1")
        samples = [_no_gt_sample("s2")]
        rows = materialize(samples, [Condition.clean, Condition.scan_low, Condition.photo], rp)
        for row in rows:
            assert row.gt_path is None
            assert row.gt_kind == GtKind.none

    def test_clean_image_path_has_no_condition_suffix(self, tmp_path):
        rp = RunPaths(root=tmp_path / "r1")
        samples = [_text_sample("s1")]
        rows = materialize(samples, [Condition.clean, Condition.scan_low], rp)
        clean_row = next(r for r in rows if r.condition == Condition.clean)
        scan_row = next(r for r in rows if r.condition == Condition.scan_low)
        assert clean_row.image_path == "img/s1.png"
        assert scan_row.image_path == "img/s1_scan_low.jpg"

    def test_degraded_image_file_decodes_to_degrade_output(self, tmp_path):
        import cv2
        import numpy as np

        from ocr_bench.data.degrade import degrade

        rp = RunPaths(root=tmp_path / "r1")
        sample = _text_sample("s1")
        rows = materialize([sample], [Condition.clean, Condition.photo], rp)
        photo_row = next(r for r in rows if r.condition == Condition.photo)
        stored = cv2.imread(str(rp.root / photo_row.image_path), cv2.IMREAD_COLOR)
        expected, _ = degrade(sample.image, "s1", Condition.photo)
        assert np.array_equal(stored, expected)

    def test_photo_sidecar_has_homography_and_transformed_words(self, tmp_path):
        rp = RunPaths(root=tmp_path / "r1")
        samples = [_text_layer_sample("s3")]
        rows = materialize(samples, [Condition.clean, Condition.photo], rp)
        clean_row = next(r for r in rows if r.condition == Condition.clean)
        photo_row = next(r for r in rows if r.condition == Condition.photo)

        clean_gt = GROUND_TRUTH.validate_json(
            (rp.root / clean_row.gt_path).read_text(encoding="utf-8")
        )
        photo_gt = GROUND_TRUTH.validate_json(
            (rp.root / photo_row.gt_path).read_text(encoding="utf-8")
        )
        assert photo_row.gt_path != clean_row.gt_path
        assert photo_gt.homography is not None
        assert len(photo_gt.homography) == 3
        assert photo_gt.gt_words != clean_gt.gt_words

    def test_rows_sorted_by_source_sample_id_condition(self, tmp_path):
        rp = RunPaths(root=tmp_path / "r1")
        samples = [_text_sample("s2"), _text_sample("s1")]
        rows = materialize(samples, [Condition.photo, Condition.clean], rp)
        keys = [(r.source.value, r.sample_id, r.condition.value) for r in rows]
        cond_rank = {"clean": 0, "scan_low": 1, "photo": 2}
        assert keys == sorted(keys, key=lambda k: (k[0], k[1], cond_rank[k[2]]))

    def test_determinism_across_two_run_dirs(self, tmp_path):
        rp1 = RunPaths(root=tmp_path / "run1")
        rp2 = RunPaths(root=tmp_path / "run2")
        samples1 = [_text_sample("s1"), _text_layer_sample("s2")]
        samples2 = [_text_sample("s1"), _text_layer_sample("s2")]
        rows1 = materialize(samples1, [Condition.clean, Condition.scan_low, Condition.photo], rp1)
        rows2 = materialize(samples2, [Condition.clean, Condition.scan_low, Condition.photo], rp2)

        text1 = "\n".join(r.model_dump_json() for r in rows1)
        text2 = "\n".join(r.model_dump_json() for r in rows2)
        assert text1 == text2

        for row in rows1:
            img1 = (rp1.root / row.image_path).read_bytes()
            img2 = (rp2.root / row.image_path).read_bytes()
            assert img1 == img2

    def test_manifest_row_model_valid(self, tmp_path):
        rp = RunPaths(root=tmp_path / "r1")
        samples = [_text_sample("s1")]
        rows = materialize(samples, [Condition.clean], rp)
        for row in rows:
            assert isinstance(row, ManifestRow)


@pytest.fixture
def prepared_config_dir(tmp_path, statements_dir):
    """Copy configs/ into tmp_path, pointing bankstmt.yaml's input_dir at the fixture corpus."""
    dst = tmp_path / "configs"
    shutil.copytree(REPO_ROOT / "configs", dst)
    bankstmt_path = dst / "datasets" / "bankstmt.yaml"
    data = yaml.safe_load(bankstmt_path.read_text(encoding="utf-8"))
    data["input_dir"] = str(statements_dir)
    bankstmt_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return dst


def _fake_row(task, id_, answer, category="Finance", question="Q?"):
    return {
        "Id": id_,
        "Task": task,
        "category": category,
        "question": question,
        "answer": answer,
        "image": Image.new("RGB", (8, 8), color=(50, 60, 70)),
    }


def _all_fake_thaiocrbench_rows():
    rows = []
    for i in range(2):
        rows.append(_fake_row("Full-page OCR", f"fp{i}", "some full page text"))
        rows.append(_fake_row("Text recognition", f"tr{i}", "some text"))
        rows.append(_fake_row("Fine-grained text recognition", f"fg{i}", "some region text"))
        rows.append(_fake_row("Handwritten content extraction", f"hw{i}", "handwritten text"))
        rows.append(_fake_row("Table parsing", f"tb{i}", "<tr><td>1</td></tr>"))
        rows.append(_fake_row("Document parsing", f"dp{i}", "doc parse text"))
        rows.append(_fake_row("Key information extraction", f"kie{i}", '{"total": "100"}'))
        rows.append(_fake_row("Key information mapping", f"kmap{i}", '{"total": "100"}'))
        rows.append(_fake_row("Document classification", f"cl{i}", "Invoice"))
    return rows


@pytest.fixture(autouse=True)
def _no_network_thaiocrbench(monkeypatch):
    import ocr_bench.data.thaiocrbench as tob_module

    monkeypatch.setattr(
        tob_module, "THAIOCRBENCH_ROWS_PROVIDER", lambda cfg: _all_fake_thaiocrbench_rows()
    )


class TestPrepareCli:
    def test_prepare_writes_manifest_and_summary(self, tmp_path, monkeypatch, prepared_config_dir):
        monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path / "data"))
        result = runner.invoke(
            app,
            [
                "prepare",
                "--config",
                str(prepared_config_dir / "run.yaml"),
                "--run-id",
                "r1",
            ],
        )
        assert result.exit_code == 0, result.output

        rp = RunPaths.for_run("r1")
        assert rp.manifest.exists()
        assert rp.config_resolved.exists()
        assert rp.prepare_summary.exists()

        rows = list(read_rows(rp.manifest, ManifestRow))
        n_samples = 9 * 2 + 4  # 9 thaiocrbench tasks x 2 rows + 4 statement pages
        assert len(rows) == n_samples * 3

        for row in rows:
            assert (rp.root / row.image_path).exists()

        summary = json.loads(rp.prepare_summary.read_text(encoding="utf-8"))
        assert summary["run_id"] == "r1"
        assert summary["n_manifest_rows"] == len(rows)

    def test_second_prepare_without_force_exits_2(self, tmp_path, monkeypatch, prepared_config_dir):
        monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path / "data"))
        args = ["prepare", "--config", str(prepared_config_dir / "run.yaml"), "--run-id", "r1"]
        first = runner.invoke(app, args)
        assert first.exit_code == 0, first.output
        second = runner.invoke(app, args)
        assert second.exit_code == 2
        assert "--force" in second.output

    def test_force_allows_overwrite(self, tmp_path, monkeypatch, prepared_config_dir):
        monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path / "data"))
        args = ["prepare", "--config", str(prepared_config_dir / "run.yaml"), "--run-id", "r1"]
        first = runner.invoke(app, args)
        assert first.exit_code == 0, first.output
        second = runner.invoke(app, [*args, "--force"])
        assert second.exit_code == 0, second.output

    def test_datasets_flag_restricts_to_thaiocrbench_without_statements_dir(
        self, tmp_path, monkeypatch, prepared_config_dir
    ):
        monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path / "data"))
        # Point bankstmt at a directory that does not exist, to prove it is never touched.
        bankstmt_path = prepared_config_dir / "datasets" / "bankstmt.yaml"
        data = yaml.safe_load(bankstmt_path.read_text(encoding="utf-8"))
        data["input_dir"] = str(tmp_path / "does_not_exist")
        bankstmt_path.write_text(yaml.safe_dump(data), encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "prepare",
                "--config",
                str(prepared_config_dir / "run.yaml"),
                "--run-id",
                "r1",
                "--datasets",
                "thaiocrbench",
            ],
        )
        assert result.exit_code == 0, result.output
        rp = RunPaths.for_run("r1")
        rows = list(read_rows(rp.manifest, ManifestRow))
        assert all(r.source == Source.thaiocrbench for r in rows)

    def test_unknown_datasets_flag_exits_2(self, tmp_path, monkeypatch, prepared_config_dir):
        monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path / "data"))
        result = runner.invoke(
            app,
            [
                "prepare",
                "--config",
                str(prepared_config_dir / "run.yaml"),
                "--run-id",
                "r1",
                "--datasets",
                "ghost",
            ],
        )
        assert result.exit_code == 2

    def test_missing_statements_dir_exits_2(self, tmp_path, monkeypatch, prepared_config_dir):
        monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path / "data"))
        bankstmt_path = prepared_config_dir / "datasets" / "bankstmt.yaml"
        data = yaml.safe_load(bankstmt_path.read_text(encoding="utf-8"))
        data["input_dir"] = str(tmp_path / "does_not_exist")
        bankstmt_path.write_text(yaml.safe_dump(data), encoding="utf-8")

        result = runner.invoke(
            app,
            ["prepare", "--config", str(prepared_config_dir / "run.yaml"), "--run-id", "r1"],
        )
        assert result.exit_code == 2

    def test_two_run_ids_give_identical_manifest_and_images(
        self, tmp_path, monkeypatch, prepared_config_dir
    ):
        monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path / "data"))
        base_args = ["prepare", "--config", str(prepared_config_dir / "run.yaml")]
        r1 = runner.invoke(app, [*base_args, "--run-id", "run-a"])
        r2 = runner.invoke(app, [*base_args, "--run-id", "run-b"])
        assert r1.exit_code == 0, r1.output
        assert r2.exit_code == 0, r2.output

        rp1 = RunPaths.for_run("run-a")
        rp2 = RunPaths.for_run("run-b")
        rows1 = list(read_rows(rp1.manifest, ManifestRow))
        rows2 = list(read_rows(rp2.manifest, ManifestRow))
        assert [r.model_dump() for r in rows1] == [r.model_dump() for r in rows2]

        for row1, row2 in zip(rows1, rows2, strict=True):
            img1 = (rp1.root / row1.image_path).read_bytes()
            img2 = (rp2.root / row2.image_path).read_bytes()
            assert img1 == img2

    def test_photo_sidecar_present_for_digital_statement_page(
        self, tmp_path, monkeypatch, prepared_config_dir
    ):
        monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path / "data"))
        result = runner.invoke(
            app,
            [
                "prepare",
                "--config",
                str(prepared_config_dir / "run.yaml"),
                "--run-id",
                "r1",
                "--datasets",
                "bankstmt",
            ],
        )
        assert result.exit_code == 0, result.output
        rp = RunPaths.for_run("r1")
        rows = list(read_rows(rp.manifest, ManifestRow))
        digital_photo_row = next(
            r for r in rows if r.doc_type == "digital" and r.condition == Condition.photo
        )
        digital_clean_row = next(
            r
            for r in rows
            if r.sample_id == digital_photo_row.sample_id and r.condition == Condition.clean
        )
        assert digital_photo_row.gt_path != digital_clean_row.gt_path
        photo_gt = json.loads((rp.root / digital_photo_row.gt_path).read_text(encoding="utf-8"))
        assert photo_gt["homography"] is not None
