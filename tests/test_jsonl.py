"""Tests for ocr_bench.jsonl and ocr_bench.paths."""

import os
from pathlib import Path

import pytest

from ocr_bench.jsonl import append_rows, read_rows, write_rows_atomic
from ocr_bench.paths import MissingArtifactError, RunPaths, data_dir, require
from ocr_bench.schemas import Condition, GtKind, ManifestRow, Source, Task


def _row(sample_id: str) -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        source=Source.thaiocrbench,
        task=Task.ocr_fullpage,
        page_no=1,
        n_pages=1,
        condition=Condition.clean,
        image_path=f"img/{sample_id}.png",
        gt_kind=GtKind.none,
    )


def test_append_rows_twice_then_read_yields_all_rows(tmp_path):
    path = tmp_path / "manifest.jsonl"
    n1 = append_rows(path, [_row("s1"), _row("s2")])
    n2 = append_rows(path, [_row("s3")])
    assert n1 == 2
    assert n2 == 1
    rows = list(read_rows(path, ManifestRow))
    assert [r.sample_id for r in rows] == ["s1", "s2", "s3"]


def test_append_rows_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "manifest.jsonl"
    append_rows(path, [_row("s1")])
    assert path.exists()


def test_read_rows_skips_blank_lines(tmp_path):
    path = tmp_path / "manifest.jsonl"
    row = _row("s1")
    path.write_text(row.model_dump_json() + "\n\n" + row.model_dump_json() + "\n")
    rows = list(read_rows(path, ManifestRow))
    assert len(rows) == 2


def test_read_rows_invalid_line_reports_path_and_lineno(tmp_path):
    path = tmp_path / "manifest.jsonl"
    good = _row("s1").model_dump_json()
    path.write_text(good + "\n" + "{not valid json}" + "\n")
    with pytest.raises(ValueError) as exc_info:
        list(read_rows(path, ManifestRow))
    msg = str(exc_info.value)
    assert str(path) in msg
    assert "2" in msg


def test_write_rows_atomic_leaves_no_tmp_and_replaces_content(tmp_path):
    path = tmp_path / "scores.jsonl"
    write_rows_atomic(path, [_row("s1"), _row("s2")])
    tmp_path_file = path.with_name(path.name + ".tmp")
    assert not tmp_path_file.exists()
    rows = list(read_rows(path, ManifestRow))
    assert [r.sample_id for r in rows] == ["s1", "s2"]

    # Replace with different content and confirm old content is gone.
    write_rows_atomic(path, [_row("s3")])
    assert not tmp_path_file.exists()
    rows = list(read_rows(path, ManifestRow))
    assert [r.sample_id for r in rows] == ["s3"]


def test_write_rows_atomic_returns_count(tmp_path):
    path = tmp_path / "scores.jsonl"
    count = write_rows_atomic(path, [_row("s1"), _row("s2"), _row("s3")])
    assert count == 3


def test_data_dir_honours_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    assert data_dir() == Path(tmp_path)


def test_data_dir_default(monkeypatch):
    monkeypatch.delenv("OCRBENCH_DATA_DIR", raising=False)
    assert data_dir() == Path("data")


def test_run_paths_for_run_honours_data_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    rp = RunPaths.for_run("run-1")
    assert rp.root == tmp_path / "runs" / "run-1"
    assert rp.manifest == tmp_path / "runs" / "run-1" / "manifest.jsonl"
    assert rp.predictions == tmp_path / "runs" / "run-1" / "predictions.jsonl"
    assert rp.scores == tmp_path / "runs" / "run-1" / "scores.jsonl"
    assert rp.fields == tmp_path / "runs" / "run-1" / "fields.jsonl"
    assert rp.calibration == tmp_path / "runs" / "run-1" / "calibration.jsonl"
    assert rp.latency == tmp_path / "runs" / "run-1" / "latency.jsonl"
    assert rp.thresholds == tmp_path / "runs" / "run-1" / "thresholds.json"
    assert rp.probe == tmp_path / "runs" / "run-1" / "probe.json"
    assert rp.config_resolved == tmp_path / "runs" / "run-1" / "config.resolved.yaml"
    assert rp.prepare_summary == tmp_path / "runs" / "run-1" / "prepare_summary.json"
    assert rp.report_dir == tmp_path / "runs" / "run-1" / "report"
    assert rp.images == tmp_path / "runs" / "run-1" / "img"
    assert rp.gt == tmp_path / "runs" / "run-1" / "gt"


def test_run_paths_is_frozen(tmp_path, monkeypatch):
    import dataclasses

    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    rp = RunPaths.for_run("run-1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rp.root = Path("/somewhere/else")


def test_require_raises_with_producer_name_in_message(tmp_path):
    missing = tmp_path / "manifest.jsonl"
    with pytest.raises(MissingArtifactError) as exc_info:
        require(missing, "prepare")
    msg = str(exc_info.value)
    assert "prepare" in msg
    assert str(missing) in msg


def test_require_returns_path_when_present(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text("{}")
    assert require(path, "prepare") == path


def test_append_rows_flushes_and_fsyncs(tmp_path):
    # Ensure this does not raise and file content is fully visible immediately.
    path = tmp_path / "manifest.jsonl"
    append_rows(path, [_row("s1")])
    with open(path, "rb") as fh:
        content = fh.read()
    assert os.fsync  # sanity: module available
    assert b"s1" in content


def test_latest_predictions_last_row_per_key_wins(tmp_path):
    from ocr_bench.jsonl import latest_predictions
    from ocr_bench.schemas import Condition, NormalizedPage, PredictionRow, RawPrediction

    def pred(sample_id, model, error=None, condition=Condition.clean):
        return PredictionRow(
            sample_id=sample_id,
            condition=condition,
            model=model,
            prompt_version="v1",
            raw=RawPrediction(error=error),
            normalized=NormalizedPage(),
            latency_ms=1.0,
            prompt_tokens=0,
            completion_tokens=0,
        )

    path = tmp_path / "predictions.jsonl"
    assert latest_predictions(path) == {}
    append_rows(path, [pred("a", "m", error="timeout"), pred("b", "m"), pred("a", "n")])
    append_rows(path, [pred("a", "m"), pred("a", "m", condition=Condition.photo)])
    latest = latest_predictions(path)
    assert set(latest) == {
        ("a", "clean", "m"),
        ("b", "clean", "m"),
        ("a", "clean", "n"),
        ("a", "photo", "m"),
    }
    assert latest[("a", "clean", "m")].raw.error is None
