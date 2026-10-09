"""Tests for docparse.eval_files."""

from pathlib import Path

import pytest
import yaml

from docparse.eval_files import DEFAULT_EVAL_FILES, EvalFilesError, load_eval_files

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_YAML = REPO_ROOT / DEFAULT_EVAL_FILES

STEMS = ["bbl-en-01", "kbank-en-01", "kbank-th-06", "ktb-en-03", "ktb-th-01", "ttb-en-01"]
HARNESS_IDS = [
    "BS-bbl-0001",
    "BS-kbank-0001",
    "BS-kbank-0007",
    "BS-ktb-0003",
    "BS-ktb-0004",
    "BS-ttb-0001",
]


def _entry(file="a-01", harness_id="BS-a-0001", bank="a"):
    return {"file": file, "harness_id": harness_id, "bank": bank}


def _write(tmp_path: Path, data: dict, name: str = "eval_files.v1.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_real_eval_files():
    eval_files = load_eval_files(EVAL_YAML)
    assert eval_files.version == 1
    assert eval_files.file_count == 6
    assert eval_files.file_ids == STEMS
    assert eval_files.harness_ids == HARNESS_IDS
    assert eval_files.anchor == []


def test_committed_file_is_ids_only():
    data = yaml.safe_load(EVAL_YAML.read_text(encoding="utf-8"))
    assert set(data) == {"version", "text_layer", "anchor"}
    for entry in data["text_layer"] + data["anchor"]:
        assert set(entry) == {"file", "harness_id", "bank"}


def test_version_filename_mismatch_raises(tmp_path: Path):
    path = _write(tmp_path, {"version": 1, "text_layer": [_entry()]}, "eval_files.v2.yaml")
    with pytest.raises(EvalFilesError) as exc_info:
        load_eval_files(path)
    assert str(path) in str(exc_info.value)


def test_unknown_entry_key_raises(tmp_path: Path):
    entry = {**_entry(), "text": "nope"}
    path = _write(tmp_path, {"version": 1, "text_layer": [entry]})
    with pytest.raises(EvalFilesError) as exc_info:
        load_eval_files(path)
    assert str(path) in str(exc_info.value)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(EvalFilesError):
        load_eval_files(tmp_path / "does-not-exist.yaml")


def test_duplicate_file_in_group_raises(tmp_path: Path):
    path = _write(tmp_path, {"version": 1, "text_layer": [_entry(), _entry()]})
    with pytest.raises(EvalFilesError):
        load_eval_files(path)


def test_bank_harness_id_mismatch_raises(tmp_path: Path):
    path = _write(tmp_path, {"version": 1, "text_layer": [_entry(bank="b")]})
    with pytest.raises(EvalFilesError):
        load_eval_files(path)


def test_file_in_both_groups_counted_once(tmp_path: Path):
    data = {
        "version": 1,
        "text_layer": [_entry()],
        "anchor": [_entry(), _entry("b-01", "BS-a-0002")],
    }
    eval_files = load_eval_files(_write(tmp_path, data))
    assert eval_files.file_ids == ["a-01", "b-01"]
    assert eval_files.file_count == 2
