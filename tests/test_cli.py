"""Tests for ocr_bench.cli."""

from pathlib import Path

from typer.testing import CliRunner

from ocr_bench.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]

runner = CliRunner()

COMMANDS = ["prepare", "infer", "probe", "score", "calibrate", "bench-latency", "report"]


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in COMMANDS:
        assert cmd in result.output


def test_score_missing_manifest_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["score", "--run-id", "x"])
    assert result.exit_code == 2
    assert "manifest.jsonl" in result.output
    assert "ocrbench prepare" in result.output


def test_prepare_bad_config_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    bad_config = tmp_path / "run.yaml"
    bad_config.write_text("models: [nope]\ndatasets: []\ngpu_hourly_rate: {}\n")
    (tmp_path / "models").mkdir()
    (tmp_path / "datasets").mkdir()
    result = runner.invoke(app, ["prepare", "--config", str(bad_config), "--run-id", "x"])
    assert result.exit_code == 2


def test_prepare_real_config_exits_3_not_implemented(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    result = runner.invoke(
        app,
        ["prepare", "--config", str(REPO_ROOT / "configs" / "run.yaml"), "--run-id", "x"],
    )
    assert result.exit_code == 3
    assert "not implemented" in result.output


def test_infer_missing_manifest_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["infer", "--run-id", "x"])
    assert result.exit_code == 2


def test_probe_missing_manifest_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["probe", "--run-id", "x"])
    assert result.exit_code == 2


def test_calibrate_missing_fields_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["calibrate", "--run-id", "x"])
    assert result.exit_code == 2


def test_bench_latency_missing_manifest_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["bench-latency", "--run-id", "x"])
    assert result.exit_code == 2


def test_report_missing_manifest_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["report", "--run-id", "x"])
    assert result.exit_code == 2
