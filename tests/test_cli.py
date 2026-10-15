"""Tests for ocr_bench.cli."""

import shutil
from pathlib import Path

import yaml
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


def test_prepare_real_config_missing_statements_dir_exits_2(tmp_path, monkeypatch):
    # Real configs, but with bankstmt's input_dir pointed at a directory that is
    # guaranteed not to exist (the real /data/statements may exist on this host).
    cfg = tmp_path / "configs"
    shutil.copytree(REPO_ROOT / "configs", cfg)
    bank_yaml = cfg / "datasets" / "bankstmt.yaml"
    data = yaml.safe_load(bank_yaml.read_text(encoding="utf-8"))
    data["input_dir"] = str(tmp_path / "no-such-dir")
    bank_yaml.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(
        app,
        ["prepare", "--config", str(cfg / "run.yaml"), "--run-id", "x", "--datasets", "bankstmt"],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


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


# --- client-data guard (runpod-serving 3.4, D9) ---

RUNPOD_URL = "https://abcdefghij12.api.runpod.ai/v1"


def _guard_setup(tmp_path, monkeypatch, source, url, state=None):
    from ocr_bench.jsonl import write_rows_atomic
    from ocr_bench.paths import RunPaths
    from ocr_bench.runpod import state as runpod_state
    from ocr_bench.schemas import ManifestRow

    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOTSOCR_BASE_URL", url)
    monkeypatch.delenv("OCRBENCH_ALLOW_UNPROMOTED", raising=False)
    state_path = tmp_path / "state.yaml"
    if state is not None:
        state_path.write_text(yaml.safe_dump(state), encoding="utf-8")
    monkeypatch.setattr(runpod_state, "DEFAULT_PATH", state_path)
    rp = RunPaths.for_run("g")
    rp.root.mkdir(parents=True)
    row = ManifestRow(
        sample_id="s1",
        source=source,
        task="ocr_fullpage",
        condition="clean",
        image_path="x.png",
        gt_kind="none",
    )
    write_rows_atomic(rp.manifest, [row])


def _endpoint_state(client_data):
    entry = dict(model="dotsocr", id="abcdefghij12", client_data=client_data)
    return dict(endpoints={"ocrbench-dotsocr": entry})


def _stub_connect(monkeypatch):
    """Replace `_connect` so a guard pass-through is visible without a server."""
    from ocr_bench import cli

    called = []

    def fake(cfgs):
        called.append(1)
        raise SystemExit(7)

    monkeypatch.setattr(cli, "_connect", fake)
    return called


def _infer():
    cfg = str(REPO_ROOT / "configs" / "run.yaml")
    return runner.invoke(app, ["infer", "--config", cfg, "--run-id", "g", "--models", "dotsocr"])


def test_infer_refuses_statement_to_unpromoted_runpod_endpoint(tmp_path, monkeypatch):
    _guard_setup(tmp_path, monkeypatch, "bankstmt", RUNPOD_URL, _endpoint_state(False))
    called = _stub_connect(monkeypatch)
    result = _infer()
    assert result.exit_code == 2
    assert "dotsocr" in result.output
    assert "ocrbench-dotsocr" in result.output
    assert not called


def test_infer_refuses_statement_to_runpod_endpoint_missing_from_state(tmp_path, monkeypatch):
    _guard_setup(tmp_path, monkeypatch, "bankstmt", RUNPOD_URL)
    called = _stub_connect(monkeypatch)
    result = _infer()
    assert result.exit_code == 2
    assert "dotsocr" in result.output
    assert not called


def test_infer_guard_passes_promoted_endpoint(tmp_path, monkeypatch):
    _guard_setup(tmp_path, monkeypatch, "bankstmt", RUNPOD_URL, _endpoint_state(True))
    called = _stub_connect(monkeypatch)
    assert _infer().exit_code == 7
    assert called


def test_infer_guard_ignores_self_hosted_url(tmp_path, monkeypatch):
    _guard_setup(tmp_path, monkeypatch, "bankstmt", "http://localhost:8000/v1")
    called = _stub_connect(monkeypatch)
    assert _infer().exit_code == 7
    assert called


def test_infer_guard_ignores_public_pages(tmp_path, monkeypatch):
    _guard_setup(tmp_path, monkeypatch, "thaiocrbench", RUNPOD_URL, _endpoint_state(False))
    called = _stub_connect(monkeypatch)
    assert _infer().exit_code == 7
    assert called


def test_unpromoted_override_does_not_cover_statements(tmp_path, monkeypatch):
    _guard_setup(tmp_path, monkeypatch, "bankstmt", RUNPOD_URL, _endpoint_state(False))
    monkeypatch.setenv("OCRBENCH_ALLOW_UNPROMOTED", "1")
    _stub_connect(monkeypatch)
    assert _infer().exit_code == 2
