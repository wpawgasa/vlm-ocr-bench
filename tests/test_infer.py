"""Tests for `ocrbench infer` (fake models over a tmp manifest; no network)."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ocr_bench.cli import app
from ocr_bench.jsonl import latest_predictions, read_rows
from ocr_bench.models import registry
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import Condition, GtKind, ManifestRow, PredictionRow, Source, Task
from tests.fakes import FakeOcrModel, make_run

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(REPO_ROOT / "configs" / "run.yaml")
runner = CliRunner()


class SimulatedCrash(BaseException):
    pass


def _rows() -> list[ManifestRow]:
    rows = []
    for i in range(3):
        for condition in (Condition.clean, Condition.photo):
            suffix = "" if condition == Condition.clean else "_photo"
            rows.append(
                ManifestRow(
                    sample_id=f"TOB-fullpage-{i}",
                    source=Source.thaiocrbench,
                    task=Task.ocr_fullpage,
                    subtask="Full-page OCR",
                    condition=condition,
                    image_path=f"img/TOB-fullpage-{i}{suffix}.png",
                    gt_kind=GtKind.text,
                )
            )
    return rows


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    make_run(tmp_path, "r1", _rows())
    return RunPaths.for_run("r1")


@pytest.fixture
def fakes(monkeypatch):
    models: dict[str, FakeOcrModel] = {}

    def factory(cfg, **kwargs):
        return models.setdefault(cfg.name, FakeOcrModel(cfg.name))

    monkeypatch.setattr(registry, "build_model", factory)
    return models


def _infer(*extra):
    return runner.invoke(app, ["infer", "--config", CONFIG, "--run-id", "r1", *extra])


def test_infer_writes_every_row(run_dir, fakes):
    result = _infer("--models", "teleocr,dotsocr")
    assert result.exit_code == 0, result.output
    latest = latest_predictions(run_dir.predictions)
    assert len(latest) == 12
    for name in ("teleocr", "dotsocr"):
        assert sorted(fakes[name].requested) == sorted(r.sample_id for r in _rows())
        assert f"{name}: pages=6 errors=0" in result.output


def test_infer_defaults_to_run_yaml_models(run_dir, fakes):
    result = _infer()
    assert result.exit_code == 0, result.output
    assert set(fakes) == {"teleocr", "dotsocr"}


def test_infer_writes_config_infer_yaml(run_dir, fakes):
    result = _infer("--models", "dotsocr", "--concurrency", "3")
    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load((run_dir.root / "config.infer.yaml").read_text(encoding="utf-8"))
    assert cfg["concurrency"] == 3
    assert set(cfg["models"]) == {"dotsocr"}
    assert cfg["models"]["dotsocr"]["sampling"]["max_tokens"] == 8192
    assert "plans" in cfg["models"]["dotsocr"]
    # a later run for another model adds to the snapshot instead of replacing it
    assert _infer("--models", "teleocr").exit_code == 0
    cfg = yaml.safe_load((run_dir.root / "config.infer.yaml").read_text(encoding="utf-8"))
    assert set(cfg["models"]) == {"dotsocr", "teleocr"}


def test_resume_after_interruption_requests_only_missing_rows(run_dir, monkeypatch):
    crashing = FakeOcrModel(
        "teleocr", behaviour=lambda n, row: SimulatedCrash() if n == 4 else "ok"
    )
    monkeypatch.setattr(registry, "build_model", lambda cfg, **kw: crashing)
    with pytest.raises(BaseException):  # noqa: B017 — the simulated crash kills the run
        _infer("--models", "teleocr", "--concurrency", "1")
    written = list(read_rows(run_dir.predictions, PredictionRow))
    assert len(written) == 3

    resumed = FakeOcrModel("teleocr")
    monkeypatch.setattr(registry, "build_model", lambda cfg, **kw: resumed)
    result = _infer("--models", "teleocr", "--concurrency", "1")
    assert result.exit_code == 0, result.output
    assert len(resumed.requested) == 3
    done = {(r.sample_id, r.condition) for r in written}
    missing = [r for r in _rows() if (r.sample_id, r.condition) not in done]
    assert resumed.requested == [r.sample_id for r in missing]
    assert len(latest_predictions(run_dir.predictions)) == 6


def test_errored_row_is_requested_again(run_dir, monkeypatch):
    first = FakeOcrModel("teleocr", behaviour=lambda n, row: "error" if n == 2 else "ok")
    monkeypatch.setattr(registry, "build_model", lambda cfg, **kw: first)
    result = _infer("--models", "teleocr", "--concurrency", "1")
    assert result.exit_code == 0, result.output  # an error row still counts as a row
    assert "errors=1" in result.output
    errored = first.requested[1]

    second = FakeOcrModel("teleocr")
    monkeypatch.setattr(registry, "build_model", lambda cfg, **kw: second)
    result = _infer("--models", "teleocr")
    assert result.exit_code == 0, result.output
    assert second.requested == [errored]
    latest = latest_predictions(run_dir.predictions)
    assert all(r.raw.error is None for r in latest.values())
    assert "errors=0" in result.output


def test_model_exception_becomes_error_row(run_dir, monkeypatch):
    boom = FakeOcrModel(
        "teleocr", behaviour=lambda n, row: RuntimeError("boom") if n == 1 else "ok"
    )
    monkeypatch.setattr(registry, "build_model", lambda cfg, **kw: boom)
    result = _infer("--models", "teleocr", "--concurrency", "1")
    assert result.exit_code == 0, result.output
    latest = latest_predictions(run_dir.predictions)
    assert len(latest) == 6
    assert sum(1 for r in latest.values() if r.raw.error and "boom" in r.raw.error) == 1


def test_health_failure_exits_2_before_any_page(run_dir, monkeypatch):
    models = {
        "teleocr": FakeOcrModel("teleocr"),
        "dotsocr": FakeOcrModel("dotsocr", healthy=False),
    }
    monkeypatch.setattr(registry, "build_model", lambda cfg, **kw: models[cfg.name])
    result = _infer("--models", "teleocr,dotsocr")
    assert result.exit_code == 2
    assert "dotsocr" in result.output
    assert "http://down:8000/v1" in result.output
    assert not run_dir.predictions.exists()
    assert models["teleocr"].requested == []


def test_unset_endpoint_env_exits_2(run_dir, monkeypatch):
    monkeypatch.delenv("DOTSOCR_BASE_URL", raising=False)
    result = _infer("--models", "dotsocr")
    assert result.exit_code == 2
    assert "DOTSOCR_BASE_URL" in result.output


def test_row_count_mismatch_exits_nonzero(run_dir, monkeypatch):
    bad = FakeOcrModel("teleocr", behaviour=lambda n, row: "wrong_id" if n == 1 else "ok")
    monkeypatch.setattr(registry, "build_model", lambda cfg, **kw: bad)
    result = _infer("--models", "teleocr", "--concurrency", "1")
    assert result.exit_code == 1
    assert "mismatch" in result.output


def test_unknown_model_exits_2(run_dir, fakes):
    result = _infer("--models", "nope")
    assert result.exit_code == 2
    assert "nope" in result.output


def test_extra_model_config_not_in_run_yaml_is_loaded(run_dir, fakes):
    result = _infer("--models", "typhoon_ocr15")
    assert result.exit_code == 0, result.output
    assert len(fakes["typhoon_ocr15"].requested) == 6
