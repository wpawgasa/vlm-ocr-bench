"""Tests for ocr_bench.config."""

from pathlib import Path

import pytest
import yaml

from ocr_bench.config import ConfigError, load_config
from ocr_bench.schemas import Task

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_TASKS = [
    Task.ocr_fullpage,
    Task.ocr_line,
    Task.handwriting,
    Task.table,
    Task.docparse,
    Task.kie,
    Task.kie_map,
    Task.classify,
    Task.statement,
]


def test_load_real_run_config():
    resolved = load_config(REPO_ROOT / "configs" / "run.yaml")
    assert set(resolved.models) == {"teleocr", "dotsocr"}
    for model_name, model_cfg in resolved.models.items():
        prompt_tasks = set(model_cfg.prompts.keys())
        for task in ALL_TASKS:
            assert task in prompt_tasks, f"{model_name} missing prompt for {task}"

    thaiocrbench = resolved.datasets["thaiocrbench"]
    assert thaiocrbench.kind == "thaiocrbench"
    assert len(thaiocrbench.tasks) == 9


def test_unknown_model_raises_config_error(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "datasets").mkdir()
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(yaml.safe_dump({"models": ["nope"], "datasets": [], "gpu_hourly_rate": {}}))
    with pytest.raises(ConfigError) as exc_info:
        load_config(run_yaml)
    assert "nope" in str(exc_info.value)


def test_unknown_dataset_raises_config_error(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "datasets").mkdir()
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        yaml.safe_dump({"models": [], "datasets": ["ghost"], "gpu_hourly_rate": {}})
    )
    with pytest.raises(ConfigError) as exc_info:
        load_config(run_yaml)
    assert "ghost" in str(exc_info.value)


def test_invalid_model_field_raises_config_error_with_path(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "datasets").mkdir()
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(yaml.safe_dump({"models": ["bad"], "datasets": [], "gpu_hourly_rate": {}}))
    bad_model_path = tmp_path / "models" / "bad.yaml"
    bad_model_path.write_text(
        yaml.safe_dump(
            {
                "endpoint_env": "BAD_URL",
                "served_model_name": "bad",
                "default_parser": "markdown",
                "prompts": {},
                "unexpected_field": True,
            }
        )
    )
    with pytest.raises(ConfigError) as exc_info:
        load_config(run_yaml)
    assert str(bad_model_path) in str(exc_info.value)


def test_to_yaml_roundtrips_through_safe_load():
    resolved = load_config(REPO_ROOT / "configs" / "run.yaml")
    dumped = resolved.to_yaml()
    loaded = yaml.safe_load(dumped)
    assert loaded["run"]["models"] == resolved.run.models
    assert set(loaded["models"].keys()) == set(resolved.models.keys())
    assert set(loaded["datasets"].keys()) == set(resolved.datasets.keys())
