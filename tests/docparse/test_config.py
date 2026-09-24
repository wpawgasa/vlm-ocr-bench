"""Tests for docparse.config."""

from pathlib import Path

import pytest
import yaml

from docparse.config import PipelineConfigError, load_pipeline_config

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_YAML = REPO_ROOT / "configs" / "docparse" / "pipeline.yaml"


def test_load_real_pipeline_config():
    config = load_pipeline_config(PIPELINE_YAML)
    assert config.roles.classifier == "qwen3vl"
    assert config.verify.max_vlm_calls == 12
    assert config.line_reader == "paddle_crop"


def test_unknown_top_level_key_raises(tmp_path: Path):
    data = yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8"))
    data["unexpected_top_level_key"] = True
    bad_path = tmp_path / "pipeline.yaml"
    bad_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(PipelineConfigError) as exc_info:
        load_pipeline_config(bad_path)
    assert str(bad_path) in str(exc_info.value)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(PipelineConfigError):
        load_pipeline_config(tmp_path / "does-not-exist.yaml")


def test_invalid_line_reader_raises(tmp_path: Path):
    data = yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8"))
    data["line_reader"] = "tesseract"
    bad_path = tmp_path / "pipeline.yaml"
    bad_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(PipelineConfigError):
        load_pipeline_config(bad_path)
