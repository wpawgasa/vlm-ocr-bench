"""Tests for ocr_bench.config."""

from pathlib import Path

import pytest
import yaml

from ocr_bench.config import ConfigError, load_config
from ocr_bench.schemas import Task

REPO_ROOT = Path(__file__).resolve().parents[1]
FINE_GRAINED_KEY = "ocr_line@Fine-grained text recognition"
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
        for task in ALL_TASKS:
            assert task.value in model_cfg.plans, f"{model_name} missing plan for {task}"
        assert FINE_GRAINED_KEY in model_cfg.plans

    thaiocrbench = resolved.datasets["thaiocrbench"]
    assert thaiocrbench.kind == "thaiocrbench"
    assert len(thaiocrbench.tasks) == 9
    assert thaiocrbench.domain_field == "category"
    expected_codes = {
        "Full-page OCR": "fullpage",
        "Text recognition": "textrec",
        "Fine-grained text recognition": "finegrained",
        "Handwritten content extraction": "handwriting",
        "Table parsing": "table",
        "Document parsing": "docparse",
        "Key information extraction": "kie",
        "Key information mapping": "kiemap",
        "Document classification": "classify",
    }
    for name, spec in thaiocrbench.tasks.items():
        assert spec.code == expected_codes[name]


def test_task_spec_requires_code():
    from pydantic import ValidationError

    from ocr_bench.config import TaskSpec

    with pytest.raises(ValidationError):
        TaskSpec(task=Task.ocr_fullpage, cap=None)


def test_thaiocrbench_config_domain_field_defaults_to_category():
    from ocr_bench.config import ThaiOCRBenchConfig

    cfg = ThaiOCRBenchConfig(
        kind="thaiocrbench",
        hf_repo="typhoon-ai/ThaiOCRBench",
        tasks={"Full-page OCR": {"task": "ocr_fullpage", "cap": None, "code": "fullpage"}},
    )
    assert cfg.domain_field == "category"


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
                "plans": _all_plans(),
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


def _question_plan() -> dict:
    return {"kind": "question", "version": "question-v1", "parser": "markdown"}


def _all_plans() -> dict:
    plans = {t.value: _question_plan() for t in ALL_TASKS}
    plans[FINE_GRAINED_KEY] = _question_plan()
    return plans


def _write_model(tmp_path, name: str, data: dict) -> Path:
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "datasets").mkdir(exist_ok=True)
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(yaml.safe_dump({"models": [name], "datasets": [], "gpu_hourly_rate": {}}))
    (tmp_path / "models" / f"{name}.yaml").write_text(yaml.safe_dump(data, allow_unicode=True))
    return run_yaml


def _model_data(**overrides) -> dict:
    data = {"endpoint_env": "X_BASE_URL", "served_model_name": "x", "plans": _all_plans()}
    data.update(overrides)
    return data


def test_minimal_model_config_loads_with_defaults(tmp_path):
    resolved = load_config(_write_model(tmp_path, "m", _model_data()))
    cfg = resolved.models["m"]
    assert cfg.sampling.temperature == 0.0
    assert cfg.sampling.max_tokens == 4096
    assert cfg.image_resize == "none"
    assert cfg.message_prefix == ""
    assert cfg.system_prompt is None


def test_missing_task_plan_raises_config_error(tmp_path):
    plans = _all_plans()
    del plans["kie_map"]
    with pytest.raises(ConfigError) as exc_info:
        load_config(_write_model(tmp_path, "m", _model_data(plans=plans)))
    assert "kie_map" in str(exc_info.value)


def test_missing_fine_grained_plan_raises_config_error(tmp_path):
    plans = _all_plans()
    del plans[FINE_GRAINED_KEY]
    with pytest.raises(ConfigError) as exc_info:
        load_config(_write_model(tmp_path, "m", _model_data(plans=plans)))
    assert "Fine-grained" in str(exc_info.value)


def test_unknown_plan_task_key_raises_config_error(tmp_path):
    plans = _all_plans()
    plans["not_a_task"] = _question_plan()
    with pytest.raises(ConfigError):
        load_config(_write_model(tmp_path, "m", _model_data(plans=plans)))


def test_plan_for_prefers_subtask_key():
    from ocr_bench.config import ModelConfig, RequestPlan

    plans = _all_plans()
    plans[FINE_GRAINED_KEY] = {
        "kind": "crop_single",
        "version": "fg-v1",
        "parser": "markdown",
        "prompt": "p",
    }
    cfg = ModelConfig(name="m", endpoint_env="E", served_model_name="m", plans=plans)
    fg = cfg.plan_for(Task.ocr_line, "Fine-grained text recognition")
    assert isinstance(fg, RequestPlan)
    assert fg.version == "fg-v1"
    assert cfg.plan_for(Task.ocr_line, "Text recognition").version == "question-v1"
    assert cfg.plan_for(Task.statement, None).version == "question-v1"


def test_request_plan_prompt_required_for_single_kinds():
    from pydantic import ValidationError

    from ocr_bench.config import RequestPlan

    for kind in ("single", "crop_single", "grounding"):
        with pytest.raises(ValidationError):
            RequestPlan(kind=kind, version="v", parser="markdown")
    RequestPlan(kind="question", version="v", parser="markdown")


def test_request_plan_two_stage_iff_kind_two_stage():
    from pydantic import ValidationError

    from ocr_bench.config import RequestPlan, Sampling, TwoStage

    two_stage = TwoStage(
        layout_prompt="\nAnalyze the image layout.",
        layout_sampling=Sampling(),
        block_prompts={"default": "p"},
        block_sampling={"default": Sampling()},
        skip_types=["image"],
    )
    RequestPlan(kind="two_stage", version="v", parser="teleocr_layout", two_stage=two_stage)
    with pytest.raises(ValidationError):
        RequestPlan(kind="two_stage", version="v", parser="teleocr_layout")
    with pytest.raises(ValidationError):
        RequestPlan(kind="question", version="v", parser="markdown", two_stage=two_stage)


def test_two_stage_requires_default_prompt_and_sampling():
    from pydantic import ValidationError

    from ocr_bench.config import Sampling, TwoStage

    base = {
        "layout_prompt": "l",
        "layout_sampling": Sampling(),
        "block_prompts": {"default": "p"},
        "block_sampling": {"default": Sampling()},
        "skip_types": [],
    }
    TwoStage(**base)
    assert TwoStage(**base).layout_size == (1036, 1036)
    with pytest.raises(ValidationError):
        TwoStage(**{**base, "block_prompts": {"text": "p"}})
    with pytest.raises(ValidationError):
        TwoStage(**{**base, "block_sampling": {"text": Sampling()}})


def test_to_prepare_yaml_has_no_models_section():
    resolved = load_config(REPO_ROOT / "configs" / "run.yaml")
    loaded = yaml.safe_load(resolved.to_prepare_yaml())
    assert set(loaded) == {"run", "datasets"}
    assert set(loaded["datasets"]) == set(resolved.datasets)


def test_real_model_configs_follow_policy_table():
    configs = REPO_ROOT / "configs"
    resolved = load_config(configs / "run.yaml")
    from ocr_bench.config import load_model_config

    typhoon = load_model_config(configs / "models" / "typhoon_ocr15.yaml", "typhoon_ocr15")
    tele, dots = resolved.models["teleocr"], resolved.models["dotsocr"]
    fg = "Fine-grained text recognition"

    for task in (Task.ocr_fullpage, Task.docparse, Task.statement):
        assert tele.plan_for(task, None).kind == "two_stage"
        assert dots.plan_for(task, None).kind == "single"
        assert dots.plan_for(task, None).prompt.startswith("Please output the layout information")
        assert typhoon.plan_for(task, None).kind == "single"
    for task in (Task.kie, Task.kie_map, Task.classify):
        for cfg in (tele, dots, typhoon):
            assert cfg.plan_for(task, None).kind == "question"
    assert tele.plan_for(Task.ocr_line, fg).kind == "crop_single"
    assert dots.plan_for(Task.ocr_line, fg).kind == "grounding"
    assert typhoon.plan_for(Task.ocr_line, fg).kind == "crop_single"
    assert dots.plan_for(Task.ocr_line, None).prompt == "Extract the text content from this image."
    assert tele.plan_for(Task.table, None).parser == "otsl"
    assert "OTSL" in tele.plan_for(Task.table, None).prompt
    assert dots.plan_for(Task.table, None).parser == "dots_layout_json"

    ts = tele.plan_for(Task.ocr_fullpage, None).two_stage
    assert ts.layout_size == (1036, 1036)
    assert ts.block_prompts["seal"] == "\nSeal Recognition:"
    assert ts.block_sampling["text"].presence_penalty == 1.0
    assert ts.block_sampling["text"].frequency_penalty == 0.05
    assert ts.block_sampling["table"].frequency_penalty == 0.005
    assert ts.block_sampling["default"].vllm_xargs == {"no_repeat_ngram_size": 100}
    assert ts.layout_sampling.presence_penalty == 0.0
    assert set(ts.skip_types) == {"image", "list", "equation_block"}
    assert tele.system_prompt == "You are a helpful assistant."

    assert dots.message_prefix == "<|img|><|imgpad|><|endofimg|>"
    assert dots.image_resize == "smart_resize" and dots.max_pixels == 4_000_000
    assert dots.sampling.max_tokens == 8192 and dots.sampling.temperature == 0.0
    assert dots.plan_for(Task.ocr_fullpage, None).prompt.endswith("single JSON object.\n")

    assert typhoon.image_resize == "max_side" and typhoon.max_side == 1800
    assert typhoon.sampling.max_tokens == 10000
    assert typhoon.plan_for(Task.table, None).prompt.startswith("Extract all text from the image.")
