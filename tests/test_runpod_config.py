from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ocr_bench.runpod.config import (
    RunpodModelConfig,
    config_hash,
    load_runpod_config,
    render_vllm_argv,
)

DIGEST = "sha256:" + "a" * 64
REV = "b" * 40


def _raw(vllm_version: str = "0.11.0") -> dict:
    return {
        "model": "dotsocr",
        "image": {
            "base": f"vllm/vllm-openai@{DIGEST}",
            "vllm_version": vllm_version,
            "variant": "plain",
        },
        "weights": {"repo": "dots-studio/dots.ocr", "revision": REV},
        "vllm_args": {
            "trust-remote-code": True,
            "dtype": "bfloat16",
            "max-model-len": 16384,
            "limit-mm-per-prompt": '{"image": 1}',
            "seed": 0,
            "chat-template": None,
        },
        "endpoint": {
            "kind": "load_balancing",
            "gpu_types": ["NVIDIA L4"],
            "workers_min": 0,
            "workers_max": 3,
            "idle_timeout_s": 120,
            "scaler": {"type": "REQUEST_COUNT", "value": 4},
            "flashboot": True,
            "data_centers": [],
            "cuda_min": "12.8",
            "container_disk_gb": 30,
        },
        "client": {
            "max_concurrency": 4,
            "request_timeout_s": 120,
            "cold_start_wait_s": 600,
            "max_request_mb": 25,
        },
        "slo": {"page_p50_s": 6, "page_p95_s": 20, "cold_start_p95_s": 90},
    }


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "dotsocr.yaml").write_text("served_model_name: dotsocr\n")
    (tmp_path / "runpod").mkdir()
    return tmp_path / "runpod"


def _write(tree: Path, raw: dict, name: str = "dotsocr.yaml") -> Path:
    p = tree / name
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    return p


def test_valid_config_renders_argv_with_logging_flags_011(tree):
    cfg = load_runpod_config(_write(tree, _raw()))
    assert isinstance(cfg, RunpodModelConfig)
    argv = render_vllm_argv(cfg)
    assert "--trust-remote-code" in argv
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--seed") + 1] == "0"
    assert "--chat-template" not in argv  # null args are dropped
    assert "--disable-log-requests" in argv
    assert "--no-enable-log-requests" not in argv
    assert argv[argv.index("--max-log-len") + 1] == "0"
    assert argv[argv.index("--uvicorn-log-level") + 1] == "warning"
    assert argv[argv.index("--mm-processor-cache-gb") + 1] == "0"


def test_newer_vllm_uses_no_enable_log_requests(tree):
    cfg = load_runpod_config(_write(tree, _raw("0.22.1")))
    argv = render_vllm_argv(cfg)
    assert "--no-enable-log-requests" in argv
    assert "--disable-log-requests" not in argv


def test_unpinned_revision_refused(tree):
    raw = _raw()
    raw["weights"]["revision"] = "main"
    with pytest.raises(ValidationError):
        load_runpod_config(_write(tree, raw))


def test_tag_base_refused(tree):
    raw = _raw()
    raw["image"]["base"] = "vllm/vllm-openai:v0.11.0"
    with pytest.raises(ValidationError):
        load_runpod_config(_write(tree, raw))


def test_missing_model_config_refused(tree):
    raw = _raw()
    raw["model"] = "nope"
    with pytest.raises(FileNotFoundError):
        load_runpod_config(_write(tree, raw, "nope.yaml"))


@pytest.mark.parametrize("key", ["served_model_name", "endpoint_env", "prompts", "message_prefix"])
def test_forbidden_keys_refused(tree, key):
    raw = _raw()
    raw[key] = "x"
    with pytest.raises(ValidationError):
        load_runpod_config(_write(tree, raw))


def test_vllm_args_cannot_override_logging_flags(tree):
    raw = _raw()
    raw["vllm_args"]["max-log-len"] = 100
    with pytest.raises(ValidationError):
        load_runpod_config(_write(tree, raw))


def test_hash_stable_across_key_order(tree):
    a = load_runpod_config(_write(tree, _raw()))
    raw = _raw()
    reordered = {k: raw[k] for k in reversed(list(raw))}
    reordered["vllm_args"] = dict(reversed(list(raw["vllm_args"].items())))
    b = load_runpod_config(_write(tree, reordered, "other.yaml"))
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 64


def test_hash_changes_with_content(tree):
    a = load_runpod_config(_write(tree, _raw()))
    raw = _raw()
    raw["vllm_args"]["seed"] = 1
    b = load_runpod_config(_write(tree, raw, "other.yaml"))
    assert config_hash(a) != config_hash(b)
