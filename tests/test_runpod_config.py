import re
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
from ocr_bench.runpod.review import load_review

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


# --- shipped configs vs the compose services (design D8) ---------------------------------------

REPO = Path(__file__).resolve().parent.parent
RUNPOD_DIR = REPO / "configs" / "runpod"
COMPOSE_SERVICE = {
    "dotsocr": "dotsocr",
    "typhoon_ocr15": "typhoon",
    "teleocr": "teleocr",
    "ovisocr2": "ovisocr2",
    "paddleocr_vl": "paddleocr_vl",
}
# mm-processor-cache-gb is appended by render_vllm_argv (D9), so it is not in vllm_args.
_HOST_FLAGS = ("model", "served-model-name", "gpu-memory-utilization", "mm-processor-cache-gb")


def _shipped(model: str) -> dict:
    return yaml.safe_load((RUNPOD_DIR / f"{model}.yaml").read_text())


def _compose_args(service: str) -> dict:
    compose = yaml.safe_load((REPO / ".devcontainer" / "compose.yaml").read_text())
    out: dict = {}
    for item in compose["services"][service]["command"]:
        flag, sep, value = item.removeprefix("--").partition("=")
        if flag in _HOST_FLAGS:
            continue
        if "${" in value:
            # a substitution like ${VLLM_DTYPE:-bfloat16}: the config pins its default
            value = re.fullmatch(r"\$\{\w+:-([^}]*)\}", value).group(1)
        out[flag] = value if sep else True
    return out


@pytest.mark.parametrize("model", COMPOSE_SERVICE)
def test_configs_match_compose(model):
    raw = _shipped(model)
    assert raw["model"] == model
    cfg_args = {k: v for k, v in raw["vllm_args"].items() if k not in _HOST_FLAGS}
    normalised = {k: (v if v is True else str(v)) for k, v in cfg_args.items() if v is not None}
    assert normalised == _compose_args(COMPOSE_SERVICE[model])


def test_variants_and_paddle_memory():
    assert _shipped("teleocr")["image"]["variant"] == "teleocr"
    assert _shipped("paddleocr_vl")["image"]["variant"] == "paddle"
    assert _shipped("paddleocr_vl")["vllm_args"]["gpu-memory-utilization"] == 0.7
    for m in ("dotsocr", "typhoon_ocr15", "ovisocr2"):
        assert _shipped(m)["image"]["variant"] == "plain"


@pytest.mark.parametrize("model", COMPOSE_SERVICE)
def test_shipped_config_pins_a_weights_commit(model):
    assert re.fullmatch(r"[0-9a-f]{40}", _shipped(model)["weights"]["revision"])
    assert "REQUIRED" not in (RUNPOD_DIR / f"{model}.yaml").read_text()
    load_runpod_config(RUNPOD_DIR / f"{model}.yaml")


def test_revision_placeholder_is_rejected(tmp_path):
    path = tmp_path / "dotsocr.yaml"
    raw = _shipped("dotsocr")
    raw["weights"]["revision"] = "REQUIRED"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError) as exc:
        load_runpod_config(path)
    assert ("weights", "revision") in {e["loc"] for e in exc.value.errors()}


_PINNED_BASE = {
    "0.11.0": "sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c",
    "0.22.1": "sha256:953d3a06d5e64ab582985cd7401289d3abf2a2c14ef2158e9a84313daeec77d7",
}


@pytest.mark.parametrize("model", COMPOSE_SERVICE)
def test_base_image_pinned_to_digest_of_its_vllm_version(model):
    image = _shipped(model)["image"]
    assert image["base"] == "vllm/vllm-openai@" + _PINNED_BASE[image["vllm_version"]]


def _allow_list() -> list[str]:
    return yaml.safe_load((RUNPOD_DIR / "data_centers.yaml").read_text())["data_centers"]


def test_data_centers_allow_list_is_filled_and_points_at_review():
    assert _allow_list()
    assert len(set(_allow_list())) == len(_allow_list())
    assert "runpod-data-processing-review.md" in (RUNPOD_DIR / "data_centers.yaml").read_text()


@pytest.mark.parametrize("model", COMPOSE_SERVICE)
def test_model_data_centers_are_a_non_empty_subset_of_the_allow_list(model):
    dcs = yaml.safe_load((RUNPOD_DIR / f"{model}.yaml").read_text())["endpoint"]["data_centers"]
    assert dcs
    assert set(dcs) <= set(_allow_list())


def test_data_processing_review_is_approved():
    review = load_review(REPO / "docs" / "runpod-data-processing-review.md")
    assert review.problem(review.reviewed_on) is None
    assert review.next_review and review.next_review > review.reviewed_on


@pytest.mark.parametrize("cuda_min", ["12", "twelve", "12.8.1"])
def test_cuda_min_must_be_major_dot_minor(cuda_min):
    raw = _raw()
    raw["endpoint"]["cuda_min"] = cuda_min
    with pytest.raises(ValidationError, match="cuda_min"):
        RunpodModelConfig.model_validate(raw)
