import json
import os
from pathlib import Path

import pytest

from ocr_bench.runpod.config import config_hash, load_runpod_config
from ocr_bench.runpod.render import GpuSelection, endpoint_body, gpu_selection, template_body
from tests.runpod_helpers import MODELS, pinned_configs

DIGEST = "sha256:" + "c" * 64
CRED = "cred-test-id"
SEL = GpuSelection(["AMPERE_24"], [])
SNAPSHOTS = Path(__file__).parent / "fixtures" / "runpod" / "render"


@pytest.mark.parametrize("model", MODELS)
def test_rendered_bodies_match_snapshot(model, tmp_path):
    cfg = load_runpod_config(pinned_configs(tmp_path) / f"{model}.yaml")
    rendered = {
        "template": template_body(cfg, DIGEST, CRED),
        "endpoint": endpoint_body(cfg, "tmpl-id", None, GpuSelection(["AMPERE_24"], ["MIG"])),
    }
    snap = SNAPSHOTS / f"{model}.json"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n")
    assert rendered == json.loads(snap.read_text())


def test_template_name_carries_config_hash_and_image_is_digest_pinned(tmp_path):
    cfg = load_runpod_config(pinned_configs(tmp_path) / "dotsocr.yaml")
    body = template_body(cfg, DIGEST, CRED)
    assert body["name"] == f"ocrbench-dotsocr-{config_hash(cfg)[:8]}"
    assert body["image"].endswith(f"runpod-dotsocr@{DIGEST}")


def test_profiles_set_min_workers(tmp_path):
    cfg = load_runpod_config(pinned_configs(tmp_path) / "dotsocr.yaml")
    assert endpoint_body(cfg, "t", "office", SEL)["workers"]["min"] == 1
    assert endpoint_body(cfg, "t", "off", SEL)["workers"]["min"] == 0
    assert endpoint_body(cfg, "t", None, SEL)["workers"]["min"] == cfg.endpoint.workers_min


def test_template_rejects_a_tag(tmp_path):
    cfg = load_runpod_config(pinned_configs(tmp_path) / "dotsocr.yaml")
    with pytest.raises(ValueError, match="digest"):
        template_body(cfg, "latest", CRED)


def test_template_pulls_from_the_private_registry_with_the_credential(tmp_path):
    cfg = load_runpod_config(pinned_configs(tmp_path) / "dotsocr.yaml")
    body = template_body(cfg, DIGEST, CRED)
    assert body["image"] == f"code.loolootech.com/loolootech/runpod-dotsocr@{DIGEST}"
    assert body["registry"] == CRED


def test_cuda_min_is_the_minimum_cuda_version(tmp_path):
    cfg = load_runpod_config(pinned_configs(tmp_path) / "dotsocr.yaml")
    assert cfg.endpoint.cuda_min == "12.8"
    gpu = endpoint_body(cfg, "t", None, GpuSelection(["AMPERE_24"], []))["gpu"]
    assert gpu["minCudaVersion"] == "12.8"
    assert "allowedCudaVersions" not in gpu  # the API rejects both together


def test_gpu_selection_takes_the_pools_and_excludes_their_other_members(tmp_path):
    cfg = load_runpod_config(pinned_configs(tmp_path) / "dotsocr.yaml")
    catalog = [
        {"id": "NVIDIA L4", "pool": "AMPERE_24"},
        {"id": "NVIDIA RTX A5000", "pool": "AMPERE_24"},
        {"id": "NVIDIA GeForce RTX 3090", "pool": "AMPERE_24"},
        {"id": "NVIDIA New 24GB Thing", "pool": "AMPERE_24"},
        {"id": "NVIDIA A40", "pool": "AMPERE_48"},
    ]
    assert gpu_selection(cfg, catalog) == GpuSelection(["AMPERE_24"], ["NVIDIA New 24GB Thing"])


@pytest.mark.parametrize(
    ("model", "port"), [("dotsocr", 8000), ("teleocr", 8000), ("paddleocr_vl", 8080)]
)
def test_template_tells_the_load_balancer_the_worker_port(model, port, tmp_path):
    """Runpod's load balancer uses PORT from the endpoint env (default 80), not the image's."""
    cfg = load_runpod_config(pinned_configs(tmp_path) / f"{model}.yaml")
    body = template_body(cfg, DIGEST, CRED)
    assert body["ports"] == [f"{port}/http"]
    assert body["env"] == {
        "PORT": str(port),
        "PORT_HEALTH": str(port),
        "HEALTH_CHECK_PATH": "/health",
    }
