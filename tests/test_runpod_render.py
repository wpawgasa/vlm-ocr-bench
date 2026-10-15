import json
import os
from pathlib import Path

import pytest

from ocr_bench.runpod.config import config_hash, load_runpod_config
from ocr_bench.runpod.render import endpoint_body, template_body
from tests.runpod_helpers import MODELS, pinned_configs

DIGEST = "sha256:" + "c" * 64
SNAPSHOTS = Path(__file__).parent / "fixtures" / "runpod" / "render"


@pytest.mark.parametrize("model", MODELS)
def test_rendered_bodies_match_snapshot(model, tmp_path):
    cfg = load_runpod_config(pinned_configs(tmp_path) / f"{model}.yaml")
    rendered = {
        "template": template_body(cfg, DIGEST),
        "endpoint": endpoint_body(cfg, "tmpl-id", None),
    }
    snap = SNAPSHOTS / f"{model}.json"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n")
    assert rendered == json.loads(snap.read_text())


def test_template_name_carries_config_hash_and_image_is_digest_pinned(tmp_path):
    cfg = load_runpod_config(pinned_configs(tmp_path) / "dotsocr.yaml")
    body = template_body(cfg, DIGEST)
    assert body["name"] == f"ocrbench-dotsocr-{config_hash(cfg)[:8]}"
    assert body["image"].endswith(f"runpod-dotsocr@{DIGEST}")


def test_profiles_set_min_workers(tmp_path):
    cfg = load_runpod_config(pinned_configs(tmp_path) / "dotsocr.yaml")
    assert endpoint_body(cfg, "t", "office")["workers"]["min"] == 1
    assert endpoint_body(cfg, "t", "off")["workers"]["min"] == 0
    assert endpoint_body(cfg, "t", None)["workers"]["min"] == cfg.endpoint.workers_min


def test_template_rejects_a_tag(tmp_path):
    cfg = load_runpod_config(pinned_configs(tmp_path) / "dotsocr.yaml")
    with pytest.raises(ValueError, match="digest"):
        template_body(cfg, "latest")
