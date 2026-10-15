"""Request bodies for the Runpod template and endpoint (design D4); no network, no ids.

The v2 create-body for a load-balancing endpoint is unverified, so the endpoint kind sits in
one field (`_kind_field`) that `deploy` can adjust once the API is confirmed.
"""

import re
from typing import Any, Literal

from ocr_bench.runpod.config import RunpodModelConfig, config_hash

IMAGE_REPO_PREFIX = "ghcr.io/wpawgasa/vlm-ocr-bench/runpod-"
Profile = Literal["office", "off"]

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def image_repo(model: str) -> str:
    return f"{IMAGE_REPO_PREFIX}{model}"


def template_name(cfg: RunpodModelConfig) -> str:
    return f"ocrbench-{cfg.model}-{config_hash(cfg)[:8]}"


def template_body(cfg: RunpodModelConfig, digest: str) -> dict[str, Any]:
    """Immutable template for one release: the image is pinned by digest, never a tag."""
    if not _SHA256_RE.match(digest):
        raise ValueError(f"image must be pinned by a sha256 digest, not a tag: {digest!r}")
    return {
        "name": template_name(cfg),
        "image": f"{image_repo(cfg.model)}@{digest}",
        "disk": cfg.endpoint.container_disk_gb,
        "ports": ["8000/http"],
        "serverless": True,
    }


def _kind_field(cfg: RunpodModelConfig) -> dict[str, Any]:
    return {"type": "load_balancer" if cfg.endpoint.kind == "load_balancing" else "queue"}


def endpoint_body(
    cfg: RunpodModelConfig, template_id: str, profile: Profile | None
) -> dict[str, Any]:
    """Endpoint settings from the config; `profile` overrides the minimum worker count."""
    ep = cfg.endpoint
    workers_min = {"office": 1, "off": 0}.get(profile or "", ep.workers_min)
    return {
        "name": f"ocrbench-{cfg.model}",
        "templateId": template_id,
        **_kind_field(cfg),
        "gpuTypeIds": ep.gpu_types,
        "dataCenterIds": ep.data_centers,
        "allowedCudaVersions": [ep.cuda_min],
        "workers": {"min": workers_min, "max": ep.workers_max},
        "idleTimeout": ep.idle_timeout_s,
        "scaling": {"type": ep.scaler.type, "value": ep.scaler.value},
        "flashboot": ep.flashboot,
    }
