"""Request bodies for the Runpod template and endpoint (design D4); no network, no ids.

The v2 create-body for a load-balancing endpoint is unverified, so the endpoint kind sits in
one field (`_kind_field`, read back from API responses by `endpoint_kind`) that can be adjusted
once the API is confirmed; `ocrbench runpod deploy --dry-run` prints the body.
"""

import re
from typing import Any, Literal

from ocr_bench.runpod.config import RunpodModelConfig, config_hash

IMAGE_REPO_PREFIX = "code.loolootech.com/loolootech/runpod-"
Profile = Literal["office", "off"]

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def image_repo(model: str) -> str:
    return f"{IMAGE_REPO_PREFIX}{model}"


def template_name(cfg: RunpodModelConfig) -> str:
    return f"ocrbench-{cfg.model}-{config_hash(cfg)[:8]}"


def template_body(cfg: RunpodModelConfig, digest: str, registry: str) -> dict[str, Any]:
    """Immutable template for one release: the image is pinned by digest, never a tag, and
    pulled from the private registry with the Runpod registry credential id `registry`."""
    if not _SHA256_RE.match(digest):
        raise ValueError(f"image must be pinned by a sha256 digest, not a tag: {digest!r}")
    return {
        "name": template_name(cfg),
        "image": f"{image_repo(cfg.model)}@{digest}",
        "disk": cfg.endpoint.container_disk_gb,
        "ports": ["8000/http"],
        "serverless": True,
        "registry": registry,
    }


# The one place that knows how the API spells the endpoint kind (unverified for v2).
_KIND_KEY = "type"
_KIND_VALUES = {"load_balancing": "load_balancer", "queue": "queue"}


def _kind_field(cfg: RunpodModelConfig) -> dict[str, Any]:
    return {_KIND_KEY: _KIND_VALUES[cfg.endpoint.kind]}


def endpoint_kind(endpoint: dict[str, Any]) -> str | None:
    """The config-side kind (`load_balancing`/`queue`) of an API endpoint; None if unknown."""
    by_value = {v: k for k, v in _KIND_VALUES.items()}
    return by_value.get(str(endpoint.get(_KIND_KEY, "")).lower())


def is_kind_key(key: str) -> bool:
    return key == _KIND_KEY


def endpoint_body(
    cfg: RunpodModelConfig, template_id: str, profile: Profile | None
) -> dict[str, Any]:
    """Endpoint settings from the config; `profile` overrides the minimum worker count."""
    ep = cfg.endpoint
    workers_min = {"office": 1, "off": 0}.get(profile or "", ep.workers_min)
    volume = {"networkVolumeId": ep.network_volume} if ep.network_volume else {}
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
        **volume,
    }
