"""Request bodies for the Runpod template and endpoint (design D4); no network, no ids.

Shapes follow the REST v2 reference and the live API (checked 2026-10-19): the kind is
`type: QUEUE|LOAD_BALANCER`, GPUs are chosen by serverless *pool* (`gpu.pools`, minus
`gpu.excludedTypes`), and `scaling`, `workers` and `mounts` are nested objects.
`ocrbench runpod deploy --dry-run` prints the bodies.
"""

import re
from dataclasses import dataclass
from typing import Any, Literal

from ocr_bench.runpod.config import RunpodModelConfig, config_hash

IMAGE_REPO_PREFIX = "code.loolootech.com/loolootech/runpod-"
Profile = Literal["office", "off"]

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def image_repo(model: str) -> str:
    return f"{IMAGE_REPO_PREFIX}{model}"


# The port each image stage listens on (`ENV PORT` in runpod/Dockerfile). Runpod's load
# balancer reads PORT/PORT_HEALTH from the endpoint env (default 80), not from the image.
_STAGE_PORTS = {"plain": 8000, "teleocr": 8000, "paddle": 8080}
HEALTH_PATH = "/health"  # vLLM and the Paddle front: 200 ready, 204 starting


def worker_port(variant: str) -> int:
    return _STAGE_PORTS[variant]


def template_name(cfg: RunpodModelConfig) -> str:
    return f"ocrbench-{cfg.model}-{config_hash(cfg)[:8]}"


def template_body(cfg: RunpodModelConfig, digest: str, registry: str) -> dict[str, Any]:
    """Immutable template for one release: the image is pinned by digest, never a tag, and
    pulled from the private registry with the Runpod registry credential id `registry`."""
    if not _SHA256_RE.match(digest):
        raise ValueError(f"image must be pinned by a sha256 digest, not a tag: {digest!r}")
    port = worker_port(cfg.image.variant)
    return {
        "name": template_name(cfg),
        "image": f"{image_repo(cfg.model)}@{digest}",
        "disk": cfg.endpoint.container_disk_gb,
        "ports": [f"{port}/http"],
        "env": {"PORT": str(port), "PORT_HEALTH": str(port), "HEALTH_CHECK_PATH": HEALTH_PATH},
        "serverless": True,
        "registry": registry,
    }


# The one place that knows how the API spells the endpoint kind.
_KIND_KEY = "type"
_KIND_VALUES = {"load_balancing": "LOAD_BALANCER", "queue": "QUEUE"}
_SCALER_VALUE_KEY = {"REQUEST_COUNT": "requestCount", "QUEUE_DELAY": "queueDelay"}


@dataclass(frozen=True)
class GpuSelection:
    """Serverless GPU pools to draw from, minus the pool members the config does not list."""

    pools: list[str]
    excluded_types: list[str]


# What a dry run shows: the pools come from the live catalog at deploy time.
PLACEHOLDER_GPU = GpuSelection(["<from /v2/catalog/gpus at deploy>"], [])


def gpu_selection(cfg: RunpodModelConfig, catalog: list[dict[str, Any]]) -> GpuSelection:
    """The pools holding `gpu_types`, excluding every other member of those pools.

    Excluding by default means a GPU type Runpod later adds to a pool is never used until
    someone lists it in `gpu_types` (and it passes the parity gate). Raises ValueError for a
    type that is not in the catalog or has no serverless pool.
    """
    wanted = cfg.endpoint.gpu_types
    pool_of = {g["id"]: g.get("pool") for g in catalog}
    missing = [t for t in wanted if t not in pool_of]
    if missing:
        raise ValueError(f"{cfg.model}: GPU types not in the Runpod catalog: {', '.join(missing)}")
    no_pool = [t for t in wanted if not pool_of[t]]
    if no_pool:
        raise ValueError(f"{cfg.model}: GPU types with no serverless pool: {', '.join(no_pool)}")
    pools = sorted({pool_of[t] for t in wanted})
    excluded = sorted(g["id"] for g in catalog if g.get("pool") in pools and g["id"] not in wanted)
    return GpuSelection(pools, excluded)


def _kind_field(cfg: RunpodModelConfig) -> dict[str, Any]:
    return {_KIND_KEY: _KIND_VALUES[cfg.endpoint.kind]}


def endpoint_kind(endpoint: dict[str, Any]) -> str | None:
    """The config-side kind (`load_balancing`/`queue`) of an API endpoint; None if unknown."""
    by_value = {v: k for k, v in _KIND_VALUES.items()}
    return by_value.get(str(endpoint.get(_KIND_KEY, "")).upper())


def is_kind_key(key: str) -> bool:
    return key == _KIND_KEY


def endpoint_body(
    cfg: RunpodModelConfig, template_id: str, profile: Profile | None, gpu: GpuSelection
) -> dict[str, Any]:
    """Endpoint settings from the config; `profile` overrides the minimum worker count."""
    ep = cfg.endpoint
    workers_min = {"office": 1, "off": 0}.get(profile or "", ep.workers_min)
    volume = {"mounts": {"network": [{"volumeId": ep.network_volume}]}} if ep.network_volume else {}
    return {
        "name": f"ocrbench-{cfg.model}",
        "templateId": template_id,
        **_kind_field(cfg),
        "gpu": {
            "pools": gpu.pools,
            "excludedTypes": gpu.excluded_types,
            "count": 1,
            "minCudaVersion": ep.cuda_min,
        },
        "dataCenterIds": ep.data_centers,
        "workers": {"min": workers_min, "max": ep.workers_max, "idleTimeout": ep.idle_timeout_s},
        "scaling": {"type": ep.scaler.type, _SCALER_VALUE_KEY[ep.scaler.type]: ep.scaler.value},
        "flashboot": "FLASHBOOT" if ep.flashboot else "OFF",
        **volume,
    }
