"""Per-model Runpod serving config (`configs/runpod/<model>.yaml`), design D2/D9."""

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Keys owned by configs/models/<model>.yaml; they must not be duplicated here.
_FORBIDDEN_KEYS = (
    "served_model_name",
    "endpoint_env",
    "message_prefix",
    "prompt",
    "prompts",
    "sampling",
)
# Flags the renderer appends itself (D9); vllm_args must not set them.
_LOGGING_FLAGS = ("max-log-len", "uvicorn-log-level", "mm-processor-cache-gb")
_RESERVED_ARGS = (
    *_LOGGING_FLAGS,
    "disable-log-requests",
    "enable-log-requests",
    "no-enable-log-requests",
    "served-model-name",
)

_DIGEST_RE = re.compile(r"^[\w./-]+@sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.\d+)?$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageConfig(_Strict):
    base: str
    vllm_version: str  # a digest carries no version, so the config states it
    variant: Literal["plain", "teleocr", "paddle"]

    @field_validator("base")
    @classmethod
    def _digest(cls, v: str) -> str:
        if not _DIGEST_RE.match(v):
            raise ValueError(f"image.base must be pinned by sha256 digest, not a tag: {v!r}")
        return v

    @field_validator("vllm_version")
    @classmethod
    def _version(cls, v: str) -> str:
        if not _VERSION_RE.match(v):
            raise ValueError(f"image.vllm_version must look like 0.11.0, got {v!r}")
        return v


class WeightsConfig(_Strict):
    repo: str
    revision: str

    @field_validator("revision")
    @classmethod
    def _commit(cls, v: str) -> str:
        if not _COMMIT_RE.match(v):
            raise ValueError(f"weights.revision must be a 40-hex commit hash, got {v!r}")
        return v


class ScalerConfig(_Strict):
    type: str
    value: int


class EndpointConfig(_Strict):
    kind: Literal["load_balancing", "queue"] = "load_balancing"
    gpu_types: list[str] = Field(min_length=1)
    workers_min: int = Field(ge=0)
    workers_max: int = Field(ge=1)
    idle_timeout_s: int = Field(ge=0)
    scaler: ScalerConfig
    flashboot: bool = True
    data_centers: list[str] = Field(default_factory=list)  # non-empty enforced at deploy time
    cuda_min: str
    container_disk_gb: int = Field(ge=1)
    network_volume: str | None = None  # never on a promotable endpoint (D9); deploy refuses


class ClientConfig(_Strict):
    max_concurrency: int = Field(ge=1)
    request_timeout_s: float = Field(gt=0)
    cold_start_wait_s: float = Field(ge=0)
    max_request_mb: float = Field(gt=0)


class SloConfig(_Strict):
    page_p50_s: float
    page_p95_s: float
    cold_start_p95_s: float


class RunpodModelConfig(_Strict):
    model: str
    image: ImageConfig
    weights: WeightsConfig
    vllm_args: dict[str, Any] = Field(default_factory=dict)
    endpoint: EndpointConfig
    client: ClientConfig
    slo: SloConfig

    @model_validator(mode="before")
    @classmethod
    def _no_model_config_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            bad = [k for k in _FORBIDDEN_KEYS if k in data]
            if bad:
                raise ValueError(
                    f"{bad} belong in configs/models/<model>.yaml, not the runpod config"
                )
        return data

    @field_validator("vllm_args")
    @classmethod
    def _no_reserved_args(cls, v: dict[str, Any]) -> dict[str, Any]:
        bad = sorted(k for k in v if k.lstrip("-") in _RESERVED_ARGS)
        if bad:
            raise ValueError(f"vllm_args must not set {bad}; they are rendered from the policy")
        return v


def load_runpod_config(path: str | Path, models_dir: str | Path | None = None) -> RunpodModelConfig:
    """Load and validate a runpod config.

    `models_dir` defaults to `<path>/../../models` (configs/runpod -> configs/models).
    Raises FileNotFoundError when `configs/models/<model>.yaml` does not exist.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    cfg = RunpodModelConfig.model_validate(raw)
    mdir = Path(models_dir) if models_dir is not None else path.resolve().parent.parent / "models"
    model_file = mdir / f"{cfg.model}.yaml"
    if not model_file.is_file():
        raise FileNotFoundError(f"no model config for {cfg.model!r}: {model_file} is missing")
    return cfg


def config_hash(cfg: RunpodModelConfig) -> str:
    """SHA-256 of the canonical (key-sorted) YAML of the config."""
    data = cfg.model_dump(mode="json")
    if data["endpoint"]["network_volume"] is None:  # optional, added later: keep old hashes
        del data["endpoint"]["network_volume"]
    canonical = yaml.safe_dump(data, sort_keys=True, default_flow_style=False, allow_unicode=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _vllm_major_minor(version: str) -> tuple[int, int]:
    m = _VERSION_RE.match(version)
    assert m is not None  # validated by ImageConfig
    return int(m.group(1)), int(m.group(2))


def logging_flags(vllm_version: str) -> list[str]:
    """The D9 flags that keep request content out of the worker logs."""
    # 0.11 has --disable-log-requests; >= 0.12 replaced it with --no-enable-log-requests.
    legacy = _vllm_major_minor(vllm_version) < (0, 12)
    return [
        "--disable-log-requests" if legacy else "--no-enable-log-requests",
        *["--max-log-len", "0", "--uvicorn-log-level", "warning", "--mm-processor-cache-gb", "0"],
    ]


def render_vllm_argv(cfg: RunpodModelConfig) -> list[str]:
    """`vllm serve` flags from `vllm_args` plus the D9 logging flags (no model path)."""
    argv: list[str] = []
    for key, value in cfg.vllm_args.items():
        if value is None or value is False:
            continue
        flag = f"--{key}"
        if value is True:
            argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv + logging_flags(cfg.image.vllm_version)
