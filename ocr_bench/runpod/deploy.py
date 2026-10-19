"""`ocrbench runpod deploy`: upsert a model's template and endpoint (design D4, D9).

Templates are immutable releases: one is reused when an existing `ocrbench-<model>-*` template
already has the same image digest and settings, else a new one named `template_name(cfg)` is
created. The endpoint `ocrbench-<model>` is created, or patched with only the settings that
differ. A new template or a new endpoint is a release: it is appended to the committed release
log (no ids) and suspends any client-data promotion until the parity gate passes again. A
scaling-only change patches the endpoint and is not a release.

Every refusal happens before the first write, so a refused deploy changes nothing.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml

from ocr_bench.runpod import releases
from ocr_bench.runpod import state as runpod_state
from ocr_bench.runpod.api import RunpodAPI
from ocr_bench.runpod.config import (
    RunpodModelConfig,
    config_hash,
    logging_flags,
    render_vllm_argv,
)
from ocr_bench.runpod.render import (
    PLACEHOLDER_GPU,
    Profile,
    endpoint_body,
    endpoint_kind,
    gpu_selection,
    image_repo,
    is_kind_key,
    template_body,
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
# Flags that put request or response content into the worker logs.
_LOGGING_ON = ("--enable-log-requests", "--enable-log-outputs")

Action = Literal["created", "updated", "patched", "up_to_date"]


class DeployRefused(ValueError):
    """The config, digest or existing endpoint fails a deploy rule; nothing was changed."""


@dataclass
class DeployResult:
    action: Action
    endpoint_name: str
    endpoint_id: str
    template_id: str
    changed: list[str] = field(default_factory=list)
    release: dict[str, Any] | None = None


def endpoint_name(cfg: RunpodModelConfig) -> str:
    return f"ocrbench-{cfg.model}"


def load_allowed_data_centers(path: Path) -> list[str]:
    """The D9 allow-list (`configs/runpod/data_centers.yaml`); an absent file allows none."""
    if not Path(path).exists():
        return []
    return list(
        (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("data_centers") or []
    )


def pinned_digest(model: str, digest: str) -> str:
    """`sha256:<hex>` from `sha256:<hex>` or `<this model's image repo>@sha256:<hex>`."""
    repo, at, bare = digest.rpartition("@")
    if at and repo != image_repo(model):
        raise DeployRefused(f"digest {digest!r} is for {repo}, not {image_repo(model)}")
    if not _SHA256_RE.match(bare):
        raise DeployRefused(f"image must be pinned by a sha256 digest, not a tag: {digest!r}")
    return bare


def logging_flag_problems(argv: list[str], vllm_version: str) -> list[str]:
    """What is wrong with the rendered argv's D9 logging flags (empty when compliant)."""
    problems = []
    required = logging_flags(vllm_version)
    i = 0
    while i < len(required):
        flag = required[i]
        has_value = i + 1 < len(required) and not required[i + 1].startswith("--")
        if flag not in argv:
            problems.append(f"missing {flag}")
        elif has_value:
            last = len(argv) - 1 - argv[::-1].index(flag)
            value = argv[last + 1] if last + 1 < len(argv) else None
            if value != required[i + 1]:
                problems.append(f"{flag} must be {required[i + 1]}, got {value}")
        i += 2 if has_value else 1
    for flag in _LOGGING_ON:
        if any(a == flag or a.startswith(f"{flag}=") for a in argv):
            problems.append(f"{flag} re-enables request/response logging")
    return problems


def _promotable(cfg: RunpodModelConfig, entry: dict[str, Any]) -> bool:
    """Load-balancing configs may be promoted (D9); a queue endpoint only once it already is."""
    return cfg.endpoint.kind == "load_balancing" or bool(entry.get("client_data"))


def check_config(
    cfg: RunpodModelConfig, digest: str, allowed_data_centers: list[str], entry: dict[str, Any]
) -> str:
    """The local refusals (no network); returns the bare digest."""
    bare = pinned_digest(cfg.model, digest)
    dcs = cfg.endpoint.data_centers
    if not dcs:
        raise DeployRefused(
            f"{cfg.model}: endpoint.data_centers is empty; fill it from "
            "configs/runpod/data_centers.yaml"
        )
    outside = [dc for dc in dcs if dc not in allowed_data_centers]
    if outside:
        raise DeployRefused(
            f"{cfg.model}: data centre(s) {outside} are not in the allow-list "
            "configs/runpod/data_centers.yaml"
        )
    problems = logging_flag_problems(render_vllm_argv(cfg), cfg.image.vllm_version)
    if problems:
        raise DeployRefused(f"{cfg.model}: vllm argv breaks the logging policy: {problems}")
    if cfg.endpoint.network_volume and _promotable(cfg, entry):
        raise DeployRefused(
            f"{cfg.model}: network volume {cfg.endpoint.network_volume!r} is not allowed on a "
            "promotable endpoint"
        )
    return bare


def registry_credential(state: dict[str, Any]) -> str:
    """The Runpod registry credential id the private images are pulled with (task 5.2)."""
    cred = state.get("registry_credential_id")
    if not cred:
        raise DeployRefused(
            "registry_credential_id is not set in .runpod/state.yaml; the worker images are "
            "private, so add the Runpod registry credential id there (task 5.2)"
        )
    return str(cred)


def dry_run_bodies(
    cfg: RunpodModelConfig,
    digest: str,
    allowed_data_centers: list[str],
    profile: Profile | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The template and endpoint bodies a deploy would send, after the local refusals."""
    state = state or {}
    entry = (state.get("endpoints") or {}).get(endpoint_name(cfg)) or {}
    bare = check_config(cfg, digest, allowed_data_centers, entry)
    return {
        "template": template_body(cfg, bare, registry_credential(state)),
        "endpoint": endpoint_body(cfg, "<template-id>", profile, PLACEHOLDER_GPU),
    }


def _check_existing(cfg: RunpodModelConfig, existing: dict[str, Any], entry: dict) -> None:
    name = endpoint_name(cfg)
    kind = endpoint_kind(existing)
    if kind is None:
        raise DeployRefused(f"endpoint {name}: cannot tell its kind from the API response")
    if kind != cfg.endpoint.kind:
        raise DeployRefused(
            f"endpoint {name} is {kind} but the config says {cfg.endpoint.kind}; "
            "refusing to change it"
        )
    mounts = existing.get("mounts") or {}
    volumes = [m.get("volumeId") for m in mounts.get("network") or []] + list(
        existing.get("networkVolumes") or []
    )
    for volume in volumes:
        if volume and volume != cfg.endpoint.network_volume and _promotable(cfg, entry):
            raise DeployRefused(
                f"endpoint {name} has network volume {volume!r}, "
                "not allowed on a promotable endpoint"
            )


def _find_or_create_template(
    api: RunpodAPI, cfg: RunpodModelConfig, body: dict[str, Any], digest: str
) -> tuple[dict[str, Any], bool]:
    templates = api.list_templates()
    prefix = f"ocrbench-{cfg.model}-"
    for t in templates:
        if not str(t.get("name", "")).startswith(prefix) or t.get("image") != body["image"]:
            continue
        if all(t[k] == v for k, v in body.items() if k != "name" and k in t):
            return t, False
    names = {t.get("name") for t in templates}
    if body["name"] in names:  # same config hash, new digest: keep names unique
        body = {**body, "name": f"{body['name']}-{digest.removeprefix('sha256:')[:8]}"}
    return api.create_template(body), True


def _same(have: Any, want: Any) -> bool:
    """`want` is already in place: nested objects compare as subsets (the API adds fields),
    an empty list equals an absent one, and string lists ignore order."""
    if isinstance(want, dict):
        have = have if isinstance(have, dict) else {}
        return all(_same(have.get(k), v) for k, v in want.items())
    if want in (None, []) and have in (None, []):
        return True
    if isinstance(want, list) and isinstance(have, list):
        if all(isinstance(x, str) for x in [*want, *have]):
            return sorted(want) == sorted(have)
    return have == want


def _diff(
    existing: dict[str, Any], desired: dict[str, Any], template: dict[str, Any]
) -> dict[str, Any]:
    """The PATCH body. v2 applies `templateId` once and keeps no link, so the endpoint is on
    this release when its container settings (image, registry, ports, env) match the template's."""
    patch = {
        k: v
        for k, v in desired.items()
        if k not in ("name", "templateId") and not is_kind_key(k) and not _same(existing.get(k), v)
    }
    if any(
        not _same(existing.get(k), template.get(k)) for k in ("image", "registry", "ports", "env")
    ):
        patch["templateId"] = desired["templateId"]
    return patch


def deploy(
    cfg: RunpodModelConfig,
    digest: str,
    api: RunpodAPI,
    *,
    allowed_data_centers: list[str],
    state_path: Path,
    releases_path: Path,
    git_sha: str,
    deployer: str,
    today: date,
    profile: Profile | None = None,
) -> DeployResult:
    name = endpoint_name(cfg)
    state = runpod_state.load_state(state_path)
    entry = dict((state.get("endpoints") or {}).get(name) or {})
    bare = check_config(cfg, digest, allowed_data_centers, entry)
    registry = registry_credential(state)
    if profile == "office" and not entry.get("client_data"):
        raise DeployRefused(
            f"{cfg.model}: the office profile is only for a promoted endpoint; "
            "promote it first (deploy --promote)"
        )
    record = {
        "model": cfg.model,
        "action": "deploy",
        "date": today.isoformat(),
        "git_sha": git_sha,
        "image_digest": bare,
        "config_hash": config_hash(cfg),
        "weights_revision": cfg.weights.revision,
        "vllm_version": cfg.image.vllm_version,
        "endpoint_kind": cfg.endpoint.kind,
        "deployer": deployer,
        "parity": "pending",
    }
    releases.check(record)

    matches = [e for e in api.list_endpoints() if e.get("name") == name]
    if len(matches) > 1:
        raise DeployRefused(f"{len(matches)} endpoints are named {name}; resolve it by hand")
    existing = matches[0] if matches else None
    if existing is not None:
        _check_existing(cfg, existing, entry)
    try:
        gpu = gpu_selection(cfg, api.gpu_catalog())
    except ValueError as e:
        raise DeployRefused(str(e)) from e

    template, _ = _find_or_create_template(api, cfg, template_body(cfg, bare, registry), bare)
    desired = endpoint_body(cfg, template["id"], profile, gpu)
    changed: list[str] = []
    if existing is None:
        endpoint = api.create_endpoint(desired)
        action: Action = "created"
    else:
        patch = _diff(existing, desired, template)
        changed = sorted(patch)
        if patch:
            api.patch_endpoint(existing["id"], patch)
        endpoint = existing
        if not patch:
            action = "up_to_date"
        elif "templateId" in patch:
            action = "updated"
        else:
            action = "patched"

    new_release = action in ("created", "updated")
    entry.update(model=cfg.model, id=endpoint["id"], template_id=template["id"])
    if new_release or "client_data" not in entry:
        entry["client_data"] = False  # a new release waits for parity before client data
    state.setdefault("endpoints", {})[name] = entry
    runpod_state.save_state(state, state_path)
    if new_release:
        releases.append(record, releases_path)
    return DeployResult(
        action=action,
        endpoint_name=name,
        endpoint_id=endpoint["id"],
        template_id=template["id"],
        changed=changed,
        release=record if new_release else None,
    )
