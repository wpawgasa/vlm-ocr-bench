"""`ocrbench runpod rollback|status` and client-data promotion (design D7, D9).

Rollback re-points the endpoint at the previous release's template and appends a `rollback`
record. Promotion marks an endpoint as allowed to receive client data: it needs a passing parity
verdict on the latest release, a current `approved` review, and a load-balancing endpoint (or a
stated retention). A refusal happens before any write.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ocr_bench.runpod import releases
from ocr_bench.runpod import state as runpod_state
from ocr_bench.runpod.api import RunpodAPI
from ocr_bench.runpod.config import RunpodModelConfig
from ocr_bench.runpod.deploy import endpoint_name
from ocr_bench.runpod.render import image_repo
from ocr_bench.runpod.review import ReviewError, load_review

_RELEASE_ACTIONS = ("deploy", "rollback")
# Fields a rollback record takes over from the release it returns to.
_CARRIED = (
    "model",
    "image_digest",
    "config_hash",
    "weights_revision",
    "vllm_version",
    "endpoint_kind",
)


class ManageRefused(ValueError):
    """A rollback, status or promotion rule failed; nothing was changed."""


@dataclass
class RollbackResult:
    endpoint_name: str
    template_id: str
    image_digest: str
    record: dict[str, Any]


@dataclass
class Status:
    endpoint_name: str
    masked_id: str | None
    release_action: str | None
    release_digest: str | None
    release_date: str | None
    parity: str | None
    promoted: bool
    approver: str | None
    approved_on: str | None
    workers: dict[str, Any] | None
    review_problem: str | None


def mask_id(endpoint_id: str) -> str:
    if len(endpoint_id) <= 6:
        return "…"
    return f"{endpoint_id[:2]}…{endpoint_id[-2:]}"


def _entry(state: dict[str, Any], name: str) -> dict[str, Any]:
    return dict((state.get("endpoints") or {}).get(name) or {})


def _history(model: str, releases_path: Path) -> list[dict[str, Any]]:
    return [
        r
        for r in releases.load_releases(releases_path)
        if r.get("model") == model and r.get("action") in _RELEASE_ACTIONS
    ]


def _remote_endpoint(api: RunpodAPI, name: str) -> dict[str, Any]:
    matches = [e for e in api.list_endpoints() if e.get("name") == name]
    if len(matches) != 1:
        raise ManageRefused(f"{len(matches)} endpoints are named {name}; deploy it first")
    return matches[0]


def rollback(
    cfg: RunpodModelConfig,
    api: RunpodAPI,
    *,
    state_path: Path,
    releases_path: Path,
    git_sha: str,
    deployer: str,
    today: date,
) -> RollbackResult:
    name = endpoint_name(cfg)
    state = runpod_state.load_state(state_path)
    entry = _entry(state, name)
    history = _history(cfg.model, releases_path)
    if not entry or not history:
        raise ManageRefused(f"{cfg.model}: nothing is deployed; run deploy first")
    current = history[-1]
    previous = next(
        (r for r in reversed(history[:-1]) if r["image_digest"] != current["image_digest"]), None
    )
    if previous is None:
        raise ManageRefused(f"{cfg.model}: no previous release to roll back to")
    image = f"{image_repo(cfg.model)}@{previous['image_digest']}"
    prefix = f"ocrbench-{cfg.model}-"
    template = next(
        (
            t
            for t in api.list_templates()
            if str(t.get("name", "")).startswith(prefix) and t.get("image") == image
        ),
        None,
    )
    if template is None:
        raise ManageRefused(
            f"{cfg.model}: the previous release's template ({image}) no longer exists on Runpod"
        )
    endpoint = _remote_endpoint(api, name)
    api.patch_endpoint(endpoint["id"], {"templateId": template["id"]})

    record = {
        **{k: previous[k] for k in _CARRIED if k in previous},
        "action": "rollback",
        "date": today.isoformat(),
        "git_sha": git_sha,
        "deployer": deployer,
        "rolled_back_from": current["image_digest"],
        "parity": "pending",
    }
    releases.check(record)
    entry.update(id=endpoint["id"], template_id=template["id"], client_data=False)
    state.setdefault("endpoints", {})[name] = entry
    runpod_state.save_state(state, state_path)
    releases.append(record, releases_path)
    return RollbackResult(name, template["id"], previous["image_digest"], record)


def _worker_counts(endpoint: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(endpoint.get("workers"), dict):
        return endpoint["workers"]
    counts = {"min": endpoint.get("workersMin"), "max": endpoint.get("workersMax")}
    return counts if any(v is not None for v in counts.values()) else None


def status(
    cfg: RunpodModelConfig,
    api: RunpodAPI,
    *,
    state_path: Path,
    releases_path: Path,
    review_path: Path | None,
    today: date,
) -> Status:
    name = endpoint_name(cfg)
    entry = _entry(runpod_state.load_state(state_path), name)
    history = _history(cfg.model, releases_path)
    latest = history[-1] if history else {}
    promoted = bool(entry.get("client_data"))
    workers = None
    if entry.get("id"):
        remote = [e for e in api.list_endpoints() if e.get("id") == entry["id"]]
        workers = _worker_counts(remote[0]) if remote else None
    problem = None
    if promoted and review_path is not None:
        try:
            problem = load_review(review_path).problem(today)
        except ReviewError as e:
            problem = str(e)
    return Status(
        endpoint_name=name,
        masked_id=mask_id(entry["id"]) if entry.get("id") else None,
        release_action=latest.get("action"),
        release_digest=latest.get("image_digest"),
        release_date=str(latest["date"]) if latest.get("date") else None,
        parity=latest.get("parity"),
        promoted=promoted,
        approver=entry.get("approved_by"),
        approved_on=str(entry["approved_on"]) if entry.get("approved_on") else None,
        workers=workers,
        review_problem=problem,
    )


def promote(
    cfg: RunpodModelConfig,
    api: RunpodAPI,
    *,
    state_path: Path,
    releases_path: Path,
    review_path: Path,
    approver: str,
    today: date,
    retention: str | None = None,
) -> dict[str, Any]:
    """Mark the endpoint as allowed to receive client data; returns the updated release record."""
    name = endpoint_name(cfg)
    state = runpod_state.load_state(state_path)
    entry = _entry(state, name)
    history = _history(cfg.model, releases_path)
    if not entry.get("id") or not history:
        raise ManageRefused(f"{cfg.model}: nothing is deployed; run deploy first")
    _remote_endpoint(api, name)
    latest = history[-1]
    if latest.get("parity") != "pass":
        raise ManageRefused(
            f"{cfg.model}: the latest release's parity is {latest.get('parity')!r}, not 'pass'"
        )
    try:
        problem = load_review(review_path).problem(today)
    except ReviewError as e:
        raise ManageRefused(str(e)) from e
    if problem:
        raise ManageRefused(problem)
    if cfg.endpoint.kind != "load_balancing" and not retention:
        raise ManageRefused(
            f"{cfg.model}: a {cfg.endpoint.kind} endpoint stores job data; "
            "state its retention with --retention to promote it"
        )

    fields: dict[str, Any] = {
        "client_data": True,
        "approved_by": approver,
        "approved_on": today.isoformat(),
    }
    if retention:
        fields["retention"] = retention
    record = releases.update_latest(cfg.model, fields, releases_path)
    entry.update(fields)
    state["endpoints"][name] = entry
    runpod_state.save_state(state, state_path)
    return record
