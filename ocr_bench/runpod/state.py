"""The local endpoint state (`.runpod/state.yaml`, gitignored), design D4 and D9.

Layout: `endpoints: {<endpoint name>: {model, id, client_data, approved_by, approved_on}}`.
The endpoint id is the first label of the endpoint's `<id>.api.runpod.ai` host. The file
holds ids, so it never goes to git.
"""

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(".runpod/state.yaml")

HOST_SUFFIX = ".api.runpod.ai"


def load_state(path: Path | None = None) -> dict[str, Any]:
    """The whole state; an absent file is an empty state."""
    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def endpoint_for_host(host: str, path: Path | None = None) -> tuple[str, dict[str, Any]] | None:
    """`(endpoint name, entry)` for a `<id>.api.runpod.ai` host, or None if it is unknown."""
    if not host.endswith(HOST_SUFFIX):
        return None
    endpoint_id = host.removesuffix(HOST_SUFFIX)
    for name, entry in (load_state(path).get("endpoints") or {}).items():
        if entry.get("id") == endpoint_id:
            return name, entry
    return None
