"""The committed release log (`configs/runpod/releases.yaml`), design D8.

One record per deploy or rollback. The log is public, so `append` refuses any record that
contains a Runpod API key or an endpoint host.
"""

import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path("configs/runpod/releases.yaml")

_SECRET = re.compile(r"rpa_[A-Za-z0-9]{20,}|[a-z0-9]{10,}\.api\.runpod\.ai")


class ReleaseRecordError(ValueError):
    """A release record that must not be committed."""


def load_releases(path: Path | None = None) -> list[dict[str, Any]]:
    """All records, oldest first; an absent file is an empty log."""
    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def latest(model: str, path: Path | None = None) -> dict[str, Any] | None:
    """The most recently appended record for `model`, or None."""
    for record in reversed(load_releases(path)):
        if record.get("model") == model:
            return record
    return None


def check(record: dict[str, Any]) -> None:
    """Raise ReleaseRecordError if any string in the record looks like a secret."""
    if _SECRET.search(yaml.safe_dump(record, allow_unicode=True)):
        raise ReleaseRecordError("release record contains an API key or endpoint id")


def update_latest(model: str, fields: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Merge `fields` into the model's most recent record (e.g. a promotion) and save the log."""
    path = Path(path or DEFAULT_PATH)
    records = load_releases(path)
    for record in reversed(records):
        if record.get("model") == model:
            check({**record, **fields})
            record.update(fields)
            path.write_text(
                yaml.safe_dump(records, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            return record
    raise ReleaseRecordError(f"no release record for {model}")


def append(record: dict[str, Any], path: Path | None = None) -> None:
    """Append a record; raise ReleaseRecordError if any string in it looks like a secret."""
    check(record)
    path = Path(path or DEFAULT_PATH)
    records = [*load_releases(path), record]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(records, sort_keys=False, allow_unicode=True), encoding="utf-8")
