"""The human data-processing review (`docs/runpod-data-processing-review.md`), design D9.

The document is Markdown with a YAML front matter holding `date`, `reviewer`, `documents`,
`decision` (`approved` or `rejected`), `conditions` and `next_review`. `deploy --promote`
refuses while the decision is not `approved` or the review is more than 12 months old.
"""

import calendar
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path("docs/runpod-data-processing-review.md")
MAX_AGE_MONTHS = 12


class ReviewError(ValueError):
    """The review document is missing or cannot be parsed."""


@dataclass
class Review:
    path: Path
    reviewed_on: date
    reviewer: str
    decision: str
    documents: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    next_review: date | None = None

    def expires_on(self) -> date:
        year, month = divmod(self.reviewed_on.month - 1 + MAX_AGE_MONTHS, 12)
        year += self.reviewed_on.year
        day = min(self.reviewed_on.day, calendar.monthrange(year, month + 1)[1])
        return date(year, month + 1, day)

    def problem(self, today: date) -> str | None:
        """Why this review does not allow promotion today, or None when it does."""
        if self.decision != "approved":
            return f"{self.path}: decision is {self.decision!r}, not 'approved'"
        if today > self.expires_on():
            return f"{self.path}: review of {self.reviewed_on} is older than 12 months"
        return None


def _date(raw: Any, key: str, path: Path) -> date:
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw))
    except ValueError as e:
        raise ReviewError(f"{path}: {key} must be a YYYY-MM-DD date, got {raw!r}") from e


def load_review(path: Path) -> Review:
    path = Path(path)
    if not path.exists():
        raise ReviewError(f"review document {path} does not exist")
    text = path.read_text(encoding="utf-8")
    parts = text.split("---\n", 2) if text.startswith("---\n") else []
    if len(parts) < 3:
        raise ReviewError(f"{path}: no YAML front matter")
    try:
        front = yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        raise ReviewError(f"{path}: front matter is not valid YAML") from e
    if not isinstance(front, dict):
        raise ReviewError(f"{path}: front matter is not a mapping")
    for key in ("date", "reviewer", "decision"):
        if not front.get(key):
            raise ReviewError(f"{path}: front matter has no {key}")
    nxt = front.get("next_review")
    return Review(
        path=path,
        reviewed_on=_date(front["date"], "date", path),
        reviewer=str(front["reviewer"]),
        decision=str(front["decision"]).lower(),
        documents=list(front.get("documents") or []),
        conditions=list(front.get("conditions") or []),
        next_review=_date(nxt, "next_review", path) if nxt else None,
    )
