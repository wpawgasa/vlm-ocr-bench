"""Pure helpers for turning raw ThaiOCRBench answers into ground truth.

No I/O, no dataset access — everything here operates on plain Python values
so it can be unit-tested without the network or Hugging Face hub.
"""

import html
import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_SEPARATOR_LINE_RE = re.compile(r"^\s*\|?\s*:?-{3,}")

_CRITICAL_KEY_RE = re.compile(
    # Bare ยอด/รวม are common inside ordinary words (ยอดผัก, รวมมิตร); only compounds count.
    r"ราคา|ยอดรวม|ยอดชำระ|ยอดเงิน|ยอดคงเหลือ|รวมเงิน|รวมทั้งสิ้น|จำนวนเงิน|บาท|วันที่|เลขที่|หมายเลข|รหัส|บัญชี|ชื่อ|"
    r"price|total|amount|date|number|\bno\b|\bid\b|name|account",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r"^\s*[฿$]?\s*\d[\d,]*(\.\d+)?\s*(บาท|บ\.|฿|THB|RS|\.-|-)?\s*$")
_DATE_NUMERIC_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_ID_RE = re.compile(r"\d[\d\- ]{5,}\d")
_THAI_MONTHS = [
    "มกราคม",
    "กุมภาพันธ์",
    "มีนาคม",
    "เมษายน",
    "พฤษภาคม",
    "มิถุนายน",
    "กรกฎาคม",
    "สิงหาคม",
    "กันยายน",
    "ตุลาคม",
    "พฤศจิกายน",
    "ธันวาคม",
]

STATEMENT_CRITICAL_FIELDS = [
    "account_no",
    "opening_balance",
    "closing_balance",
    "period_start",
    "period_end",
]


def _strip_fence(s: str) -> str:
    m = _FENCE_RE.match(s)
    if m:
        return m.group(1)
    return s


def parse_answer_json(s: str) -> Any:
    """Parse a ThaiOCRBench JSON answer, tolerating a code fence and trailing commas."""
    stripped = _strip_fence(s)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    repaired = _TRAILING_COMMA_RE.sub(r"\1", stripped)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse answer as JSON: {s!r}") from exc


def flatten_fields(obj: Any) -> dict[str, str | None]:
    """Flatten a (possibly nested) JSON value into a dotted-key string map."""
    if not isinstance(obj, dict):
        obj = {"value": obj}

    result: dict[str, str | None] = {}

    def _visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                _visit(key, v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                key = f"{prefix}[{i}]"
                _visit(key, v)
        elif value is None:
            result[prefix] = None
        elif isinstance(value, bool):
            result[prefix] = "true" if value else "false"
        elif isinstance(value, str):
            result[prefix] = value
        elif isinstance(value, (int, float)):
            result[prefix] = json.dumps(value, ensure_ascii=False)
        else:
            result[prefix] = str(value)

    _visit("", obj)
    return result


def markdown_table_to_html(md: str) -> str:
    """Convert a markdown pipe table to an HTML `<table>`."""
    lines = [line for line in md.splitlines() if "|" in line]
    rows: list[list[str]] = []
    for line in lines:
        if _SEPARATOR_LINE_RE.match(line):
            continue
        cells = line.split("|")
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]
        rows.append([html.escape(c.strip()) for c in cells])

    if not rows:
        return "<table></table>"

    head, *body = rows
    head_html = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
    body_html = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body)
    return f"<table><thead>{head_html}</thead><tbody>{body_html}</tbody></table>"


def table_answer_to_html(ans: str) -> str:
    """Normalize a table answer (HTML or markdown) to a single HTML `<table>`."""
    stripped = ans.strip()
    if "<table" in stripped:
        return stripped
    if stripped.startswith("<tr") or "<tr" in stripped:
        return f"<table>{stripped}</table>"
    return markdown_table_to_html(ans)


def critical_fields_for(fields: dict[str, str | None]) -> list[str]:
    """Return the keys of `fields` that count as critical (ids, amounts, dates, names)."""
    result = []
    for key, value in fields.items():
        if _CRITICAL_KEY_RE.search(key):
            result.append(key)
            continue
        if value is None:
            continue
        if _AMOUNT_RE.search(value):
            result.append(key)
            continue
        if _DATE_NUMERIC_RE.search(value) or any(month in value for month in _THAI_MONTHS):
            result.append(key)
            continue
        if _ID_RE.search(value):
            result.append(key)
            continue
    return result
