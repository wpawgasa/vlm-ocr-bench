"""Field value normalization by type, identical for prediction and ground truth (task 4.2).

The type is inferred from the key first (a KIE key such as "วันที่" or `total_amount`),
then from the value's shape. A value that cannot be parsed as its type is kept as
canonical text and flagged `unparsed_<type>`; it is never dropped.
"""

import datetime as dt
import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from ocr_bench.data.gt_utils import _AMOUNT_RE, _THAI_MONTHS
from ocr_bench.normalize.text import canonical_text
from ocr_bench.schemas import FieldValue

FieldType = Literal["amount", "date", "account", "name", "other"]

# Key patterns, checked in this order ("ชื่อบัญชี" is an account *name*, "ยอดเงินในบัญชี"
# an amount, so amount and name precede account).
_DATE_KEY_RE = re.compile(r"วันที่|date|period", re.IGNORECASE)
_AMOUNT_KEY_RE = re.compile(
    r"ราคา|ยอดรวม|ยอดชำระ|ยอดเงิน|ยอดคงเหลือ|รวมเงิน|รวมทั้งสิ้น|จำนวนเงิน|บาท|"
    r"price|total|amount|balance|debit|credit",
    re.IGNORECASE,
)
_NAME_KEY_RE = re.compile(r"ชื่อ|name", re.IGNORECASE)
_ACCOUNT_KEY_RE = re.compile(r"บัญชี|account", re.IGNORECASE)

_ACCOUNT_SHAPE_RE = re.compile(r"^\d{3}-\d-\d{5}-\d$")
_DIGITS_DASHES_RE = re.compile(r"^[\d\- ]+$")
_NUMERIC_DATE_RE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})$")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_DAY_MONTH_YEAR_RE = re.compile(r"^(\d{1,2})\s*([^\d\s,][^\d,]*?)\s*,?\s*(\d{4}|\d{2})$")
_MONTH_DAY_YEAR_RE = re.compile(r"^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$")

_THAI_MONTH_ABBR = ["มค", "กพ", "มีค", "เมย", "พค", "มิย", "กค", "สค", "กย", "ตค", "พย", "ธค"]
_EN_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_MONTHS: dict[str, int] = {}
for _i in range(12):
    _MONTHS[_THAI_MONTHS[_i]] = _i + 1
    _MONTHS[_THAI_MONTH_ABBR[_i]] = _i + 1
    _MONTHS[_EN_MONTHS[_i]] = _i + 1
    _MONTHS[_EN_MONTHS[_i][:3]] = _i + 1
_MONTHS["sept"] = 9

_CURRENCY_RE = re.compile(r"฿|บาท|บ\.|THB|RS|,", re.IGNORECASE)


def _looks_like_date(value: str) -> bool:
    return _parse_date(value) is not None


def _looks_like_account(value: str) -> bool:
    if _ACCOUNT_SHAPE_RE.match(value):
        return True
    return bool(_DIGITS_DASHES_RE.match(value)) and "-" in value and _digit_count(value) >= 10


def _digit_count(value: str) -> int:
    return sum(ch.isdigit() for ch in value)


def field_type(key: str, value: str | None) -> FieldType:
    """Infer a field's type from its key, falling back to its (canonical) value's shape."""
    if _DATE_KEY_RE.search(key):
        return "date"
    if _AMOUNT_KEY_RE.search(key):
        return "amount"
    if _NAME_KEY_RE.search(key):
        return "name"
    if _ACCOUNT_KEY_RE.search(key):
        return "account"
    if value is None:
        return "other"
    v = canonical_text(value)
    if _looks_like_date(v):
        return "date"
    if _looks_like_account(v):
        return "account"
    if _AMOUNT_RE.match(v):
        return "amount"
    return "other"


def _parse_amount(v: str) -> str | None:
    s = v.strip()
    if s.endswith(".-"):
        s = s[:-2]
    s = _CURRENCY_RE.sub("", s).replace(" ", "")
    if s.endswith("-") and len(s) > 1:
        s = s[:-1]
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    if not d.is_finite():
        return None
    # [IMPLEMENTER DECIDES] canonical form drops trailing zeros, so "1,000.00" == "1000".
    text = format(d.normalize(), "f")
    return "0" if text in ("-0", "0") else text


def _year(y: str) -> int:
    n = int(y)
    if len(y) == 2:
        # [IMPLEMENTER DECIDES] 2-digit years: >= 40 is Buddhist era 25yy (BE 2540 = 1997
        # onwards, which covers the "58"/"67" years common on Thai documents); < 40 is
        # CE 20yy. The handover's suggested cut-off of 60 would read BE 2550-2559 as
        # 2050-2059, so 40 is used instead.
        n = 2500 + n if n >= 40 else 2000 + n
    if n > 2400:
        n -= 543
    return n


def _month(token: str) -> int | None:
    key = token.replace(".", "").replace(" ", "").lower()
    return _MONTHS.get(key)


def _iso(y: int, m: int, d: int) -> str | None:
    try:
        return dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def _parse_date(v: str) -> str | None:
    m = _ISO_DATE_RE.match(v)
    if m:
        return _iso(_year(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _NUMERIC_DATE_RE.match(v)
    if m:
        return _iso(_year(m.group(3)), int(m.group(2)), int(m.group(1)))
    m = _DAY_MONTH_YEAR_RE.match(v)
    if m:
        month = _month(m.group(2))
        if month is not None:
            return _iso(_year(m.group(3)), month, int(m.group(1)))
    m = _MONTH_DAY_YEAR_RE.match(v)
    if m:
        month = _month(m.group(1))
        if month is not None:
            return _iso(_year(m.group(3)), month, int(m.group(2)))
    return None


def _parse_account(v: str) -> str | None:
    digits = "".join(ch for ch in v if ch.isdigit())
    return digits or None


def normalize_field(key: str, raw: str | None) -> FieldValue:
    """Normalize one field value by its inferred type; `raw` is kept verbatim."""
    if raw is None:
        return FieldValue(value=None, raw=None)
    ftype = field_type(key, raw)
    text = canonical_text(raw)
    parsed: str | None
    if ftype == "amount":
        parsed = _parse_amount(text)
    elif ftype == "date":
        parsed = _parse_date(text)
    elif ftype == "account":
        parsed = _parse_account(text)
    else:
        return FieldValue(value=text, raw=raw)
    if parsed is None:
        return FieldValue(value=text, raw=raw, flags=[f"unparsed_{ftype}"])
    return FieldValue(value=parsed, raw=raw)
