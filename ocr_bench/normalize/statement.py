"""Rule-based, model-independent statement mapping (task 5.1).

`map_statement_page` turns one model's full-page layout parse (`NormalizedPage.tables`
plus its text blocks) into a `StatementPage`. It is pure: the same tables and blocks
always give the same statement page, whichever model produced them (the spec's
"model-independent mapping" scenario). Nothing here is bank-specific — columns are
found by Thai/English header keywords, and `BankOverrides` only supplies what the
page's own words cannot reveal.

This module also owns the keyword table, the amount/date cell parsers and the header
label table that Check A (`metrics/statement_gt.py`) reuses on the ground-truth side,
so both sides of a comparison are parsed by exactly one implementation.
"""

import datetime as dt
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from ocr_bench.config import BankOverrides
from ocr_bench.normalize.fields import normalize_field
from ocr_bench.normalize.text import canonical_text
from ocr_bench.schemas import (
    BlockType,
    NormalizedPage,
    StatementFile,
    StatementPage,
    StatementRow,
)

# --- column roles ----------------------------------------------------------------------------

WITHDRAWAL_DEPOSIT = "withdrawal_deposit"

#: Column-header keywords, Thai and English, matched case-insensitively on canonical text.
COLUMN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "date": ("date", "วันที่", "วันทำรายการ"),
    "time": ("time", "เวลา"),
    "description": ("descriptions", "description", "particulars", "รายการ", "รายละเอียด"),
    "withdrawal": ("withdrawal", "debit", "ถอน", "เดบิต"),
    "deposit": ("deposit", "credit", "ฝาก", "เครดิต"),
    "amount": ("amount", "จำนวนเงิน"),
    "balance": ("balance", "outstanding", "คงเหลือ", "ยอดคงเหลือ"),
    "channel": ("channel", "ช่องทาง"),
    "cheque": ("chq", "cheque", "เช็ค"),
}

MONEY_ROLES = frozenset({"withdrawal", "deposit", "amount", WITHDRAWAL_DEPOSIT})
#: A header line/row must name at least two of these groups to be a transaction header.
_HEADER_GROUPS = ("date", "money", "balance")
MIN_HEADER_SCORE = 2


def role_for(text: str) -> str | None:
    """The column role of a header cell, or None.

    A cell naming both withdrawal and deposit (KBank's "Withdrawal / Deposit") is the
    combined role; otherwise the keyword appearing earliest in the cell wins, so a
    bilingual or stacked header such as "Time/ Eff.Date" reads as `time`, not `date`.
    """
    t = canonical_text(text).lower()
    if not t:
        return None
    hits: list[tuple[int, str]] = []
    for role, keywords in COLUMN_KEYWORDS.items():
        positions = [t.find(k) for k in keywords if k in t]
        if positions:
            hits.append((min(positions), role))
    if not hits:
        return None
    hits.sort()
    roles = [role for _, role in hits]
    if "withdrawal" in roles and "deposit" in roles:
        return WITHDRAWAL_DEPOSIT
    return roles[0]


def header_score(roles: Iterable[str | None]) -> int:
    """How many of (date, money, balance) a set of column roles covers."""
    role_set = {r for r in roles if r}
    groups = {
        "date": "date" in role_set,
        "money": bool(role_set & MONEY_ROLES),
        "balance": "balance" in role_set,
    }
    return sum(groups[g] for g in _HEADER_GROUPS)


# --- cell parsers ----------------------------------------------------------------------------

_BLANK_CELLS = {"", "-", "--", "—", "–", ".", "n/a"}
_SIDE_SUFFIX_RE = re.compile(r"(?:^|\s)(cr|dr)\.?$", re.IGNORECASE)
_MINUS = ("-", "−", "–")


def parse_decimal(text: str | None) -> Decimal | None:
    """A signed amount (balances may be negative), or None if the cell is not a number."""
    if text is None:
        return None
    t = canonical_text(text)
    if t.lower() in _BLANK_CELLS:
        return None
    negative = False
    if t.startswith("(") and t.endswith(")"):
        negative, t = True, t[1:-1].strip()
    if t.startswith("+"):
        t = t[1:].strip()
    elif t.startswith(_MINUS):
        negative, t = True, t[1:].strip()
    value = normalize_field("amount", t)
    if value.value is None or value.flags:
        return None
    try:
        # Parsed from the cleaned text rather than the normalized value, so the printed
        # scale survives ("20.00" stays 20.00 in the review queue and in statements.jsonl);
        # Decimal compares numerically, so this never changes an equality check.
        parsed = Decimal(t.replace(",", "").replace(" ", ""))
    except InvalidOperation:
        parsed = Decimal(value.value)
    return -parsed if negative else parsed


def parse_money(text: str | None) -> tuple[Decimal | None, str | None]:
    """`(absolute amount, side)` for a transaction cell.

    The side is `"debit"`/`"credit"` when the cell carries its own sign — a leading
    `+`/`-`, parentheses, or a trailing `CR`/`DR` (TTB's signed single column) — else
    None, and the caller decides (x-position, or Check B's balance delta).
    """
    if text is None:
        return None, None
    t = canonical_text(text)
    if t.lower() in _BLANK_CELLS:
        return None, None
    side: str | None = None
    suffix = _SIDE_SUFFIX_RE.search(t)
    if suffix:
        side = "credit" if suffix.group(1).lower() == "cr" else "debit"
        t = t[: suffix.start()].strip()
    signed = t.startswith(("+", "(")) or t.startswith(_MINUS)
    amount = parse_decimal(t)
    if amount is None:
        return None, None
    if amount < 0:
        return -amount, "debit"
    if signed and side is None:
        side = "credit"
    return amount, side


def parse_date(text: str | None, date_format: str | None = None) -> str | None:
    """An ISO date, using the bank's `date_format` first, then the shared date parser."""
    if text is None:
        return None
    t = canonical_text(text)
    if not t:
        return None
    if date_format:
        try:
            return dt.datetime.strptime(t, date_format).date().isoformat()
        except ValueError:
            pass
    value = normalize_field("date", t)
    if value.value is None or value.flags:
        return None
    return value.value


_OPENING_RE = re.compile(
    r"b\s*/\s*f|brought\s*forward|beginning\s*balance|opening\s*balance|"
    r"balance\s*forward|ยอดยกมา|ยอดคงเหลือยกมา|ยกมา",
    re.IGNORECASE,
)


def is_opening_text(text: str | None) -> bool:
    """True for an opening-balance row label (B/F, Beginning Balance, ยอดยกมา)."""
    return bool(text) and bool(_OPENING_RE.search(canonical_text(text or "")))


# --- header fields ---------------------------------------------------------------------------

#: `(field, label)` pairs; labels are matched on canonical lowercase text.
HEADER_LABELS: tuple[tuple[str, str], ...] = (
    ("account_name", "account name"),
    ("account_name", "customer name"),
    ("account_name", "ชื่อบัญชี"),
    ("account_name", "name"),
    ("account_name", "ชื่อ"),
    ("account_no", "account number"),
    ("account_no", "เลขที่บัญชีเงินฝาก"),
    ("account_no", "account no"),
    ("account_no", "acct no"),
    ("account_no", "เลขที่บัญชี"),
    ("account_no", "เลขบัญชี"),
    ("period", "statement period"),
    ("period", "period"),
    ("period", "รอบรายการบัญชี"),
    ("period", "รอบระหว่างวันที่"),
    ("period", "หว่างวันที่"),  # KBank's text layer splits "รอบระหว่างวันที่" as "รอบร หว่างวันที่"
    ("period", "รอบบัญชี"),
    ("closing_balance", "ending balance"),
    ("closing_balance", "closing balance"),
    ("closing_balance", "ยอดคงเหลือปลายงวด"),
    ("closing_balance", "ยอดยกไป"),
    ("opening_balance", "beginning balance"),
    ("opening_balance", "opening balance"),
    ("opening_balance", "ยอดยกมา"),
    ("total_debit", "total withdrawals"),
    ("total_debit", "รวมถอนเงิน"),
    ("total_debit", "total withdrawal"),
    ("total_debit", "รวมรายการถอน"),
    ("total_debit", "รวมถอน"),
    ("total_credit", "total deposits"),
    ("total_credit", "รวมฝากเงิน"),
    ("total_credit", "total deposit"),
    ("total_credit", "รวมรายการฝาก"),
    ("total_credit", "รวมฝาก"),
)
#: Header labels that name no scored field. They end a value ("หน้าที่/Page" is never an
#: account name) and take a slot when labels and values come as separate columns.
OTHER_HEADER_LABELS: tuple[str, ...] = (
    "หน้าที่",
    "page",
    "สกุลเงิน",
    "currency",
    "เลขที่อ้างอิง",
    "reference code",
    "ref. no",
    "สาขาเจ้าของบัญชี",
    "owner branch",
)
HEADER_FIELDS = ("account_no", "account_name", "period", "opening_balance", "closing_balance")
_LABELS_BY_FIELD: dict[str, list[str]] = {}
for _field, _label in HEADER_LABELS:
    _LABELS_BY_FIELD.setdefault(_field, []).append(_label)

_VALUE_SEPARATORS = " \t:：-–—/|.#*"
_PERIOD_SPLIT_RE = re.compile(r"\s+(?:-|–|—|to|ถึง)\s+|\s+(?:to|ถึง)\s+", re.IGNORECASE)


def _label_match(line: str) -> tuple[str, int] | None:
    """The `(field, end offset)` of the best label on `line`: the longest label wins, and
    the value starts after that field's *last* label, so "ชื่อ/Name X" and
    "เลขที่บัญชี/Account No. X" do not keep their second label in the value."""
    low = canonical_text(line).lower()
    best: tuple[int, int, str] | None = None  # (-len(label), position, field)
    for field, labels in _LABELS_BY_FIELD.items():
        for label in labels:
            pos = low.find(label)
            if pos < 0:
                continue
            candidate = (-len(label), pos, field)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    field = best[2]
    end = max(low.find(label) + len(label) for label in _LABELS_BY_FIELD[field] if label in low)
    return field, end


def _all_labels() -> list[str]:
    labels = [label for _, label in HEADER_LABELS] + list(OTHER_HEADER_LABELS)
    return sorted(set(labels), key=len, reverse=True)


_ALL_LABELS = _all_labels()


def _starts_with_label(text: str) -> bool:
    low = canonical_text(text).lower().lstrip(_VALUE_SEPARATORS)
    return any(low.startswith(label) for label in _ALL_LABELS)


def _is_label_only(text: str) -> bool:
    """A line made only of labels and separators, like "เลขที่บัญชี/Account No."."""
    low = canonical_text(text).lower()
    if not low.strip(_VALUE_SEPARATORS):
        return False
    for label in _ALL_LABELS:
        low = low.replace(label, " ")
    return not low.strip(_VALUE_SEPARATORS)


_MONEY_RE = re.compile(r"\d[\d,]*\.\d{2}(?!\d)")
#: Fields whose shape check is strong enough to pick a value out of an unaligned column.
_STRONG_FIELDS = frozenset(
    {"account_no", "period", "opening_balance", "closing_balance", "total_debit", "total_credit"}
)


def _valid_value(field: str, value: str) -> bool:
    """Whether `value` can be `field`'s value: never another label, and shaped like the field
    (an account number has at least 4 digits, amounts parse, a period has a digit)."""
    value = value.strip(_VALUE_SEPARATORS).strip()
    if not value or _starts_with_label(value):
        return False
    digits = sum(ch.isdigit() for ch in value)
    if field == "account_no":
        return digits >= 4
    if field == "account_name":
        return any(ch.isalpha() for ch in value) and digits <= 0.3 * len(value)
    if field == "period":
        return digits > 0
    return _MONEY_RE.search(value) is not None  # a balance/total has cents ("510 รายการ" has not)


def _single_field(line: str) -> str | None:
    """The field a label-only line names, or None when it names none or several (a
    multi-column label line such as "ชื่อ/Name เลขที่บัญชี/Account No." cannot be paired).
    Labels are claimed longest first, so "name" inside "account name" is not counted twice."""
    low = canonical_text(line).lower()
    field_of = {label: field for field, label in HEADER_LABELS}
    fields: set[str] = set()
    for label in sorted(field_of, key=len, reverse=True):
        if label in low:
            fields.add(field_of[label])
            low = low.replace(label, " ")
    return fields.pop() if len(fields) == 1 else None


def _pair_columns(fields: list[str | None], values: list[str]) -> list[tuple[str, str]]:
    """Pair a run of label-only lines with the value lines after it: by position when the
    counts match (the column layout), else each field with a strong shape check takes the
    first unused value that fits it (a model dropped or merged a value; a name cannot be told
    from other text, so it is left unset)."""
    if len(fields) == len(values):
        return [(f, v) for f, v in zip(fields, values, strict=True) if f is not None]
    pairs: list[tuple[str, str]] = []
    used: set[int] = set()
    for field in fields:
        if field not in _STRONG_FIELDS:
            continue
        for i, value in enumerate(values):
            if i not in used and _valid_value(field, value):
                used.add(i)
                pairs.append((field, value))
                break
    return pairs


def extract_header_fields(lines: Sequence[str]) -> dict[str, str]:
    """Raw `{field: value}` from labelled lines; the first (topmost) match per field wins.

    Three layouts are read:
    - label and value on one line ("Account Number XXX-X-XX446-5");
    - a run of label-only lines followed by the same number of value lines, paired in order
      (KBank's header box, which models emit column by column; a single label line followed
      by its value is the one-row case);
    - several labels on one line with their values missing (redacted statements).
    A value is kept only if it fits the field (`_valid_value`): a following line that is
    itself a label ("สกุลเงิน/Currency THB", "หน้าที่ 1/21") is never taken as a value.
    """
    canonical = [c for c in (canonical_text(line) for line in lines) if c.strip()]
    found: dict[str, str] = {}

    def keep(field: str, value: str) -> None:
        value = value.strip(_VALUE_SEPARATORS).strip()
        if field not in found and _valid_value(field, value):
            found[field] = value

    i = 0
    while i < len(canonical):
        line = canonical[i]
        if _is_label_only(line):
            run = [line]
            j = i + 1
            while j < len(canonical) and _is_label_only(canonical[j]):
                run.append(canonical[j])
                j += 1
            values: list[str] = []
            k = j
            while (
                k < len(canonical)
                and len(values) < len(run)
                and not _starts_with_label(canonical[k])
                and _label_match(canonical[k]) is None  # a line with its own label is not a value
            ):
                values.append(canonical[k])
                k += 1
            fields = [_single_field(label) for label in run]
            for field, value in _pair_columns(fields, values):
                keep(field, value)
            i = k
            continue
        match = _label_match(line)
        if match is not None:
            field, end = match
            value = line[end:]
            nxt = canonical[i + 1] if i + 1 < len(canonical) else ""
            if not _valid_value(field, value) and nxt and not _starts_with_label(nxt):
                value = f"{value} {nxt}"  # "รวมถอนเงิน 510 รายการ" / "418,694.01"
            keep(field, value)
        i += 1
    return found


def _last_amount(text: str) -> Decimal | None:
    """The last money-looking token of a value ("1 ITEMS 3,210.00" -> 3210.00)."""
    for token in reversed(text.split()):
        amount = parse_decimal(token)
        if amount is not None:
            return amount
    return None


def apply_header_fields(
    page: StatementPage, raw: dict[str, str], date_format: str | None = None
) -> None:
    """Type and store the raw header values on `page`."""
    if "account_no" in raw:
        page.account_no = raw["account_no"]
    if "account_name" in raw:
        page.account_name = raw["account_name"]
    if "period" in raw:
        parts = [p for p in _PERIOD_SPLIT_RE.split(raw["period"]) if p.strip()]
        dates = [parse_date(p, date_format) for p in parts]
        dates = [d for d in dates if d]
        if dates:
            page.period_start = dates[0]
        if len(dates) > 1:
            page.period_end = dates[-1]
    for field in ("opening_balance", "closing_balance", "total_debit", "total_credit"):
        if field in raw:
            amount = _last_amount(raw[field])
            if amount is not None:
                setattr(page, field, amount)


# --- table -> rows ---------------------------------------------------------------------------

_MAX_HEADER_ROWS = 3  # KBank stacks the column header over three lines


@dataclass(frozen=True)
class HeaderBand:
    """A table's detected header: one role per column, and where the data starts."""

    roles: list[str | None]
    first_data_row: int
    score: int


def _is_data_row(cells: Sequence[str], date_format: str | None) -> bool:
    return any(parse_date(c, date_format) or parse_decimal(c) is not None for c in cells)


def _merge_cells(rows: Sequence[Sequence[str]]) -> list[str]:
    width = max((len(r) for r in rows), default=0)
    merged = []
    for i in range(width):
        parts = [canonical_text(r[i]) for r in rows if i < len(r) and canonical_text(r[i])]
        merged.append(" ".join(parts))
    return merged


def _header_band(
    table: Sequence[Sequence[str]], start: int, date_format: str | None
) -> HeaderBand | None:
    """Merge up to `_MAX_HEADER_ROWS` leading non-data rows from `start` into one header."""
    end = start + 1
    while (
        end < len(table)
        and end - start < _MAX_HEADER_ROWS
        and not _is_data_row(table[end], date_format)
    ):
        end += 1
    roles = [role_for(cell) for cell in _merge_cells(table[start:end])]
    score = header_score(roles)
    if score < MIN_HEADER_SCORE:
        return None
    return HeaderBand(roles=roles, first_data_row=end, score=score)


def find_header(
    tables: Sequence[Sequence[Sequence[str]]], date_format: str | None = None
) -> tuple[int, HeaderBand] | None:
    """The `(table index, header band)` whose header best matches the column keywords."""
    best: tuple[int, int, int, HeaderBand] | None = None  # (-score, table, start, band)
    for t_index, table in enumerate(tables):
        for start in range(min(_MAX_HEADER_ROWS + 1, len(table))):
            band = _header_band(table, start, date_format)
            if band is None:
                continue
            candidate = (-band.score, t_index, start, band)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        return None
    return best[1], best[3]


@dataclass
class RowValues:
    """The role-keyed cell texts of one candidate row, plus an optional side hint from a
    combined column's x-position (Check A only — a table cell has no geometry)."""

    values: dict[str, str]
    side_hint: str | None = None


def build_row(
    row: RowValues,
    *,
    page_no: int,
    overrides: BankOverrides,
    previous_date: str | None = None,
) -> tuple[StatementRow | None, Decimal | None]:
    """`(row, opening_balance)`: exactly one of the two is set, or neither.

    An opening row ("B/F", "Beginning Balance", ยอดยกมา) contributes its balance as the
    page's opening balance and is not a transaction. A row whose date does not parse, or
    that carries no money at all, is dropped (continuation lines and footers).

    [IMPLEMENTER DECIDES] a row with no date of its own but with both an amount and a
    balance continues the previous row's date: banks print the date once per day and
    leave it blank on the day's further transactions (KBank does, for 53 of the 513 rows
    of the longest real statement), and dropping those rows breaks the balance chain.
    A dateless line without both figures stays a description continuation and is dropped.
    """
    values = row.values
    description = values.get("description") or None
    balance = parse_decimal(values.get("balance"))
    amount, side = parse_money(values.get(WITHDRAWAL_DEPOSIT) or values.get("amount"))
    debit, _ = parse_money(values.get("withdrawal"))
    credit, _ = parse_money(values.get("deposit"))

    if is_opening_text(description) and amount is None and debit is None and credit is None:
        return None, balance

    date = parse_date(values.get("date"), overrides.date_format)
    has_money = amount is not None or debit is not None or credit is not None
    if date is None and previous_date is not None and has_money and balance is not None:
        date = previous_date
    if date is None:
        return None, None

    if amount is not None:
        if side is None and overrides.amount_column_split:
            side = row.side_hint
        if side == "debit":
            debit, amount = amount, None
        elif side == "credit":
            credit, amount = amount, None
    if debit is None and credit is None and amount is None and balance is None:
        return None, None

    return (
        StatementRow(
            date=date,
            description=description,
            debit=debit,
            credit=credit,
            amount=amount,
            balance=balance,
            channel=values.get("channel") or None,
            page_no=page_no,
        ),
        None,
    )


def _row_values(cells: Sequence[str], roles: Sequence[str | None]) -> RowValues:
    values: dict[str, str] = {}
    for index, cell in enumerate(cells):
        if index >= len(roles):
            break
        role = roles[index]
        text = canonical_text(cell)
        if role is None or not text:
            continue
        values[role] = f"{values[role]} {text}" if role in values else text
    return RowValues(values=values)


_HTML_TABLE_RE = re.compile(r"<table\b.*?(?:</table>|$)", re.DOTALL | re.IGNORECASE)


def _text_lines(page: NormalizedPage) -> list[str]:
    """Lines of the page's non-table blocks, in block order.

    Blocks are read, not `NormalizedPage.text` (which embeds table HTML in each model's own
    reading order), so two models with the same tables and text blocks map to identical
    statement pages. Only a page with no text blocks at all (the markdown parsers) falls
    back to its text with the tables cut out.
    """
    lines: list[str] = []
    for block in page.blocks:
        if block.type in (BlockType.table, BlockType.figure):
            continue
        lines.extend(block.text.splitlines())
    has_text_blocks = any(b.type in (BlockType.text, BlockType.header) for b in page.blocks)
    if has_text_blocks or not page.text:
        return lines
    # Markdown parsers (typhoon, ovis) give only table/figure blocks: read the page text
    # with its tables removed, so the header is not silently lost for those models.
    text = _HTML_TABLE_RE.sub("\n", page.text)
    return [line for line in text.splitlines() if not line.lstrip().startswith("|")]


def map_statement_page(
    page: NormalizedPage,
    *,
    bank: str | None,
    page_no: int,
    overrides: BankOverrides | None = None,
) -> StatementPage:
    """Map one model's page parse to a `StatementPage`; never raises on garbage input."""
    ov = overrides or BankOverrides()
    statement = StatementPage(bank=bank, page_no=page_no)
    apply_header_fields(statement, extract_header_fields(_text_lines(page)), ov.date_format)

    found = find_header(page.tables, ov.date_format)
    if found is None:
        return statement
    table_index, band = found
    table = page.tables[table_index]
    previous_date: str | None = None
    for cells in table[band.first_data_row :]:
        row, opening = build_row(
            _row_values(cells, band.roles),
            page_no=page_no,
            overrides=ov,
            previous_date=previous_date,
        )
        if opening is not None and statement.opening_balance is None:
            statement.opening_balance = opening
        if row is not None:
            statement.rows.append(row)
            previous_date = row.date
    return statement


def merge_file(
    pages: Sequence[StatementPage], file_id: str, bank: str | None = None
) -> StatementFile:
    """Order pages by page number and tag every row with its source page."""
    ordered = sorted(pages, key=lambda p: p.page_no)
    merged = [
        p.model_copy(
            update={"rows": [r.model_copy(update={"page_no": p.page_no}) for r in p.rows]},
            deep=True,
        )
        for p in ordered
    ]
    if bank is None and ordered:
        bank = ordered[0].bank
    return StatementFile(file_id=file_id, bank=bank, pages=merged)
