"""Check A: statement ground truth parsed from a PDF text layer's positioned words,
and the row/field scoring built on it (task 5.2).

The parser is bank-agnostic. It groups `gt_words` into lines, finds the column-header
band by the same Thai/English keywords the model-side mapper uses
(`normalize.statement.COLUMN_KEYWORDS`), takes each column's x-span from its header
words, and assigns every token below the header to the column it overlaps most.
`BankOverrides` only supplies what the words cannot reveal: the date format, a signed
single amount column (TTB), and whether a combined "Withdrawal / Deposit" column's side
may be read from the x-position (KBank).
"""

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

from ocr_bench.config import BankOverrides
from ocr_bench.metrics.kie_f1 import FieldScore, score_fields, worst_case_fields
from ocr_bench.normalize.fields import normalize_field
from ocr_bench.normalize.statement import (
    WITHDRAWAL_DEPOSIT,
    RowValues,
    apply_header_fields,
    build_row,
    extract_header_fields,
    header_score,
    parse_decimal,
    role_for,
)
from ocr_bench.normalize.text import canonical_text
from ocr_bench.schemas import StatementPage, StatementRow, TextLayerGT, Word

#: Header fields scored on statements (the manifest's `critical_fields` are a subset).
SCORED_FIELDS = (
    "account_no",
    "account_name",
    "period_start",
    "period_end",
    "opening_balance",
    "closing_balance",
)

_MAX_DATE_TOKENS = 3  # "9 Mar 24"
_MAX_BAND_EXTENSION = 2  # header lines added above and below the seed line


# --- lines -------------------------------------------------------------------------------------


@dataclass
class Line:
    top: float
    bottom: float
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in sorted(self.words, key=lambda w: w.bbox[0]))

    @property
    def height(self) -> float:
        return self.bottom - self.top


def group_lines(words: list[Word], overlap: float = 0.4) -> list[Line]:
    """Group words into text lines by vertical overlap (top-to-bottom, then left-to-right)."""
    lines: list[Line] = []
    for word in sorted(words, key=lambda w: (w.bbox[1], w.bbox[0])):
        _, top, _, bottom = word.bbox
        current = lines[-1] if lines else None
        if current is not None:
            shared = min(current.bottom, bottom) - max(current.top, top)
            smallest = min(current.height, bottom - top) or 1.0
            if shared > overlap * smallest:
                current.words.append(word)
                current.top = min(current.top, top)
                current.bottom = max(current.bottom, bottom)
                continue
        lines.append(Line(top=top, bottom=bottom, words=[word]))
    for line in lines:
        line.words.sort(key=lambda w: w.bbox[0])
    return lines


def _median_height(lines: list[Line]) -> float:
    heights = sorted(line.height for line in lines)
    if not heights:
        return 1.0
    return heights[len(heights) // 2] or 1.0


# --- columns -----------------------------------------------------------------------------------


@dataclass
class Column:
    """One detected column: its header words' span, its role, and the span it owns."""

    x0: float
    x1: float
    text: str
    role: str | None = None
    left: float = float("-inf")
    right: float = float("inf")
    split_x: float | None = None
    #: How a figure is placed against `split_x`: by its left edge (the header's separator)
    #: or by its right edge (the midpoint between the page's two right-aligned groups).
    split_edge: str = "left"


def _line_roles(line: Line) -> list[str | None]:
    return [role_for(w.text) for w in line.words]


def _find_seed(lines: list[Line]) -> int | None:
    """The line that best looks like a column header: most of (date, money, balance)
    covered, then the most distinct roles, then topmost."""
    best: tuple[int, int, int] | None = None  # (-score, -n_roles, index)
    for index, line in enumerate(lines):
        roles = _line_roles(line)
        score = header_score(roles)
        if score < 2:
            continue
        candidate = (-score, -len({r for r in roles if r}), index)
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[2]


def _is_header_continuation(line: Line, date_format: str | None) -> bool:
    """A stacked/bilingual header line: it names columns and carries no data."""
    if not any(_line_roles(line)):
        return False
    from ocr_bench.normalize.statement import parse_date

    return not any(
        parse_date(w.text, date_format) or parse_decimal(w.text) is not None for w in line.words
    )


def header_band(lines: list[Line], date_format: str | None) -> tuple[int, int] | None:
    """The `[first, last]` line indices of the column header, or None if there is none."""
    seed = _find_seed(lines)
    if seed is None:
        return None
    gap_limit = 1.5 * _median_height(lines)
    first = last = seed
    for _ in range(_MAX_BAND_EXTENSION):
        above = first - 1
        if above < 0 or lines[first].top - lines[above].bottom > gap_limit:
            break
        if not _is_header_continuation(lines[above], date_format):
            break
        first = above
    for _ in range(_MAX_BAND_EXTENSION):
        below = last + 1
        if below >= len(lines) or lines[below].top - lines[last].bottom > gap_limit:
            break
        if not _is_header_continuation(lines[below], date_format):
            break
        last = below
    return first, last


def build_columns(lines: list[Line], band: tuple[int, int]) -> list[Column]:
    """Cluster the header band's words into columns, then give each the span it owns.

    Words are one column when their x-spans overlap or sit within one line height of each
    other ("Withdrawal / Deposit", or a Thai label stacked over its English translation).
    Each column's span is then extended to the midpoint between neighbouring columns, so
    right-aligned figures and left-aligned text that overhang their header still land in it.
    """
    first, last = band
    band_lines = lines[first : last + 1]
    words = [(index, w) for index, line in enumerate(band_lines) for w in line.words]
    if not words:
        return []
    gap_limit = _median_height(band_lines)

    clusters: list[list[tuple[int, Word]]] = []
    for item in sorted(words, key=lambda iw: iw[1].bbox[0]):
        if clusters and item[1].bbox[0] <= max(w.bbox[2] for _, w in clusters[-1]) + gap_limit:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    columns: list[Column] = []
    for cluster in clusters:
        ordered = sorted(cluster, key=lambda iw: (iw[0], iw[1].bbox[0]))
        text = " ".join(w.text for _, w in ordered)
        x0 = min(w.bbox[0] for _, w in cluster)
        x1 = max(w.bbox[2] for _, w in cluster)
        column = Column(x0=x0, x1=x1, text=text, role=role_for(text))
        if column.role == WITHDRAWAL_DEPOSIT:
            # [IMPLEMENTER DECIDES] a combined column splits at its "/" separator when it
            # has one ("Withdrawal / Deposit"), else at the middle of its header span, and
            # a figure is placed by its *left* edge: on the real pages the header label can
            # sit anywhere over its sub-column, but a deposit always starts right of the
            # separator. `resolve_amount_split` refines this from the data when it can.
            slash = [w for _, w in cluster if w.text.strip() == "/"]
            column.split_x = (slash[0].bbox[0] + slash[0].bbox[2]) / 2 if slash else (x0 + x1) / 2
        columns.append(column)

    for index, column in enumerate(columns):
        if index:
            column.left = (columns[index - 1].x1 + column.x0) / 2
        if index + 1 < len(columns):
            column.right = (column.x1 + columns[index + 1].x0) / 2
    return columns


def resolve_amount_split(lines: list[Line], band: tuple[int, int], columns: list[Column]) -> None:
    """Refine a combined amount column's split point from the page's own figures.

    Withdrawals and deposits are each right-aligned, at two different x positions, so the
    figures below the header form two tight groups of right edges with a wide gap between
    them. Splitting at the middle of that gap is far more reliable than the header's
    separator, which can sit inside the withdrawal group's span (it does on the Thai
    KBank header). With only one group the page cannot tell the two apart, and the
    header's separator decides instead.
    """
    index = next((i for i, c in enumerate(columns) if c.role == WITHDRAWAL_DEPOSIT), None)
    if index is None:
        return
    edges = sorted(
        word.bbox[2]
        for line in lines[band[1] + 1 :]
        for word in line.words
        if _assign(word, columns) == index and parse_decimal(word.text) is not None
    )
    if len(edges) < 2:
        return
    gap, low, high = max((b - a, a, b) for a, b in zip(edges, edges[1:], strict=False))
    if gap >= _median_height(lines):
        columns[index].split_x = (low + high) / 2
        columns[index].split_edge = "right"


def _assign(word: Word, columns: list[Column]) -> int | None:
    """The index of the column `word` overlaps most (spans tile the x axis)."""
    x0, _, x1, _ = word.bbox
    best_index, best_overlap = None, 0.0
    for index, column in enumerate(columns):
        overlap = min(x1, column.right) - max(x0, column.left)
        if overlap > best_overlap:
            best_index, best_overlap = index, overlap
    return best_index


# --- rows ---------------------------------------------------------------------------------------


def _cell_money(tokens: list[str]) -> str | None:
    """The money text of a cell: the whole cell, else its last number-looking token
    (channel words often overhang into the balance column)."""
    joined = " ".join(tokens)
    if parse_decimal(joined) is not None:
        return joined
    for token in reversed(tokens):
        if parse_decimal(token) is not None:
            return token
    return joined or None


def _split_date(tokens: list[str], date_format: str | None) -> tuple[str | None, list[str]]:
    """The longest leading token run that parses as a date, and the leftover tokens
    ("01/03/24 B/F" -> the date plus the B/F label that overhangs from the next column)."""
    from ocr_bench.normalize.statement import parse_date

    for count in range(min(_MAX_DATE_TOKENS, len(tokens)), 0, -1):
        candidate = " ".join(tokens[:count])
        if parse_date(candidate, date_format):
            return candidate, tokens[count:]
    return None, tokens


def _row_values(line: Line, columns: list[Column], overrides: BankOverrides) -> RowValues:
    cells: dict[int, list[str]] = {}
    for word in line.words:
        index = _assign(word, columns)
        if index is not None:
            cells.setdefault(index, []).append(word.text)

    values: dict[str, str] = {}
    leftover: list[str] = []
    side_hint: str | None = None
    for index, tokens in sorted(cells.items()):
        column = columns[index]
        if column.role is None:
            continue
        if column.role == "date":
            date_text, extra = _split_date(tokens, overrides.date_format)
            leftover.extend(extra)
            if date_text:
                values["date"] = date_text
            continue
        if column.role in ("withdrawal", "deposit", "amount", WITHDRAWAL_DEPOSIT, "balance"):
            text = _cell_money(tokens)
            if column.role == WITHDRAWAL_DEPOSIT and column.split_x is not None:
                words = [w for w in line.words if _assign(w, columns) == index]
                numeric = [w for w in words if parse_decimal(w.text) is not None] or words
                if numeric:
                    edge = (
                        max(w.bbox[2] for w in numeric)
                        if column.split_edge == "right"
                        else min(w.bbox[0] for w in numeric)
                    )
                    side_hint = "debit" if edge < column.split_x else "credit"
        else:
            text = " ".join(tokens)
        if text:
            values[column.role] = f"{values[column.role]} {text}" if column.role in values else text

    if leftover:
        description = " ".join(leftover)
        values["description"] = (
            f"{description} {values['description']}" if "description" in values else description
        )
    return RowValues(values=values, side_hint=side_hint)


@dataclass
class ParsedTextLayer:
    """The statement page parsed from a text layer, plus whether a header was found
    (`row_f1` is not applicable to a page without one)."""

    page: StatementPage
    has_header: bool


def parse_text_layer(
    gt: TextLayerGT,
    *,
    bank: str | None = None,
    page_no: int = 1,
    overrides: BankOverrides | None = None,
) -> ParsedTextLayer:
    """Parse one digital page's ground truth from its positioned words."""
    ov = overrides or BankOverrides()
    statement = StatementPage(bank=bank, page_no=page_no)
    lines = group_lines(list(gt.gt_words))
    band = header_band(lines, ov.date_format)

    above = lines if band is None else lines[: band[0]]
    apply_header_fields(
        statement, extract_header_fields([line.text for line in above]), ov.date_format
    )
    if band is None:
        return ParsedTextLayer(page=statement, has_header=False)

    columns = build_columns(lines, band)
    resolve_amount_split(lines, band, columns)
    previous_date: str | None = None
    for line in lines[band[1] + 1 :]:
        row, opening = build_row(
            _row_values(line, columns, ov),
            page_no=page_no,
            overrides=ov,
            previous_date=previous_date,
        )
        if opening is not None and statement.opening_balance is None:
            statement.opening_balance = opening
        if row is not None:
            statement.rows.append(row)
            previous_date = row.date
    return ParsedTextLayer(page=statement, has_header=True)


def parse_text_layer_rows(
    gt: TextLayerGT,
    *,
    bank: str | None = None,
    page_no: int = 1,
    overrides: BankOverrides | None = None,
) -> StatementPage:
    """The parsed ground-truth statement page (see `parse_text_layer` for the header flag)."""
    return parse_text_layer(gt, bank=bank, page_no=page_no, overrides=overrides).page


# --- scoring -------------------------------------------------------------------------------------


def field_map(page: StatementPage | None) -> dict[str, str | None]:
    """The scored header fields as strings, for `kie_f1.score_fields` on both sides."""
    out: dict[str, str | None] = {}
    for name in SCORED_FIELDS:
        value = None if page is None else getattr(page, name)
        if isinstance(value, Decimal):
            out[name] = format(value, "f")
        else:
            out[name] = canonical_text(value) if isinstance(value, str) else None
    return out


#: The row cells a manual label covers (`amount` only where the side is unknown).
ROW_CELLS = ("date", "debit", "credit", "amount", "balance")


def _as_text(value: object) -> str | None:
    if isinstance(value, Decimal):
        return format(value, "f")
    return canonical_text(value) if isinstance(value, str) else None


def cell_map(page: StatementPage | None) -> dict[str, str | None]:
    """A page as labellable cells: the header fields plus `row[i].<cell>` by position.

    The row index is the row's position on its page, which is what the review queue
    exports and what a manual label refers back to.
    """
    out: dict[str, str | None] = dict(field_map(page))
    for index, row in enumerate(page.rows if page else []):
        for cell in ROW_CELLS:
            value = _as_text(getattr(row, cell))
            if value is not None or cell != "amount":
                out[f"row[{index}].{cell}"] = value
    return out


def normalize_cell(key: str, value: str | None) -> str | None:
    """Normalize a cell value by the type its key implies (`row[3].debit` -> amount)."""
    if value is None:
        return None
    return normalize_field(key.rsplit(".", 1)[-1], value).value


def cell_disagreements(a: dict[str, str | None], b: dict[str, str | None]) -> list[str]:
    """Keys where two cell maps differ, including a value only one side has."""
    return sorted(
        key
        for key in set(a) | set(b)
        if normalize_cell(key, a.get(key)) != normalize_cell(key, b.get(key))
    )


def score_statement(
    pred: StatementPage | None, gt: StatementPage, has_header: bool = True
) -> tuple[dict[str, float | None], list[FieldScore]]:
    """Check A's per-page metrics: header `field_exact` rows and row precision/recall/F1.

    Without a detectable header row the ground truth has no rows to match, so the row
    metrics are not applicable (value None) rather than zero.
    """
    fields = score_fields(field_map(pred), field_map(gt))
    if not has_header:
        return {"row_precision": None, "row_recall": None, "row_f1": None}, fields
    precision, recall, f1 = row_prf(pred.rows if pred else [], gt.rows)
    return {"row_precision": precision, "row_recall": recall, "row_f1": f1}, fields


def worst_case_statement(
    gt: StatementPage, has_header: bool = True
) -> tuple[dict[str, float | None], list[FieldScore]]:
    """The metric names a scored page would get, at their worst values (a failed page)."""
    if not has_header:
        values: dict[str, float | None] = {
            "row_precision": None,
            "row_recall": None,
            "row_f1": None,
        }
    else:
        values = {"row_precision": 0.0, "row_recall": 0.0, "row_f1": 0.0}
    return values, worst_case_fields(field_map(gt))


def row_key(row: StatementRow) -> tuple[str | None, str | None]:
    """A row's match key: normalized date and the absolute amount (debit, credit or the
    unsigned amount of a combined column), with zero tolerance."""
    amount = row.debit if row.debit is not None else row.credit
    if amount is None:
        amount = row.amount
    # Trailing zeros are formatting, not value: "20.00" and "20" are the same amount.
    return row.date, None if amount is None else format(abs(amount).normalize(), "f")


def row_prf(
    pred_rows: list[StatementRow], gt_rows: list[StatementRow]
) -> tuple[float, float, float]:
    """Micro precision, recall and F1 of one-to-one row matches on (date, amount).

    [IMPLEMENTER DECIDES] with no rows on a side, that side's rate is 1 when the other
    side is empty too and 0 otherwise — the same convention as `kie_f1.micro_prf`.
    """
    pred_counts = Counter(row_key(r) for r in pred_rows)
    gt_counts = Counter(row_key(r) for r in gt_rows)
    tp = sum((pred_counts & gt_counts).values())
    precision = tp / len(pred_rows) if pred_rows else (1.0 if not gt_rows else 0.0)
    recall = tp / len(gt_rows) if gt_rows else (1.0 if not pred_rows else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
