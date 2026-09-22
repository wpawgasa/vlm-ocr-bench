"""Check B: arithmetic self-consistency of an extracted statement file (task 5.3).

Every comparison is exact `Decimal` arithmetic. The equations are evaluated in
chronological order (some banks print newest first), opening rows are the opening
balance rather than transactions, and a row whose debit/credit side the extraction did
not give is inferred from its balance delta and marked `side_inferred` — such a row is
consistent by construction, so it is excluded from `row_consistency_rate`.
"""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from ocr_bench.schemas import ErrorFlag, StatementFile, StatementPage, StatementRow


class RowCheck(BaseModel):
    """One row's verdict, in chronological order (`row_index` is 1-based)."""

    model_config = ConfigDict(extra="forbid")

    row_index: int
    page_no: int
    #: Position of the row within its own page, as extracted (the review queue's
    #: `row_index`), so a flag can be matched to a manual label.
    page_row_index: int = 0
    date: str | None = None
    side_inferred: bool = False
    #: True/False when the balance equation was evaluated, None when the row is excluded
    #: (an inferred side, a derived opening row, or a missing balance).
    consistent: bool | None = None
    expected_balance: Decimal | None = None
    actual_balance: Decimal | None = None
    flags: list[ErrorFlag] = Field(default_factory=list)


class ArithmeticResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str
    model: str | None = None
    bank: str | None = None
    n_rows: int = 0
    rows: list[RowCheck] = Field(default_factory=list)
    row_consistency_rate: float | None = None
    file_reconciles: bool = True
    first_break_row: int | None = None
    page_gap_pages: list[int] = Field(default_factory=list)
    balance_mismatch_rows: list[int] = Field(default_factory=list)
    total_mismatch: bool = False
    closing_mismatch: bool = False
    newest_first: bool = False
    opening_balance: Decimal | None = None
    sum_debit: Decimal = Decimal(0)
    sum_credit: Decimal = Decimal(0)
    #: `{row_index: "debit" | "credit"}` for the rows whose side Check B inferred.
    inferred_sides: dict[int, str] = Field(default_factory=dict)
    flags: list[ErrorFlag] = Field(default_factory=list)


def _net(row: StatementRow) -> Decimal | None:
    if row.debit is None and row.credit is None:
        return None
    return (row.credit or Decimal(0)) - (row.debit or Decimal(0))


def _ordered_pages(sf: StatementFile) -> list[StatementPage]:
    return sorted(sf.pages, key=lambda p: p.page_no)


def is_newest_first(rows: list[StatementRow]) -> bool:
    """True when the rows are printed newest first (TTB).

    The dates decide; when they cannot (one date, or all equal) the balance chain does:
    the direction whose balance equation holds for more rows wins, and a tie keeps the
    printed order.
    """
    dates = [r.date for r in rows if r.date]
    if len(dates) >= 2 and dates[0] != dates[-1]:
        return dates[-1] < dates[0]
    return _chain_score(list(reversed(rows))) > _chain_score(rows)


def _chain_score(rows: list[StatementRow]) -> int:
    """How many rows satisfy the balance equation against the previous printed balance."""
    score = 0
    for previous, row in zip(rows, rows[1:], strict=False):
        net = _net(row)
        if previous.balance is None or row.balance is None or net is None:
            continue
        score += row.balance == previous.balance + net
    return score


def _opening_balance(pages: list[StatementPage], rows: list[StatementRow]) -> Decimal | None:
    """The file's opening balance.

    [IMPLEMENTER DECIDES] the first page that prints an opening balance wins; failing
    that it is derived from the first row (its balance minus its net), and failing that
    (an unknown side) the first row's balance is taken as the basis. A derived basis
    makes the first row consistent by construction, so that row is then excluded from
    `row_consistency_rate` exactly like an inferred side.
    """
    for page in pages:
        if page.opening_balance is not None:
            return page.opening_balance
    if not rows or rows[0].balance is None:
        return None
    net = _net(rows[0])
    return rows[0].balance - net if net is not None else rows[0].balance


def check_file(sf: StatementFile, model: str | None = None) -> ArithmeticResult:
    """Evaluate every balance, total and cross-page equation of one statement file."""
    pages = _ordered_pages(sf)
    rows = [(row, index) for page in pages for index, row in enumerate(page.rows)]
    result = ArithmeticResult(file_id=sf.file_id, model=model, bank=sf.bank, n_rows=len(rows))

    result.newest_first = is_newest_first([r for r, _ in rows])
    if result.newest_first:
        rows = list(reversed(rows))

    opening = _opening_balance(pages, [r for r, _ in rows])
    result.opening_balance = opening
    derived_opening = opening is not None and all(p.opening_balance is None for p in pages)

    page_opening = {p.page_no: p.opening_balance for p in pages}
    running = opening
    previous_page: int | None = None
    checked = consistent = 0

    for index, (row, page_row_index) in enumerate(rows, start=1):
        check = RowCheck(
            row_index=index,
            page_no=row.page_no,
            page_row_index=page_row_index,
            date=row.date,
        )
        net = _net(row)
        starts_page = previous_page is not None and row.page_no != previous_page
        page_gap = False

        if starts_page:
            declared = page_opening.get(row.page_no)
            if declared is not None and running is not None and declared != running:
                page_gap = True
                running = declared

        inference_failed = False
        if net is None and row.amount is not None and running is not None:
            # A single unsigned amount column: the balance delta gives the side.
            if row.balance is not None:
                delta = row.balance - running
                if abs(delta) == row.amount and delta != 0:
                    net = delta
                    check.side_inferred = True
                    result.inferred_sides[index] = "credit" if delta > 0 else "debit"
                else:
                    # No side makes the equation hold: a real break, not a row that is
                    # consistent by construction, so it counts against the rate.
                    inference_failed = True

        check.expected_balance = None if running is None or net is None else running + net
        check.actual_balance = row.balance

        if inference_failed:
            check.consistent = False
            checked += 1
            check.flags.append(ErrorFlag.balance_mismatch)
            result.balance_mismatch_rows.append(index)
            if result.first_break_row is None:
                result.first_break_row = index
        elif check.side_inferred or (index == 1 and derived_opening):
            check.consistent = None
        elif check.expected_balance is None or row.balance is None:
            check.consistent = None
        else:
            check.consistent = row.balance == check.expected_balance
            checked += 1
            consistent += int(check.consistent)
            if not check.consistent:
                check.flags.append(ErrorFlag.balance_mismatch)
                result.balance_mismatch_rows.append(index)
                if result.first_break_row is None:
                    result.first_break_row = index
        if check.consistent is False and starts_page:
            page_gap = True
        if page_gap:
            check.flags.append(ErrorFlag.page_gap)
            if row.page_no not in result.page_gap_pages:
                result.page_gap_pages.append(row.page_no)

        if net is not None:
            result.sum_credit += max(net, Decimal(0))
            result.sum_debit += max(-net, Decimal(0))
        # The next row's equation is read against the *printed* previous balance, so one
        # misprinted balance breaks its own row and its successor, not the whole file.
        running = row.balance if row.balance is not None else check.expected_balance
        previous_page = row.page_no
        result.rows.append(check)

    result.row_consistency_rate = consistent / checked if checked else None

    totals = _summary_totals(pages)
    if totals["total_debit"] is not None and totals["total_debit"] != result.sum_debit:
        result.total_mismatch = True
    if totals["total_credit"] is not None and totals["total_credit"] != result.sum_credit:
        result.total_mismatch = True
    closing = totals["closing_balance"]
    if closing is not None and running is not None and closing != running:
        result.closing_mismatch = True

    if result.balance_mismatch_rows:
        result.flags.append(ErrorFlag.balance_mismatch)
    if result.total_mismatch or result.closing_mismatch:
        result.flags.append(ErrorFlag.total_mismatch)
    if result.page_gap_pages:
        result.flags.append(ErrorFlag.page_gap)
    result.file_reconciles = not result.flags
    return result


def _summary_totals(pages: list[StatementPage]) -> dict[str, Decimal | None]:
    """The file's summary totals: the first page that prints each total wins, except the
    closing balance, where the last page's wins (it is printed at the end of the file)."""
    totals: dict[str, Decimal | None] = {
        "total_debit": None,
        "total_credit": None,
        "closing_balance": None,
    }
    for page in pages:
        for name in ("total_debit", "total_credit"):
            if totals[name] is None:
                totals[name] = getattr(page, name)
        if page.closing_balance is not None:
            totals["closing_balance"] = page.closing_balance
    return totals
