"""TDD tests for Check B, arithmetic self-consistency (task 5.3). Synthetic rows only."""

from decimal import Decimal

from ocr_bench.metrics.arithmetic import check_file
from ocr_bench.schemas import ErrorFlag, StatementFile, StatementPage, StatementRow


def _row(date, *, debit=None, credit=None, amount=None, balance=None, page_no=1):
    return StatementRow(
        date=date,
        debit=None if debit is None else Decimal(debit),
        credit=None if credit is None else Decimal(credit),
        amount=None if amount is None else Decimal(amount),
        balance=None if balance is None else Decimal(balance),
        page_no=page_no,
    )


def _file(rows, *, opening="100.00", page_no=1, **page_kw):
    page = StatementPage(
        page_no=page_no,
        opening_balance=None if opening is None else Decimal(opening),
        rows=rows,
        **page_kw,
    )
    return StatementFile(file_id="BS-synth-0001", bank="synth", pages=[page])


def _chain(n, *, start=Decimal("100.00"), reverse=False, break_at=None):
    """`n` consistent alternating debit/credit rows from `start`.

    `break_at` (1-based) misprints that row's *amount*, so only its own equation breaks:
    every printed balance stays on the true chain, as with one misread digit.
    """
    rows, balance = [], start
    for i in range(1, n + 1):
        amount = Decimal("10.00") * i
        credit = i % 2 == 1
        balance = balance + amount if credit else balance - amount
        shown = amount + Decimal("7.00") if i == break_at else amount
        rows.append(
            _row(
                f"2024-03-{i:02d}",
                credit=str(shown) if credit else None,
                debit=None if credit else str(shown),
                balance=str(balance),
            )
        )
    return list(reversed(rows)) if reverse else rows


class TestConsistentFile:
    def test_file_reconciles(self):
        result = check_file(_file(_chain(5)))
        assert result.file_reconciles is True
        assert result.row_consistency_rate == 1.0
        assert result.first_break_row is None
        assert result.flags == []

    def test_closing_balance_checked(self):
        rows = _chain(3)
        page = StatementPage(
            page_no=1,
            opening_balance=Decimal("100.00"),
            closing_balance=rows[-1].balance,
            rows=rows,
        )
        result = check_file(StatementFile(file_id="f", pages=[page]))
        assert result.file_reconciles is True and result.closing_mismatch is False

    def test_closing_balance_mismatch_flags_total_mismatch(self):
        rows = _chain(3)
        page = StatementPage(
            page_no=1,
            opening_balance=Decimal("100.00"),
            closing_balance=Decimal("1.00"),
            rows=rows,
        )
        result = check_file(StatementFile(file_id="f", pages=[page]))
        assert result.closing_mismatch is True
        assert ErrorFlag.total_mismatch in result.flags
        assert result.file_reconciles is False


class TestBrokenRow:
    def test_row_seven_is_the_first_break(self):
        result = check_file(_file(_chain(8, break_at=7)))
        assert result.first_break_row == 7
        assert result.file_reconciles is False
        broken = result.rows[6]
        assert broken.row_index == 7
        assert ErrorFlag.balance_mismatch in broken.flags
        assert broken.consistent is False
        # only row 7's own equation breaks: the chain continues from its printed balance
        assert [r.row_index for r in result.rows if r.consistent is False] == [7]
        assert result.row_consistency_rate == 7 / 8


class TestNewestFirst:
    def test_reversed_rows_reconcile(self):
        result = check_file(_file(_chain(5, reverse=True)))
        assert result.newest_first is True
        assert result.file_reconciles is True
        assert result.row_consistency_rate == 1.0
        assert [r.date for r in result.rows] == [f"2024-03-0{i}" for i in range(1, 6)]

    def test_chronological_rows_are_not_reversed(self):
        assert check_file(_file(_chain(5))).newest_first is False


class TestInferredSide:
    def test_inferred_rows_are_excluded_from_the_rate(self):
        rows = [
            _row("2024-03-01", amount="10.00", balance="90.00"),
            _row("2024-03-02", credit="40.00", balance="130.00"),
            _row("2024-03-03", amount="30.00", balance="160.00"),
        ]
        result = check_file(_file(rows))
        assert [r.side_inferred for r in result.rows] == [True, False, True]
        assert [r.consistent for r in result.rows] == [None, True, None]
        assert result.row_consistency_rate == 1.0
        assert result.file_reconciles is True
        assert result.inferred_sides == {1: "debit", 3: "credit"}

    def test_an_amount_that_matches_no_balance_delta_is_a_mismatch(self):
        rows = [
            _row("2024-03-01", amount="10.00", balance="90.00"),
            _row("2024-03-02", amount="99.00", balance="95.00"),
        ]
        result = check_file(_file(rows))
        assert result.rows[1].side_inferred is False
        assert result.rows[1].consistent is False
        assert ErrorFlag.balance_mismatch in result.rows[1].flags
        assert result.first_break_row == 2


class TestOpeningBalance:
    def test_missing_opening_balance_is_derived_from_the_first_row(self):
        result = check_file(_file(_chain(4), opening=None))
        assert result.opening_balance == Decimal("100.00")
        assert result.file_reconciles is True
        # the derived row is consistent by construction, so it is excluded
        assert result.rows[0].consistent is None
        assert result.row_consistency_rate == 1.0

    def test_no_rows_gives_no_rate(self):
        result = check_file(_file([]))
        assert result.row_consistency_rate is None
        assert result.file_reconciles is True and result.n_rows == 0


class TestTotals:
    def test_total_mismatch(self):
        result = check_file(
            _file(_chain(4), total_debit=Decimal("999.00"), total_credit=Decimal("40.00"))
        )
        assert result.total_mismatch is True
        assert ErrorFlag.total_mismatch in result.flags
        assert result.file_reconciles is False

    def test_matching_totals(self):
        # _chain(4): credits 10 + 30, debits 20 + 40
        result = check_file(
            _file(_chain(4), total_debit=Decimal("60.00"), total_credit=Decimal("40.00"))
        )
        assert result.total_mismatch is False and result.file_reconciles is True


class TestPageGap:
    def _two_pages(self, second_opening_balance):
        page1 = StatementPage(
            page_no=1,
            opening_balance=Decimal("100.00"),
            rows=[_row("2024-03-01", credit="50.00", balance="150.00", page_no=1)],
        )
        page2 = StatementPage(
            page_no=2,
            rows=[
                _row("2024-03-02", debit="20.00", balance=second_opening_balance, page_no=2),
                _row("2024-03-03", debit="30.00", balance="100.00", page_no=2),
            ],
        )
        return StatementFile(file_id="BS-synth-0002", pages=[page1, page2])

    def test_continuous_pages_reconcile(self):
        result = check_file(self._two_pages("130.00"))
        assert result.page_gap_pages == []
        assert result.file_reconciles is True

    def test_discontinuity_flags_the_page(self):
        result = check_file(self._two_pages("500.00"))
        assert result.page_gap_pages == [2]
        assert ErrorFlag.page_gap in result.flags
        assert result.file_reconciles is False

    def test_page_opening_balance_must_match_the_previous_page(self):
        file = self._two_pages("130.00")
        file.pages[1].opening_balance = Decimal("999.00")
        result = check_file(file)
        assert result.page_gap_pages == [2]
