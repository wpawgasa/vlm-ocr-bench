"""TDD tests for the model-side statement mapper (task 5.1).

Every fixture here is synthesized: invented account numbers, names and amounts in
layouts that *resemble* the real corpus (KBank's single amount column, TTB's signed
newest-first rows, BBL's bilingual two-line header with a B/F row). No client content.
"""

from decimal import Decimal

import pytest

from ocr_bench.config import BankOverrides
from ocr_bench.normalize.statement import map_statement_page, merge_file
from ocr_bench.schemas import Block, BlockType, NormalizedPage

# --- synthetic pages -------------------------------------------------------------------------

KBANK_TABLE = [
    ["", "Time/", "", "", "Outstanding", ""],
    ["Date", "Eff.Date", "Descriptions", "Withdrawal / Deposit", "Balance", "Channel"],
    ["", "", "", "", "", ""],
    ["01-03-24", "", "Beginning Balance", "", "500.00", ""],
    ["04-03-24", "09:42", "Cash Withdrawal", "200.00", "300.00", "ATM"],
    ["07-03-24", "10:15", "Transfer Deposit", "1,000.00", "1,300.00", "K PLUS"],
]

KBANK_BLOCKS = [
    Block(
        type=BlockType.text,
        text=(
            "Account Number 111-2-33333-4\n"
            "Account Name MS. SYNTH SAMPLE\n"
            "Period 01/03/2024 - 31/03/2024\n"
            "ENDING BALANCE 1,300.00\n"
            "TOTAL WITHDRAWAL 1 ITEMS 200.00\n"
            "TOTAL DEPOSIT 1 ITEMS 1,000.00"
        ),
    )
]

TTB_TABLE = [
    ["Date", "Time", "Descriptions", "Channel", "Amount", "Balance"],
    ["9 Mar 24", "7:47 PM", "Cash Withdrawal", "Mobile", "-250.00", "1,050.00"],
    ["8 Mar 24", "10:09 AM", "Transfer in", "KTB", "+1,200.00", "1,300.00"],
    ["7 Mar 24", "11:00 AM", "Bill payment", "Mobile", "-100.00", "100.00"],
]

BBL_TABLE = [
    ["วันที่", "รายการ", "เลขที่เช็ค", "ถอน", "ฝาก", "คงเหลือ", "ผ่านทาง"],
    ["Date", "Particulars", "Chq.No.", "Withdrawal", "Deposit", "Balance", "Via"],
    ["01/03/24", "B/F", "", "", "", "800.00", ""],
    ["05/03/24", "PMT FOR GOODS", "", "300.00", "", "500.00", "Auto"],
    ["09/03/24", "TRF FR OTH BK", "", "", "250.00", "750.00", "mPhone"],
]

BBL_BLOCKS = [
    Block(type=BlockType.text, text="ชื่อ/Name นาย ตัวอย่าง สังเคราะห์"),
    Block(type=BlockType.text, text="เลขที่บัญชี/Account No. 222-3-44444-5"),
    Block(
        type=BlockType.text,
        text="รอบรายการบัญชี / Statement Period 01/03/2024 - 31/03/2024",
    ),
]


def _page(tables, blocks=(), text="page text"):
    return NormalizedPage(text=text, blocks=list(blocks), tables=tables)


# --- the three bank styles -------------------------------------------------------------------


class TestKbankStyle:
    @pytest.fixture
    def mapped(self):
        return map_statement_page(
            _page([KBANK_TABLE], KBANK_BLOCKS), bank="kbank", page_no=1, overrides=None
        )

    def test_three_line_header_and_opening_row(self, mapped):
        assert mapped.opening_balance == Decimal("500.00")
        assert len(mapped.rows) == 2

    def test_combined_column_leaves_side_unknown_with_amount_kept(self, mapped):
        row = mapped.rows[0]
        assert row.date == "2024-03-04"
        assert row.debit is None and row.credit is None
        assert row.amount == Decimal("200.00")
        assert row.side_inferred is False
        assert row.balance == Decimal("300.00")
        assert row.description == "Cash Withdrawal"
        assert row.channel == "ATM"

    def test_header_fields(self, mapped):
        assert mapped.account_no == "111-2-33333-4"
        assert mapped.account_name == "MS. SYNTH SAMPLE"
        assert mapped.period_start == "2024-03-01"
        assert mapped.period_end == "2024-03-31"
        assert mapped.closing_balance == Decimal("1300.00")
        assert mapped.total_debit == Decimal("200.00")
        assert mapped.total_credit == Decimal("1000.00")
        assert mapped.bank == "kbank" and mapped.page_no == 1


class TestTtbStyle:
    @pytest.fixture
    def mapped(self):
        return map_statement_page(_page([TTB_TABLE]), bank="ttb", page_no=1, overrides=None)

    def test_signed_amounts_give_the_side(self, mapped):
        assert [r.date for r in mapped.rows] == ["2024-03-09", "2024-03-08", "2024-03-07"]
        assert mapped.rows[0].debit == Decimal("250.00") and mapped.rows[0].credit is None
        assert mapped.rows[1].credit == Decimal("1200.00") and mapped.rows[1].debit is None
        assert all(r.amount is None for r in mapped.rows)

    def test_rows_keep_source_order(self, mapped):
        assert [r.balance for r in mapped.rows] == [
            Decimal("1050.00"),
            Decimal("1300.00"),
            Decimal("100.00"),
        ]


class TestBblStyle:
    @pytest.fixture
    def mapped(self):
        return map_statement_page(
            _page([BBL_TABLE], BBL_BLOCKS), bank="bbl", page_no=2, overrides=None
        )

    def test_bilingual_header_separate_columns_and_bf_row(self, mapped):
        assert mapped.opening_balance == Decimal("800.00")
        assert [r.debit for r in mapped.rows] == [Decimal("300.00"), None]
        assert [r.credit for r in mapped.rows] == [None, Decimal("250.00")]
        assert [r.balance for r in mapped.rows] == [Decimal("500.00"), Decimal("750.00")]
        assert mapped.rows[0].description == "PMT FOR GOODS"

    def test_thai_header_fields(self, mapped):
        assert mapped.account_no == "222-3-44444-5"
        assert mapped.account_name == "นาย ตัวอย่าง สังเคราะห์"
        assert mapped.period_start == "2024-03-01"
        assert mapped.page_no == 2


# --- robustness and purity -------------------------------------------------------------------


def test_garbage_page_gives_empty_rows_without_raising():
    page = _page(
        [[["qwrtyp", "zzz"], ["###", "@@@"]]],
        [Block(type=BlockType.text, text="0HHSXRW ,BBMSLR=R@RDK")],
        text="0HHSXRW ,BBMSLR=R@RDK",
    )
    mapped = map_statement_page(page, bank="ktb", page_no=1, overrides=None)
    assert mapped.rows == []
    assert mapped.opening_balance is None and mapped.account_no is None


def test_no_tables_at_all_gives_empty_rows():
    mapped = map_statement_page(_page([]), bank=None, page_no=1, overrides=None)
    assert mapped.rows == []


def test_model_independent_mapping():
    """Same tables and blocks from two models -> identical statement pages."""
    a = map_statement_page(
        _page([KBANK_TABLE], KBANK_BLOCKS, text="model A reading order"),
        bank="kbank",
        page_no=1,
        overrides=None,
    )
    b = map_statement_page(
        _page([KBANK_TABLE], KBANK_BLOCKS, text="<table>model B html</table>"),
        bank="kbank",
        page_no=1,
        overrides=None,
    )
    assert a == b


def test_best_scoring_table_is_chosen():
    other = [["Ref. No.", "Code"], ["A1", "B2"]]
    page = _page([other, KBANK_TABLE], KBANK_BLOCKS)
    mapped = map_statement_page(page, bank="kbank", page_no=1, overrides=None)
    assert len(mapped.rows) == 2


def test_date_format_override_wins_over_the_generic_parser():
    table = [
        ["Date", "Description", "Withdrawal", "Deposit", "Balance"],
        ["03/04/24", "Fee", "10.00", "", "90.00"],
    ]
    overrides = BankOverrides(date_format="%m/%d/%y")
    mapped = map_statement_page(_page([table]), bank="xx", page_no=1, overrides=overrides)
    assert mapped.rows[0].date == "2024-03-04"


# --- merge -------------------------------------------------------------------------------------


def test_merge_four_pages_tags_rows_with_their_page():
    pages = []
    for page_no in range(1, 5):
        table = [
            ["Date", "Description", "Withdrawal", "Deposit", "Balance"],
            [f"0{page_no}/03/24", "Fee", "10.00", "", f"{100 - page_no}.00"],
        ]
        pages.append(
            map_statement_page(_page([table]), bank="kbank", page_no=page_no, overrides=None)
        )
    merged = merge_file(list(reversed(pages)), file_id="BS-kbank-0001", bank="kbank")
    assert merged.file_id == "BS-kbank-0001" and merged.bank == "kbank"
    assert [p.page_no for p in merged.pages] == [1, 2, 3, 4]
    assert [r.page_no for r in merged.rows] == [1, 2, 3, 4]
    assert [r.date for r in merged.rows] == [
        "2024-03-01",
        "2024-03-02",
        "2024-03-03",
        "2024-03-04",
    ]


def test_table_row_without_a_date_continues_the_previous_row():
    table = [
        ["Date", "Description", "Withdrawal", "Deposit", "Balance"],
        ["01/03/24", "Fee", "10.00", "", "90.00"],
        ["", "Fee", "20.00", "", "70.00"],
        ["", "see above", "", "", ""],
        ["03/03/24", "In", "", "30.00", "100.00"],
    ]
    mapped = map_statement_page(_page([table]), bank="kbank", page_no=1, overrides=None)
    assert [r.date for r in mapped.rows] == ["2024-03-01", "2024-03-01", "2024-03-03"]
    assert [r.debit for r in mapped.rows] == [Decimal("10.00"), Decimal("20.00"), None]
