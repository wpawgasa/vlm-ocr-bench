"""TDD tests for Check A: statement ground truth parsed from positioned words (task 5.2).

The word layouts are synthesized to *resemble* the three real digital styles (KBank's
single "Withdrawal / Deposit" column, TTB's signed newest-first amounts, BBL's bilingual
two-line header with a B/F row): invented account numbers, names, dates and amounts at
hand-written 200-dpi x positions. No client content.
"""

from decimal import Decimal

import pytest

from ocr_bench.config import BankOverrides
from ocr_bench.metrics.statement_gt import (
    field_map,
    parse_text_layer,
    parse_text_layer_rows,
    row_prf,
)
from ocr_bench.schemas import StatementPage, StatementRow, TextLayerGT, Word

LINE_HEIGHT = 18


def _line(y, items):
    return [Word(text=t, bbox=(x0, float(y), x1, float(y + LINE_HEIGHT))) for t, x0, x1 in items]


def _gt(*lines, text="synthetic page"):
    words = [w for line in lines for w in line]
    return TextLayerGT(gt_kind="text_layer", gt_text=text, gt_words=words)


# --- KBank style: one amount column, side from x position ------------------------------------

KBANK = _gt(
    _line(100, [("Account", 960, 1030), ("Number", 1036, 1104), ("111-2-33333-4", 1150, 1300)]),
    _line(
        140,
        [
            ("Period", 960, 1016),
            ("01/03/2024", 1150, 1244),
            ("-", 1250, 1256),
            ("31/03/2024", 1262, 1356),
        ],
    ),
    _line(180, [("ENDING", 960, 1030), ("BALANCE", 1036, 1124), ("1,300.00", 1490, 1570)]),
    _line(
        220,
        [
            ("TOTAL", 960, 1018),
            ("WITHDRAWAL", 1024, 1150),
            ("1", 1158, 1170),
            ("ITEMS", 1180, 1240),
            ("200.00", 1500, 1570),
        ],
    ),
    _line(
        260,
        [
            ("TOTAL", 960, 1018),
            ("DEPOSIT", 1024, 1104),
            ("1", 1158, 1170),
            ("ITEMS", 1180, 1240),
            ("1,000.00", 1490, 1570),
        ],
    ),
    _line(400, [("Time/", 228, 276), ("Outstanding", 831, 935)]),
    _line(
        420,
        [
            ("Date", 143, 183),
            ("Descriptions", 364, 471),
            ("Withdrawal", 573, 668),
            ("/", 675, 681),
            ("Deposit", 688, 754),
            ("Channel", 1052, 1122),
        ],
    ),
    _line(440, [("Eff.Date", 218, 286), ("Balance", 848, 917)]),
    _line(
        480,
        [
            ("01-03-24", 126, 201),
            ("Beginning", 300, 378),
            ("Balance", 385, 449),
            ("500.00", 909, 969),
        ],
    ),
    _line(
        520,
        [
            ("04-03-24", 126, 201),
            ("09:42", 231, 277),
            ("Cash", 300, 341),
            ("Withdrawal", 347, 435),
            ("200.00", 638, 694),
            ("300.00", 909, 969),
            ("ATM", 980, 1017),
        ],
    ),
    _line(
        560,
        [
            ("07-03-24", 126, 201),
            ("10:15", 231, 277),
            ("Transfer", 300, 364),
            ("Deposit", 371, 431),
            ("1,000.00", 708, 780),
            ("1,300.00", 897, 969),
            ("K", 980, 992),
            ("PLUS", 999, 1044),
        ],
    ),
)


class TestKbankWords:
    @pytest.fixture
    def page(self):
        return parse_text_layer_rows(KBANK, bank="kbank", page_no=1, overrides=None)

    def test_opening_row_and_row_count(self, page):
        assert page.opening_balance == Decimal("500.00")
        assert len(page.rows) == 2

    def test_single_column_side_comes_from_x_position(self, page):
        assert page.rows[0].debit == Decimal("200.00") and page.rows[0].credit is None
        assert page.rows[1].credit == Decimal("1000.00") and page.rows[1].debit is None
        assert [r.balance for r in page.rows] == [Decimal("300.00"), Decimal("1300.00")]
        assert [r.date for r in page.rows] == ["2024-03-04", "2024-03-07"]

    def test_header_fields_from_labelled_words_above_the_table(self, page):
        assert page.account_no == "111-2-33333-4"
        assert page.period_start == "2024-03-01" and page.period_end == "2024-03-31"
        assert page.closing_balance == Decimal("1300.00")
        assert page.total_debit == Decimal("200.00")
        assert page.total_credit == Decimal("1000.00")

    def test_channel_column(self, page):
        assert page.rows[0].channel == "ATM"

    def test_split_disabled_leaves_the_side_unknown(self):
        page = parse_text_layer_rows(
            KBANK, bank="kbank", page_no=1, overrides=BankOverrides(amount_column_split=False)
        )
        assert page.rows[0].debit is None and page.rows[0].credit is None
        assert page.rows[0].amount == Decimal("200.00")


# --- TTB style: signed amounts, newest first --------------------------------------------------

TTB = _gt(
    _line(100, [("Customer", 88, 211), ("Name", 218, 288)]),
    _line(140, [("Account", 911, 997), ("Number", 1003, 1080), ("555-6-77777-8", 1155, 1300)]),
    _line(
        180,
        [
            ("Period", 908, 974),
            ("1", 1155, 1163),
            ("Mar", 1169, 1206),
            ("24", 1211, 1237),
            ("-", 1242, 1252),
            ("31", 1257, 1283),
            ("Mar", 1289, 1326),
            ("24", 1332, 1357),
        ],
    ),
    _line(
        300,
        [
            ("Date", 111, 158),
            ("Time", 277, 326),
            ("Descriptions", 444, 572),
            ("Channel", 777, 860),
            ("Amount", 1176, 1255),
            ("Balance", 1461, 1541),
        ],
    ),
    _line(
        340,
        [
            ("9", 111, 124),
            ("Mar", 129, 167),
            ("24", 172, 198),
            ("7:47", 277, 319),
            ("PM", 324, 355),
            ("Cash", 444, 495),
            ("Withdrawal", 501, 611),
            ("Mobile", 777, 844),
            ("-250.00", 1173, 1255),
            ("1,050.00", 1461, 1541),
        ],
    ),
    _line(
        380,
        [
            ("8", 111, 124),
            ("Mar", 129, 167),
            ("24", 172, 198),
            ("10:09", 277, 331),
            ("AM", 336, 369),
            ("Transfer", 444, 530),
            ("in", 535, 553),
            ("KTB", 777, 820),
            ("+1,200.00", 1138, 1255),
            ("1,300.00", 1461, 1541),
        ],
    ),
    _line(
        420,
        [
            ("7", 111, 124),
            ("Mar", 129, 167),
            ("24", 172, 198),
            ("11:00", 277, 331),
            ("AM", 336, 369),
            ("Bill", 444, 480),
            ("payment", 486, 580),
            ("Mobile", 777, 844),
            ("-100.00", 1179, 1255),
            ("100.00", 1473, 1541),
        ],
    ),
)


class TestTtbWords:
    @pytest.fixture
    def page(self):
        return parse_text_layer_rows(
            TTB, bank="ttb", page_no=1, overrides=BankOverrides(signed_amount=True)
        )

    def test_multi_token_dates_and_source_order(self, page):
        assert [r.date for r in page.rows] == ["2024-03-09", "2024-03-08", "2024-03-07"]

    def test_signs_give_the_side(self, page):
        assert page.rows[0].debit == Decimal("250.00")
        assert page.rows[1].credit == Decimal("1200.00")
        assert page.rows[2].debit == Decimal("100.00")
        assert all(r.amount is None for r in page.rows)

    def test_description_and_balance(self, page):
        assert page.rows[0].description == "Cash Withdrawal"
        assert page.rows[1].balance == Decimal("1300.00")

    def test_labels_without_a_value_stay_empty(self, page):
        assert page.account_name is None
        assert page.account_no == "555-6-77777-8"
        assert page.period_start == "2024-03-01" and page.period_end == "2024-03-31"


# --- BBL style: bilingual two-line header, separate columns, B/F row --------------------------

BBL = _gt(
    _line(100, [("ชื่อ/Name", 55, 149), ("นาย", 400, 450), ("ตัวอย่าง", 456, 540)]),
    _line(
        140,
        [
            ("เลขที่บัญชี/", 855, 1045),
            ("Account", 1050, 1130),
            ("No.", 1135, 1180),
            ("222-3-44444-5", 1250, 1400),
        ],
    ),
    _line(
        180,
        [
            ("รอบรายการบัญชี", 855, 1026),
            ("/", 1034, 1040),
            ("Statement", 1047, 1147),
            ("Period", 1154, 1218),
            ("01/03/2024", 1250, 1359),
            ("-", 1365, 1372),
            ("31/03/2024", 1378, 1487),
        ],
    ),
    _line(
        300,
        [
            ("วันที่", 81, 126),
            ("รายการ", 235, 311),
            ("เลขที่เช็ค", 450, 541),
            ("ถอน", 740, 786),
            ("ฝาก", 942, 983),
            ("คงเหลือ", 1100, 1180),
            ("ผ่านทาง", 1359, 1440),
        ],
    ),
    _line(
        340,
        [
            ("Date", 81, 127),
            ("Particulars", 222, 325),
            ("Chq.No.", 455, 535),
            ("Withdrawal", 678, 786),
            ("Deposit", 909, 983),
            ("Balance", 1101, 1180),
            ("Via", 1384, 1415),
        ],
    ),
    _line(380, [("01/03/24", 68, 139), ("B/F", 158, 185), ("800.00", 1124, 1180)]),
    _line(
        420,
        [
            ("05/03/24", 68, 139),
            ("PMT", 158, 194),
            ("FOR", 199, 236),
            ("GOODS", 241, 307),
            ("300.00", 730, 786),
            ("500.00", 1124, 1180),
            ("Auto", 1202, 1240),
        ],
    ),
    _line(
        460,
        [
            ("09/03/24", 68, 139),
            ("TRF", 158, 190),
            ("FR", 195, 218),
            ("OTH", 223, 260),
            ("BK", 265, 289),
            ("250.00", 912, 983),
            ("750.00", 1109, 1180),
            ("mPhone", 1202, 1269),
        ],
    ),
)


class TestBblWords:
    @pytest.fixture
    def page(self):
        return parse_text_layer_rows(BBL, bank="bbl", page_no=3, overrides=None)

    def test_bf_row_is_the_opening_balance(self, page):
        assert page.opening_balance == Decimal("800.00")
        assert len(page.rows) == 2

    def test_separate_withdrawal_and_deposit_columns(self, page):
        assert [r.debit for r in page.rows] == [Decimal("300.00"), None]
        assert [r.credit for r in page.rows] == [None, Decimal("250.00")]
        assert [r.balance for r in page.rows] == [Decimal("500.00"), Decimal("750.00")]
        assert page.rows[0].description == "PMT FOR GOODS"

    def test_thai_header_fields(self, page):
        assert page.account_no == "222-3-44444-5"
        assert page.account_name == "นาย ตัวอย่าง"
        assert page.period_start == "2024-03-01" and page.period_end == "2024-03-31"
        assert page.page_no == 3 and page.bank == "bbl"


# --- no header --------------------------------------------------------------------------------


def test_page_without_a_column_header_is_flagged():
    gt = _gt(
        _line(100, [("Account", 100, 200), ("Number", 210, 300), ("999-9-99999-9", 320, 470)]),
        _line(140, [("Dear", 100, 150), ("customer,", 160, 260)]),
        _line(180, [("thank", 100, 160), ("you", 170, 210)]),
    )
    parsed = parse_text_layer(gt, bank="kbank", page_no=1, overrides=None)
    assert parsed.has_header is False
    assert parsed.page.rows == []
    assert parsed.page.account_no == "999-9-99999-9"


def test_empty_word_list_does_not_raise():
    parsed = parse_text_layer(
        TextLayerGT(gt_kind="text_layer", gt_text="", gt_words=[]),
        bank=None,
        page_no=1,
        overrides=None,
    )
    assert parsed.has_header is False and parsed.page.rows == []


# --- scoring helpers ---------------------------------------------------------------------------


def _row(date, debit=None, credit=None, balance=None, amount=None):
    return StatementRow(
        date=date,
        debit=None if debit is None else Decimal(debit),
        credit=None if credit is None else Decimal(credit),
        amount=None if amount is None else Decimal(amount),
        balance=None if balance is None else Decimal(balance),
    )


def test_row_prf_matches_on_date_and_amount_one_to_one():
    gt = [_row("2024-03-04", debit="200.00"), _row("2024-03-07", credit="1000.00")]
    pred = [
        _row("2024-03-04", amount="200.00"),  # side unknown still matches on |amount|
        _row("2024-03-07", credit="999.00"),  # wrong amount
        _row("2024-03-09", debit="5.00"),  # extra
    ]
    precision, recall, f1 = row_prf(pred, gt)
    assert precision == pytest.approx(1 / 3)
    assert recall == pytest.approx(1 / 2)
    assert f1 == pytest.approx(2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2))


def test_row_prf_perfect_and_empty():
    gt = [_row("2024-03-04", debit="200.00")]
    assert row_prf(list(gt), gt) == (1.0, 1.0, 1.0)
    assert row_prf([], []) == (1.0, 1.0, 1.0)
    # `kie_f1.micro_prf`'s convention: rows predicted where there are none is a miss on
    # both sides, not a free recall point.
    assert row_prf([_row("2024-03-04", debit="1.00")], []) == (0.0, 0.0, 0.0)
    assert row_prf([], [_row("2024-03-04", debit="1.00")]) == (0.0, 0.0, 0.0)


def test_field_map_exposes_the_scored_header_fields():
    page = StatementPage(
        account_no="111-2-33333-4",
        account_name="MS. SYNTH",
        period_start="2024-03-01",
        period_end="2024-03-31",
        opening_balance=Decimal("500.00"),
        closing_balance=Decimal("1300.00"),
    )
    assert field_map(page) == {
        "account_no": "111-2-33333-4",
        "account_name": "MS. SYNTH",
        "period_start": "2024-03-01",
        "period_end": "2024-03-31",
        "opening_balance": "500.00",
        "closing_balance": "1300.00",
    }
    assert field_map(StatementPage())["account_no"] is None


# --- end to end on a generated digital PDF -----------------------------------------------------

COLUMNS = {"date": 0.05, "description": 0.22, "withdrawal": 0.45, "deposit": 0.60, "balance": 0.78}
PDF_ROWS = [
    ("01/03/24", "B/F", "", "", "100.00"),
    ("04/03/24", "Fee", "20.00", "", "80.00"),
    ("07/03/24", "Salary", "", "50.00", "130.00"),
]


@pytest.fixture
def statement_pdf(tmp_path):
    """A synthetic one-page digital statement PDF with a real text layer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams["pdf.fonttype"] = 42
    path = tmp_path / "statement.pdf"
    with PdfPages(str(path)) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.05, 0.92, "Account Number 111-2-33333-4", fontsize=8)
        fig.text(0.05, 0.89, "Period 01/03/2024 - 31/03/2024", fontsize=8)
        header = ("Date", "Description", "Withdrawal", "Deposit", "Balance")
        for x, text in zip(COLUMNS.values(), header, strict=True):
            fig.text(x, 0.80, text, fontsize=8)
        for index, cells in enumerate(PDF_ROWS):
            for x, text in zip(COLUMNS.values(), cells, strict=True):
                if text:
                    fig.text(x, 0.76 - index * 0.04, text, fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)
    return path


def test_generated_digital_pdf_parses_end_to_end(statement_pdf):
    from ocr_bench.data.bankstmt import text_layer

    gt = text_layer(statement_pdf, 1, 200)
    parsed = parse_text_layer(gt, bank="synth", page_no=1, overrides=None)
    assert parsed.has_header is True
    page = parsed.page
    assert page.account_no == "111-2-33333-4"
    assert page.period_start == "2024-03-01" and page.period_end == "2024-03-31"
    assert page.opening_balance == Decimal("100.00")
    assert [(r.date, r.debit, r.credit, r.balance) for r in page.rows] == [
        ("2024-03-04", Decimal("20.00"), None, Decimal("80.00")),
        ("2024-03-07", None, Decimal("50.00"), Decimal("130.00")),
    ]


# --- the combined column's split ---------------------------------------------------------------


def _kbank_like(header_items, rows):
    """A KBank-like page: a combined amount column whose header may sit off-centre."""
    lines = [_line(300, header_items)]
    for index, (date, amount, x0, x1, balance) in enumerate(rows):
        lines.append(
            _line(
                340 + index * 40,
                [(date, 126, 201), ("Payment", 300, 380), (amount, x0, x1), (balance, 909, 969)],
            )
        )
    return _gt(*lines)


HEADER_OFF_CENTRE = [
    ("Date", 145, 182),
    ("Descriptions", 386, 450),
    # a Thai header whose "/" sits left of the middle of the two figure groups
    ("ถอนเงิน", 589, 655),
    ("/", 660, 671),
    ("ฝากเงิน", 676, 739),
    ("Balance", 831, 935),
]


def test_side_follows_the_right_aligned_figure_groups_not_the_header_middle():
    """Real KBank pages right-align withdrawals at one x and deposits further right; the
    header's separator can sit inside the withdrawal group's span."""
    gt = _kbank_like(
        HEADER_OFF_CENTRE,
        [
            ("01-03-24", "10.00", 638, 694, "90.00"),
            ("02-03-24", "5,000.00", 623, 694, "5,090.00"),
            ("03-03-24", "20.00", 709, 781, "5,110.00"),
        ],
    )
    page = parse_text_layer_rows(gt, bank="kbank", page_no=1, overrides=None)
    assert [r.debit for r in page.rows] == [Decimal("10.00"), Decimal("5000.00"), None]
    assert [r.credit for r in page.rows] == [None, None, Decimal("20.00")]


def test_one_figure_group_falls_back_to_the_header_split():
    gt = _kbank_like(
        HEADER_OFF_CENTRE,
        [
            ("01-03-24", "10.00", 638, 694, "90.00"),
            ("02-03-24", "30.00", 638, 694, "60.00"),
        ],
    )
    page = parse_text_layer_rows(gt, bank="kbank", page_no=1, overrides=None)
    assert [r.debit for r in page.rows] == [Decimal("10.00"), Decimal("30.00")]


def test_a_row_without_a_date_continues_the_previous_row_s_date():
    """Banks omit the date on further transactions of the same day (KBank does); such a
    line is a transaction, not a description continuation."""
    gt = _gt(
        _line(
            300,
            [
                ("Date", 143, 183),
                ("Descriptions", 364, 471),
                ("Withdrawal", 573, 668),
                ("/", 675, 681),
                ("Deposit", 688, 754),
                ("Balance", 831, 935),
            ],
        ),
        _line(
            340,
            [("01-03-24", 126, 201), ("Fee", 300, 340), ("10.00", 638, 694), ("90.00", 909, 969)],
        ),
        _line(380, [("Fee", 300, 340), ("20.00", 638, 694), ("70.00", 909, 969)]),
        _line(420, [("LTD.", 300, 360)]),
        _line(
            460,
            [("03-03-24", 126, 201), ("In", 300, 330), ("30.00", 709, 781), ("100.00", 909, 969)],
        ),
    )
    page = parse_text_layer_rows(gt, bank="kbank", page_no=1, overrides=None)
    assert [r.date for r in page.rows] == ["2024-03-01", "2024-03-01", "2024-03-03"]
    assert [r.debit for r in page.rows] == [Decimal("10.00"), Decimal("20.00"), None]


def test_photo_ground_truth_is_parsed_from_unwarped_words():
    import numpy as np

    from ocr_bench.metrics.statement_gt import unwarp_words
    from ocr_bench.schemas import Word

    h = np.array([[0.998, 0.038, -54.7], [-0.074, 1.015, 57.0], [-1.6e-05, 1.4e-05, 1.0]])
    word = Word(text="1,554.22", bbox=(100.0, 200.0, 180.0, 220.0))
    corners = np.array([[100, 200, 1], [180, 200, 1], [180, 220, 1], [100, 220, 1]]) @ h.T
    xs, ys = corners[:, 0] / corners[:, 2], corners[:, 1] / corners[:, 2]
    warped = Word(text=word.text, bbox=(xs.min(), ys.min(), xs.max(), ys.max()))
    (back,) = unwarp_words([warped], h.tolist())
    assert back.bbox == pytest.approx((100.0, 200.0, 180.0, 220.0), abs=3.0)
    assert unwarp_words([word], None) == [word]
