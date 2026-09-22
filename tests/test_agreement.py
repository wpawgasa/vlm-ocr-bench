"""TDD tests for Check C, cross-model agreement (task 5.4). Synthetic pages only."""

from decimal import Decimal

from ocr_bench.metrics.agreement import agreement_rate, field_agreement, pair_agreements
from ocr_bench.schemas import StatementPage, StatementRow


def _row(date, amount, balance, *, credit=False):
    value = Decimal(amount)
    return StatementRow(
        date=date,
        debit=None if credit else value,
        credit=value if credit else None,
        balance=None if balance is None else Decimal(balance),
    )


def _page(**kw):
    defaults = dict(
        account_no="111-2-33333-4",
        account_name="MS. SYNTH",
        period_start="2024-03-01",
        period_end="2024-03-31",
        opening_balance=Decimal("100.00"),
        closing_balance=Decimal("130.00"),
        rows=[_row("2024-03-04", "20.00", "80.00"), _row("2024-03-07", "50.00", "130.00")],
    )
    defaults.update(kw)
    return StatementPage(**defaults)


def test_identical_pages_agree_everywhere():
    agree = field_agreement(_page(), _page())
    assert set(agree.values()) == {True}
    assert agreement_rate(agree) == 1.0


def test_missing_in_one_model_is_null():
    agree = field_agreement(_page(), _page(account_no=None))
    assert agree["account_no"] is None
    assert agree["account_name"] is True


def test_differing_value_disagrees():
    agree = field_agreement(_page(), _page(closing_balance=Decimal("999.00")))
    assert agree["closing_balance"] is False
    assert agreement_rate(agree) < 1.0


def test_normalized_comparison_ignores_formatting():
    a = _page(account_no="111-2-33333-4", closing_balance=Decimal("130.0"))
    b = _page(account_no="111233333 4", closing_balance=Decimal("130.00"))
    agree = field_agreement(a, b)
    assert agree["account_no"] is True and agree["closing_balance"] is True


def test_matched_rows_compare_the_balance():
    other = _page(
        rows=[_row("2024-03-04", "20.00", "80.00"), _row("2024-03-07", "50.00", "131.00")]
    )
    agree = field_agreement(_page(), other)
    assert agree["row:2024-03-04|20#0:balance"] is True
    assert agree["row:2024-03-07|50#0:balance"] is False


def test_row_missing_in_one_model_is_a_disagreement():
    other = _page(rows=[_row("2024-03-04", "20.00", "80.00")])
    agree = field_agreement(_page(), other)
    assert agree["row:2024-03-07|50#0:present"] is False


def test_missing_page_gives_all_null():
    agree = field_agreement(_page(), None)
    assert set(agree.values()) == {None}


def test_agreement_rate_ignores_nulls():
    assert agreement_rate({"a": True, "b": False, "c": None}) == 0.5
    assert agreement_rate({"c": None}) is None


def test_pair_agreements_covers_every_model_pair():
    pages = {"teleocr": _page(), "dotsocr": _page(account_no="999"), "typhoon": _page()}
    pairs = pair_agreements(pages)
    assert sorted(pairs) == [
        ("dotsocr", "teleocr"),
        ("dotsocr", "typhoon"),
        ("teleocr", "typhoon"),
    ]
    assert pairs[("dotsocr", "teleocr")]["account_no"] is False
    assert agreement_rate(pairs[("teleocr", "typhoon")]) == 1.0
