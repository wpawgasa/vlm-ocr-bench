"""Tests for ocr_bench.normalize.fields (task 4.2)."""

import pytest

from ocr_bench.normalize.fields import field_type, normalize_field


def test_spec_buddhist_era_date():
    fv = normalize_field("date", "15/03/2568")
    assert fv.value == "2025-03-15"
    assert fv.raw == "15/03/2568"
    assert fv.flags == []


def test_spec_thai_amount_raw_preserved():
    fv = normalize_field("total_amount", "฿152,340.75 บาท")
    assert fv.value == "152340.75"
    assert fv.raw == "฿152,340.75 บาท"


def test_spec_account_number():
    fv = normalize_field("account_no", "123-4-56789-0")
    assert fv.value == "1234567890"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,000.00", "1000"),  # trailing zeros dropped: 1,000.00 == 1000
        ("1,500.-", "1500"),
        ("THB 20", "20"),
        ("35 บ.", "35"),
        ("RS 99.5", "99.5"),
        ("๑,๒๓๔.๕๐ บาท", "1234.5"),
        ("-500.00", "-500"),
    ],
)
def test_amount_variants(raw, expected):
    assert normalize_field("amount", raw).value == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("5-1-2025", "2025-01-05"),
        ("5.1.2025", "2025-01-05"),
        ("31/12/67", "2024-12-31"),  # 2-digit year >= 40 -> BE 2567
        ("30/10/58", "2015-10-30"),  # BE 2558
        ("01/02/25", "2025-02-01"),  # 2-digit year < 40 -> CE 2025
        ("2025-03-15", "2025-03-15"),
        ("30ต.ค.58", "2015-10-30"),
        ("March 15, 2025", "2025-03-15"),
        ("15 มีนาคม 2568", "2025-03-15"),
        ("15 มี.ค. 68", "2025-03-15"),
        ("3 ธ.ค. 2567", "2024-12-03"),
        ("15 March 2025", "2025-03-15"),
        ("15 Mar 2025", "2025-03-15"),
    ],
)
def test_date_variants(raw, expected):
    assert normalize_field("วันที่", raw).value == expected


def test_account_detected_from_value_shape():
    assert normalize_field("ref", "123-4-56789-0").value == "1234567890"
    assert normalize_field("เลขที่บัญชี", "012 3 45678 9").value == "0123456789"


def test_name_is_canonical_text():
    fv = normalize_field("name", "  นาย​สมชาย   ใจดี ")
    assert fv.value == "นายสมชาย ใจดี"
    assert fv.flags == []


def test_unparseable_amount_flagged_not_dropped():
    fv = normalize_field("total_amount", "ไม่ระบุ")
    assert fv.value == "ไม่ระบุ"
    assert fv.flags == ["unparsed_amount"]


def test_unparseable_date_flagged():
    fv = normalize_field("date", "32/13/2568")
    assert fv.value == "32/13/2568"
    assert fv.flags == ["unparsed_date"]


def test_none_stays_none():
    fv = normalize_field("date", None)
    assert fv.value is None and fv.raw is None and fv.flags == []


def test_same_normalization_both_sides():
    assert normalize_field("amount", "1,000").value == normalize_field("amount", "1000.").value


def test_field_type_inference():
    assert field_type("ยอดรวม", "x") == "amount"
    assert field_type("foo", "1,234.00") == "amount"
    assert field_type("foo", "12/03/2567") == "date"
    assert field_type("บัญชี", "x") == "account"
    assert field_type("ชื่อ", "สมชาย") == "name"
    assert field_type("foo", "bar") == "other"
