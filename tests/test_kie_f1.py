"""Tests for ocr_bench.metrics.kie_f1 (task 4.6)."""

import pytest

from ocr_bench.metrics.kie_f1 import micro_prf, score_fields, worst_case_fields


def _by_field(rows):
    return {r.field: r for r in rows}


def test_spec_false_accept():
    rows = _by_field(score_fields({"id_number": "1234"}, {"id_number": None}))
    r = rows["id_number"]
    assert r.false_accept is True
    assert r.false_reject is False
    assert r.field_exact == 0


def test_spec_false_reject():
    rows = _by_field(score_fields({"total_amount": None}, {"total_amount": "500.00"}))
    r = rows["total_amount"]
    assert r.false_reject is True
    assert r.false_accept is False
    assert r.field_exact == 0


def test_missing_prediction_key_is_false_reject():
    rows = _by_field(score_fields({}, {"total_amount": "500.00"}))
    assert rows["total_amount"].false_reject is True
    assert rows["total_amount"].pred is None


def test_null_equals_null_is_correct():
    r = _by_field(score_fields({"x": None}, {"x": None}))["x"]
    assert r.field_exact == 1 and r.field_fuzzy == 1
    assert not r.false_accept and not r.false_reject


def test_exact_uses_type_normalization_both_sides():
    r = _by_field(score_fields({"วันที่": "2025-03-15"}, {"วันที่": "15/03/2568"}))["วันที่"]
    assert r.field_exact == 1
    r = _by_field(score_fields({"ยอดรวม": "1,000.00 บาท"}, {"ยอดรวม": "1000"}))["ยอดรวม"]
    assert r.field_exact == 1


def test_fuzzy_threshold():
    gt = "abcdefghijklmnopqrst"  # 20 chars: 2 edits = 0.1 -> fuzzy, 3 edits -> not
    assert _by_field(score_fields({"n": "abcdefghijklmnopqrXX"}, {"n": gt}))["n"].field_fuzzy == 1
    assert _by_field(score_fields({"n": "abcdefghijklmnopqXXX"}, {"n": gt}))["n"].field_fuzzy == 0


def test_extra_prediction_field_gets_row_with_gt_none():
    rows = _by_field(score_fields({"a": "1", "extra": "z"}, {"a": "1"}))
    assert rows["extra"].gt is None
    assert rows["extra"].field_exact == 0
    assert rows["extra"].false_accept is True


def test_micro_prf_counts_extra_as_false_positive():
    rows = score_fields({"a": "1", "b": "2", "extra": "z"}, {"a": "1", "b": "9", "c": None})
    p, r, f1 = micro_prf(rows)
    assert p == pytest.approx(1 / 3)
    assert r == pytest.approx(1 / 2)
    assert f1 == pytest.approx(2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2))


def test_micro_prf_all_null_both_sides_is_perfect():
    assert micro_prf(score_fields({"a": None}, {"a": None})) == (1.0, 1.0, 1.0)


def test_micro_prf_nothing_predicted():
    assert micro_prf(score_fields({}, {"a": "1"})) == (0.0, 0.0, 0.0)


def test_worst_case_fields():
    rows = _by_field(worst_case_fields({"a": "1", "b": None}))
    assert rows["a"].field_exact == 0 and rows["a"].field_fuzzy == 0 and rows["a"].false_reject
    assert rows["b"].field_exact == 0 and rows["b"].field_fuzzy == 0
    assert not rows["b"].false_reject and not rows["b"].false_accept
    assert micro_prf(list(rows.values())) == (0.0, 0.0, 0.0)


def test_blank_string_is_null_on_both_sides():
    r = _by_field(score_fields({"x": "  "}, {"x": ""}))["x"]
    assert r.field_exact == 1 and not r.false_accept
    r = _by_field(score_fields({"x": "abc"}, {"x": ""}))["x"]
    assert r.false_accept is True


def test_lenient_finds_true_value_inside_a_labelled_prediction():
    rows = _by_field(
        score_fields(
            {"price": "ราคาตั๋ว / Ticket Price: 5 บาท / Baht", "seat": "25", "no": "x" * 80 + " 5"},
            {"price": "5", "seat": "5", "no": "5"},
        )
    )
    assert rows["price"].field_exact == 0 and rows["price"].field_lenient == 1
    assert rows["seat"].field_lenient == 0  # "5" is not a token of "25"
    assert rows["no"].field_lenient == 0  # too much extra text to count as found


def test_lenient_is_never_below_exact():
    rows = score_fields({"a": "x", "b": None}, {"a": "x", "b": None})
    assert all(r.field_lenient >= r.field_exact for r in rows)
