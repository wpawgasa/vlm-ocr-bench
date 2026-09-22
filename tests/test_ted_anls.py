"""Tests for ocr_bench.metrics.ted and ocr_bench.metrics.anls (task 4.5)."""

import pytest

from ocr_bench.metrics.anls import anls
from ocr_bench.metrics.ted import first_table, ted, ted_docparse

TABLE = (
    '<table><tr><th>ชื่อ</th><th colspan="2">ยอด</th></tr>'
    "<tr><td>ก</td><td>1</td><td>2</td></tr></table>"
)


def test_spec_identical_table_is_one():
    assert ted(TABLE, TABLE) == 1.0


def test_table_in_surrounding_text_is_found():
    assert ted("# หัวเรื่อง\n\n" + TABLE + "\n\nท้ายหน้า", TABLE) == 1.0


def test_no_table_in_prediction_is_zero():
    assert ted("ไม่มีตาราง", TABLE) == 0.0


def test_changed_cell_scores_between_zero_and_one():
    other = TABLE.replace("<td>2</td>", "<td>3</td>")
    assert 0.0 < ted(other, TABLE) < 1.0


def test_first_table_picks_first():
    two = "<table><tr><td>a</td></tr></table> x <table><tr><td>b</td></tr></table>"
    assert first_table(two) == "<table><tr><td>a</td></tr></table>"
    assert first_table("none") is None


def test_ted_docparse_is_official_doc_parsing_score():
    md = "# หัวข้อ\nบรรทัด\n## ย่อย\nอีกบรรทัด"
    assert ted_docparse(md, md) == 1.0
    assert 0.0 <= ted_docparse("อย่างอื่น", md) < 1.0


def test_anls_exact_is_one():
    assert anls("ใบเสร็จรับเงิน", "ใบเสร็จรับเงิน") == 1.0


def test_anls_canonicalizes_both_sides():
    assert anls("  ใบเสร็จ​รับเงิน ", "ใบเสร็จรับเงิน") == 1.0
    assert anls("ABC", "abc") == 1.0


def test_spec_anls_below_threshold_is_zero():
    # similarity 1 - 3/4 = 0.25 < 0.5
    assert anls("wxyz", "abcz") == 0.0


def test_anls_above_threshold_is_similarity():
    assert anls("abcx", "abcd") == pytest.approx(0.75)


def test_anls_empty_prediction_is_zero():
    assert anls("", "abc") == 0.0
    assert anls(None, "abc") == 0.0
