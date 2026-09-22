"""Tests for ocr_bench.normalize.text (task 4.1)."""

from ocr_bench.normalize.text import canonical_text, plain_text


def test_spec_scenario_thai_digits_and_zero_width():
    assert canonical_text("๑๒๓​ 45") == "123 45"


def test_nfc_applied():
    # "e" + combining acute -> precomposed é
    assert canonical_text("é") == "é"


def test_zero_width_variants_removed():
    s = "ก​ข‌ค‍ง⁠จ﻿ฉ"
    assert canonical_text(s) == "กขคงจฉ"


def test_whitespace_collapsed_and_stripped():
    assert canonical_text("  a \t\n b  c  ") == "a b c"


def test_legacy_thai_pua_glyphs_mapped():
    # BBL text layers: "ที่" with the low-left mai ek variant, "ปี" with the left sara ii.
    assert canonical_text("ที") == "ที่"
    assert canonical_text("ป") == "ปี"
    assert canonical_text("") == "ฐญ"
    assert canonical_text("") == "์์ฺ"


def test_every_pua_codepoint_maps_to_thai_block():
    for cp in range(0xF700, 0xF71B):
        out = canonical_text(chr(cp))
        assert len(out) == 1 and 0x0E00 <= ord(out) <= 0x0E7F, hex(cp)


def test_plain_text_table_html():
    s = "<table><tr><th>ชื่อ</th><th>ยอด</th></tr><tr><td>ก&amp;ข</td><td>๑๐</td></tr></table>"
    assert plain_text(s) == "ชื่อ ยอด ก&ข 10"


def test_plain_text_markdown():
    s = "# หัวข้อ\n\n**ตัวหนา** และ _เอียง_\n\n| a | b |\n|---|:-:|\n| 1 | 2 |\n\n$$x$$"
    assert plain_text(s) == "หัวข้อ ตัวหนา และ เอียง a b 1 2 x"


def test_plain_text_figure_dropped_page_number_kept():
    s = "ก่อน<figure>ภาพ<b>x</b></figure>หลัง <page_number>12</page_number><br>จบ</p>ท้าย"
    assert plain_text(s) == "ก่อน หลัง 12 จบ ท้าย"


def test_plain_text_same_on_both_sides():
    gt = "ยอดรวม ๑,๐๐๐ บาท"
    pred = "<p>ยอดรวม 1,000 บาท</p>"
    assert plain_text(gt) == plain_text(pred)
