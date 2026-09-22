"""Tests for ocr_bench.data.gt_utils pure helpers."""

import pytest

from ocr_bench.data.gt_utils import (
    STATEMENT_CRITICAL_FIELDS,
    critical_fields_for,
    flatten_fields,
    markdown_table_to_html,
    parse_answer_json,
    table_answer_to_html,
)


class TestParseAnswerJson:
    def test_plain_json(self):
        assert parse_answer_json('{"a": 1}') == {"a": 1}

    def test_json_fence(self):
        s = '```json\n{"a": 1}\n```'
        assert parse_answer_json(s) == {"a": 1}

    def test_bare_fence(self):
        s = '```\n{"a": 1}\n```'
        assert parse_answer_json(s) == {"a": 1}

    def test_trailing_comma_repaired(self):
        s = '{"a": 1, "b": [1, 2,],}'
        assert parse_answer_json(s) == {"a": 1, "b": [1, 2]}

    def test_fence_and_trailing_comma(self):
        s = '```json\n{"a": 1,}\n```'
        assert parse_answer_json(s) == {"a": 1}

    def test_unparseable_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_answer_json("not json at all {{{")


class TestFlattenFields:
    def test_nested_dict_and_list(self):
        obj = {"a": {"b": [1, "x"]}, "c": None}
        assert flatten_fields(obj) == {"a.b[0]": "1", "a.b[1]": "x", "c": None}

    def test_str_kept(self):
        assert flatten_fields({"a": "hello"}) == {"a": "hello"}

    def test_float_becomes_json_dumped(self):
        assert flatten_fields({"a": 1.5}) == {"a": "1.5"}

    def test_bool_becomes_true_false(self):
        assert flatten_fields({"a": True, "b": False}) == {"a": "true", "b": "false"}

    def test_top_level_non_dict_wrapped(self):
        assert flatten_fields("hello") == {"value": "hello"}
        assert flatten_fields(42) == {"value": "42"}

    def test_deeply_nested(self):
        obj = {"x": {"y": {"z": "deep"}}}
        assert flatten_fields(obj) == {"x.y.z": "deep"}

    def test_list_of_dicts(self):
        obj = {"items": [{"name": "a"}, {"name": "b"}]}
        assert flatten_fields(obj) == {"items[0].name": "a", "items[1].name": "b"}


class TestMarkdownTableToHtml:
    def test_basic_table(self):
        md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        html = markdown_table_to_html(md)
        assert "<table>" in html
        assert "<thead><tr><th>a</th><th>b</th></tr></thead>" in html
        assert "<tbody><tr><td>1</td><td>2</td></tr></tbody>" in html

    def test_escapes_cell_text(self):
        md = "| a | b |\n| --- | --- |\n| <x> | y&z |"
        html = markdown_table_to_html(md)
        assert "&lt;x&gt;" in html
        assert "y&amp;z" in html

    def test_ignores_non_pipe_lines(self):
        md = "some preamble\n| a | b |\n| --- | --- |\n| 1 | 2 |\nsome trailer"
        html = markdown_table_to_html(md)
        assert "preamble" not in html
        assert "trailer" not in html


class TestTableAnswerToHtml:
    def test_full_table_returned_as_is(self):
        ans = "  <table><tr><td>1</td></tr></table>  "
        assert table_answer_to_html(ans) == "<table><tr><td>1</td></tr></table>"

    def test_bare_tr_wrapped(self):
        ans = "<tr><td>1</td></tr>"
        result = table_answer_to_html(ans)
        assert result.startswith("<table>")
        assert result.endswith("</table>")
        assert "<tr><td>1</td></tr>" in result

    def test_markdown_converted(self):
        ans = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        result = table_answer_to_html(ans)
        assert "<table>" in result
        assert "<th>a</th>" in result


class TestCriticalFieldsFor:
    def test_key_match_thai(self):
        fields = {"ราคารวม": "some text"}
        assert critical_fields_for(fields) == ["ราคารวม"]

    def test_key_match_english(self):
        fields = {"total_amount": "some text", "unrelated_key": "value"}
        assert critical_fields_for(fields) == ["total_amount"]

    def test_value_amount_match(self):
        fields = {"x": "1,234.50 บาท"}
        assert critical_fields_for(fields) == ["x"]

    def test_value_amount_thai_abbreviations(self):
        # Real ThaiOCRBench menu prices: "100 บ." and "280.-"
        fields = {"ไก่รวนตะไคร้": "100 บ.", "หมูสันนอกย่าง": "280.-", "จาน": "120-"}
        assert critical_fields_for(fields) == ["ไก่รวนตะไคร้", "หมูสันนอกย่าง", "จาน"]

    def test_bare_yod_ruam_in_dish_names_not_key_match(self):
        # ยอด/รวม occur inside ordinary words (ยอดผักบุ้ง = morning-glory shoots, รวมมิตร = mixed)
        fields = {"ยำยอดผักบุ้งกรอบ": "ใส่กุ้ง", "ยำรวมมิตรทะเล": "เผ็ด"}
        assert critical_fields_for(fields) == []

    def test_compound_total_keywords_match(self):
        fields = {"ยอดรวม": "x", "ยอดชำระ": "x", "รวมเงิน": "x", "รวมทั้งสิ้น": "x", "ยอดคงเหลือ": "x"}
        assert critical_fields_for(fields) == list(fields)

    def test_value_date_match(self):
        fields = {"y": "15/03/2025"}
        assert critical_fields_for(fields) == ["y"]

    def test_value_thai_month_match(self):
        fields = {"y": "15 มีนาคม 2568"}
        assert critical_fields_for(fields) == ["y"]

    def test_value_id_match(self):
        fields = {"z": "123-456-789"}
        assert critical_fields_for(fields) == ["z"]

    def test_no_match(self):
        fields = {"unrelated": "just some plain text"}
        assert critical_fields_for(fields) == []

    def test_null_value_no_value_match_but_key_match(self):
        fields = {"amount": None}
        assert critical_fields_for(fields) == ["amount"]

    def test_null_value_no_key_match(self):
        fields = {"unrelated": None}
        assert critical_fields_for(fields) == []

    def test_insertion_order_preserved(self):
        fields = {"total": "x", "name": "y", "plain": "z"}
        assert critical_fields_for(fields) == ["total", "name"]


def test_statement_critical_fields_constant():
    assert STATEMENT_CRITICAL_FIELDS == [
        "account_no",
        "opening_balance",
        "closing_balance",
        "period_start",
        "period_end",
    ]
