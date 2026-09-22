"""Tests for ocr_bench.models.parsers on recorded responses in tests/fixtures/responses/."""

from pathlib import Path

import pytest

from ocr_bench.models.parsers import (
    PARSERS,
    ParseContext,
    fix_equation,
    html_tables,
    otsl_to_html,
    parse,
    parse_teleocr_layout,
    teleocr_page,
)
from ocr_bench.schemas import BlockType, NormalizedPage, Task

FIXTURES = Path(__file__).parent / "fixtures" / "responses"


def fx(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_registry_has_every_dialect():
    assert {
        "markdown",
        "html_table",
        "json_fields",
        "dots_layout_json",
        "teleocr_layout",
        "otsl",
        "typhoon_markdown",
    } <= set(PARSERS)


# --- OTSL -----------------------------------------------------------------------------------


def test_otsl_spec_scenario_merged_cell():
    html = otsl_to_html("<fcel>A<lcel><nl><fcel>1<fcel>2<nl>")
    assert html == ('<table><tr><td colspan="2">A</td></tr><tr><td>1</td><td>2</td></tr></table>')
    [(_, rows)] = html_tables(html)
    assert rows == [["A"], ["1", "2"]]


def test_otsl_rowspan_and_xcel():
    html = otsl_to_html("<fcel>A<lcel><fcel>B<nl><xcel><xcel><ucel><nl><fcel>1<fcel>2<fcel>3<nl>")
    assert '<td colspan="2" rowspan="2">A</td>' in html
    assert '<td rowspan="2">B</td>' in html
    assert html.count("<tr>") == 3


def test_otsl_empty_cell_and_escaping():
    html = otsl_to_html("<fcel>a<b<ecel><nl>")
    assert html == "<table><tr><td>a&lt;b</td><td></td></tr></table>"


def test_otsl_existing_html_kept():
    html = "<table><tr><td>x</td></tr></table>"
    assert otsl_to_html(f"  {html}\n") == html


def test_otsl_parser_on_fixture():
    page = parse("otsl", fx("teleocr_table_otsl.txt"), ParseContext(task=Task.table))
    assert page.parse_error is None
    assert page.tables == [
        [["วันที่", "รายการ", "ยอด"], ["01/03", "ฝาก", "1,000.00"], ["รวม", "1,000.00"]]
    ]
    assert page.blocks[0].type == BlockType.table
    assert 'colspan="2"' in page.blocks[0].text


def test_otsl_parser_without_otsl_sets_parse_error():
    page = parse("otsl", "just some text", ParseContext(task=Task.table))
    assert page.parse_error is not None
    assert page.text == "just some text"
    assert page.tables == []


# --- markdown / html_table ------------------------------------------------------------------


def test_markdown_plain_text():
    page = parse("markdown", "  สวัสดีครับ\n", ParseContext())
    assert page.text == "สวัสดีครับ"
    assert page.parse_error is None


def test_markdown_pipe_table_converted_to_html():
    page = parse("markdown", fx("markdown_pipe_table.md"), ParseContext())
    assert page.tables == [[["สินค้า", "ราคา"], ["ข้าว", "50"], ["น้ำ", "10"]]]
    assert page.blocks[0].type == BlockType.table
    assert page.blocks[0].text.startswith("<table>")


def test_html_table_extracts_rows_and_cells():
    raw = "<table><tr><th>ก</th><th>ข</th></tr><tr><td>1</td><td> 2 </td></tr></table>"
    page = parse("html_table", raw, ParseContext(task=Task.table))
    assert page.tables == [[["ก", "ข"], ["1", "2"]]]
    assert page.parse_error is None


def test_html_table_without_table_sets_parse_error():
    page = parse("html_table", "no table here", ParseContext(task=Task.table))
    assert page.parse_error is not None
    assert page.text == "no table here"


# --- json_fields ----------------------------------------------------------------------------


def test_json_fields_flattens_like_ground_truth():
    page = parse("json_fields", fx("kie_fenced.txt"), ParseContext(task=Task.kie))
    assert page.parse_error is None
    assert page.fields["ชื่อร้าน"].value == "ร้านข้าวมันไก่"
    assert page.fields["ราคารวม"].value == "120"
    assert page.fields["รายการ[0].ชื่อ"].value == "ข้าวมันไก่"
    assert page.fields["หมายเหตุ"].value is None


def test_json_fields_tolerates_prose_around_json():
    raw = 'นี่คือผลลัพธ์: {"a": "1", "b": 2} ครับ'
    page = parse("json_fields", raw, ParseContext(task=Task.kie_map))
    assert page.parse_error is None
    assert page.fields["a"].value == "1"


def test_json_fields_invalid_json_keeps_text_and_sets_parse_error():
    raw = fx("kie_invalid.txt")
    page = parse("json_fields", raw, ParseContext(task=Task.kie))
    assert page.fields == {}
    assert page.parse_error is not None
    assert page.text == raw.strip()


def test_json_fields_classify_uses_label():
    page = parse("json_fields", "  ใบเสร็จรับเงิน \n", ParseContext(task=Task.classify))
    assert page.label == "ใบเสร็จรับเงิน"
    assert page.parse_error is None


# --- dots_layout_json -----------------------------------------------------------------------


def test_dots_layout_blocks_mapped_to_original_pixels():
    ctx = ParseContext(task=Task.ocr_fullpage, orig_size=(1200, 1600), sent_size=(600, 800))
    page = parse("dots_layout_json", fx("dots_layout_fenced.txt"), ctx)
    assert page.parse_error is None
    types = [b.type for b in page.blocks]
    assert types == [
        BlockType.header,
        BlockType.text,
        BlockType.text,
        BlockType.table,
        BlockType.figure,
        BlockType.text,
        BlockType.footer,
    ]
    assert page.blocks[0].bbox == (80.0, 40.0, 1120.0, 120.0)
    assert page.tables == [
        [["วันที่", "รายการ", "ยอดคงเหลือ"], ["01/03/2568", "ฝากเงิน", "1,000.00"], ["รวม", "1,000.00"]]
    ]
    assert page.blocks[3].text.startswith("<table>")
    assert "ธนาคารกสิกรไทย" in page.text
    assert "เลขที่บัญชี 123-4-56789-0" in page.text
    assert page.text.index("ธนาคารกสิกรไทย") < page.text.index("หน้า 1/2")


def test_dots_layout_truncated_json_salvages_complete_blocks():
    page = parse("dots_layout_json", fx("dots_layout_truncated.txt"), ParseContext())
    assert [b.text for b in page.blocks] == ["ใบแจ้งยอด", "ชื่อบัญชี นายสมชาย ใจดี"]
    assert page.parse_error is not None and "salvaged 2" in page.parse_error
    assert page.blocks[0].bbox == (10.0, 10.0, 200.0, 40.0)  # no scale info -> identity


def test_dots_layout_garbage_keeps_text_and_sets_parse_error():
    raw = fx("dots_layout_garbage.txt")
    page = parse("dots_layout_json", raw, ParseContext())
    assert page.blocks == []
    assert page.text == raw.strip()
    assert page.parse_error is not None


# --- teleocr_layout -------------------------------------------------------------------------


def test_parse_teleocr_layout_fixture():
    items, warnings = parse_teleocr_layout(fx("teleocr_layout.txt"))
    assert [i.label for i in items] == [
        "header",
        "text",
        "table",
        "seal",
        "image",
        "list",
        "text",
        "equation",
        "page_number",
    ]
    assert [i.index for i in items] == list(range(9))
    assert items[0].points == [(100, 50), (900, 100)]
    assert len(items[3].points) == 4
    assert items[0].angle == 0
    assert items[6].angle == 90
    assert len(warnings) == 4
    assert any("banana" in w for w in warnings)


def test_teleocr_angle_mapping():
    raw = "\n".join(
        f"<box:0 0 10 10><label:text><rotate_{a}>" for a in ("up", "right", "down", "left", "x")
    )
    items, _ = parse_teleocr_layout(raw)
    assert [i.angle for i in items] == [0, 90, 180, 270, None]


def test_teleocr_page_assembles_blocks_in_reading_order():
    items, _ = parse_teleocr_layout(fx("teleocr_layout.txt"))
    texts = {1: "บรรทัดที่หนึ่ง", 2: otsl_to_html("<fcel>A<lcel><nl><fcel>1<fcel>2<nl>"), 0: "หัว"}
    page = teleocr_page(
        items, texts, ParseContext(orig_size=(2000, 1000)), skip_types=["list", "equation_block"]
    )
    labels = [b.type for b in page.blocks]
    assert BlockType.figure in labels  # the image block is kept as a figure without text
    assert len(page.blocks) == 8  # list is skipped
    assert page.blocks[0].type == BlockType.header
    assert page.blocks[0].bbox == (200.0, 50.0, 1800.0, 100.0)
    assert page.tables == [[["A"], ["1", "2"]]]
    assert page.text.startswith("หัว\n\nบรรทัดที่หนึ่ง")
    footer = [b for b in page.blocks if b.type == BlockType.footer]
    assert len(footer) == 1


def test_teleocr_layout_parser_on_garbage_sets_parse_error():
    page = parse("teleocr_layout", "nothing useful", ParseContext(orig_size=(10, 10)))
    assert page.parse_error is not None


def test_fix_equation():
    assert fix_equation("\\[ x^2 \\]") == "$$x^2$$"
    assert fix_equation("x^2") == "$$x^2$$"
    assert fix_equation("$x$") == "$x$"
    assert fix_equation("$$x$$") == "$$x$$"


# --- typhoon_markdown -----------------------------------------------------------------------


def test_typhoon_markdown_fixture():
    page = parse("typhoon_markdown", fx("typhoon_page.md"), ParseContext(task=Task.statement))
    assert page.parse_error is None
    assert "โลโก้ธนาคาร" not in page.text  # figure descriptions excluded from text
    assert "<figure>" not in page.text
    assert "<page_number>" not in page.text
    assert "เลขที่บัญชี 111-2-33333-4" in page.text
    assert page.tables == [
        [["วันที่", "ถอน", "ฝาก", "คงเหลือ"], ["02/03/2568", "500.00", "", "9,500.00"]]
    ]
    types = [b.type for b in page.blocks]
    assert BlockType.table in types and BlockType.figure in types
    footer = [b for b in page.blocks if b.type == BlockType.footer]
    assert footer[0].text == "3"


def test_typhoon_truncated_figure_removed():
    page = parse("typhoon_markdown", "ข้อความ\n<figure>\nคำอธิบายที่ถูกตัด", ParseContext())
    assert page.text == "ข้อความ"


# --- never raises ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PARSERS))
@pytest.mark.parametrize("raw", ["", "{", "<table><tr><td>x", "<box:1 2", "```json\n[{"])
def test_parsers_never_raise(name, raw):
    page = parse(name, raw, ParseContext(task=Task.kie, orig_size=(10, 10), sent_size=(10, 10)))
    assert isinstance(page, NormalizedPage)


def test_unknown_parser_name_is_a_parse_error():
    page = parse("nope", "text", ParseContext())
    assert page.parse_error is not None and "nope" in page.parse_error
