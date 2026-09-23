"""Parsers per output dialect (design D5): raw model text -> `NormalizedPage`.

Every parser has the signature `(raw_text, ctx) -> NormalizedPage` and is reached through
`parse(name, raw, ctx)`, which never raises: a failure yields a page that keeps the plain
text and records `parse_error` (output-normalization spec).

Conventions shared by all parsers:
- Block bboxes are `[x0, y0, x1, y1]` in the pixel space of the *original* manifest image.
- A table block's `text` holds the table as HTML; `NormalizedPage.tables` holds the same
  tables as rows x cells (cell text only, spans not expanded).
"""

import html as html_lib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import lxml.html

from ocr_bench.data.gt_utils import flatten_fields, markdown_table_to_html, parse_answer_json
from ocr_bench.schemas import Block, BlockType, FieldValue, NormalizedPage, Task


@dataclass
class ParseContext:
    task: Task | None = None
    orig_size: tuple[int, int] | None = None  # (width, height) of the original image
    sent_size: tuple[int, int] | None = None  # (width, height) of the image actually sent

    @property
    def scale(self) -> tuple[float, float]:
        """Factors mapping sent-image pixels to original-image pixels."""
        if self.orig_size is None or self.sent_size is None:
            return 1.0, 1.0
        return (
            self.orig_size[0] / max(self.sent_size[0], 1),
            self.orig_size[1] / max(self.sent_size[1], 1),
        )


Parser = Callable[[str, ParseContext], NormalizedPage]


# --- tables ---------------------------------------------------------------------------------

_TABLE_RE = re.compile(r"<table\b.*?(?:</table>|$)", re.DOTALL | re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}")


def _table_rows(fragment: str) -> list[list[str]] | None:
    try:
        root = lxml.html.fragment_fromstring(fragment, create_parent="div")
    except Exception:
        return None
    tables = root.xpath(".//table")
    if not tables:
        return None
    rows: list[list[str]] = []
    for tr in tables[0].iter("tr"):
        cells = [cell.text_content().strip() for cell in tr.xpath("./td|./th")]
        rows.append(cells)
    return rows


def html_tables(text: str) -> list[tuple[str, list[list[str]]]]:
    """Find `<table>` elements in `text`; return (html, rows x cells) per table."""
    found: list[tuple[str, list[list[str]]]] = []
    for match in _TABLE_RE.finditer(text):
        fragment = match.group(0)
        if not fragment.lower().rstrip().endswith("</table>"):
            fragment = fragment + "</table>"
        rows = _table_rows(fragment)
        if rows is not None:
            found.append((fragment.strip(), rows))
    return found


def pipe_tables(text: str) -> list[tuple[str, list[list[str]]]]:
    """Markdown pipe tables (with a `|---|` separator line) converted to HTML."""
    found: list[tuple[str, list[list[str]]]] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 2 and any(_SEPARATOR_RE.match(line) for line in run):
            html = markdown_table_to_html("\n".join(run))
            rows = _table_rows(html)
            if rows is not None:
                found.append((html, rows))
        run.clear()

    for line in text.splitlines():
        if "|" in line:
            run.append(line)
        else:
            flush()
    flush()
    return found


def _tables_page(text: str, extra_blocks: list[Block] | None = None) -> NormalizedPage:
    tables = html_tables(text) or pipe_tables(text)
    blocks = [Block(type=BlockType.table, text=h) for h, _ in tables]
    return NormalizedPage(
        text=text,
        blocks=blocks + (extra_blocks or []),
        tables=[rows for _, rows in tables],
    )


# --- OTSL (TeleOCR tables) ------------------------------------------------------------------

_OTSL_TAGS = ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>")
_OTSL_TOKEN_RE = re.compile(r"(<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)")


def is_otsl(text: str) -> bool:
    return any(tag in text for tag in _OTSL_TAGS)


def otsl_to_html(otsl: str) -> str:
    """Port of TeleOCR's reference `convert_otsl_to_html`.

    Rows are separated by `<nl>`. `<fcel>` opens a cell with content and `<ecel>` an empty
    one; `<lcel>` merges into the cell on its left, `<ucel>` into the cell above, and
    `<xcel>` into both, so spans extend right over lcel/xcel and down over ucel/xcel.
    Text that is already an HTML table is returned unchanged.
    """
    stripped = otsl.strip()
    if stripped.startswith("<table") and stripped.endswith("</table>"):
        return stripped
    stripped = stripped.replace("<otsl>", "").replace("</otsl>", "")

    grid: list[list[tuple[str, str]]] = []
    for line in stripped.split("<nl>"):
        parts = _OTSL_TOKEN_RE.split(line)
        row: list[tuple[str, str]] = []
        i = 1
        while i < len(parts):
            row.append((parts[i], parts[i + 1] if i + 1 < len(parts) else ""))
            i += 2
        if row:
            grid.append(row)

    def tag_at(r: int, c: int) -> str | None:
        if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
            return grid[r][c][0]
        return None

    out = ["<table>"]
    for r, row in enumerate(grid):
        out.append("<tr>")
        for c, (tag, content) in enumerate(row):
            if tag not in ("<fcel>", "<ecel>"):
                continue
            colspan = 1
            while tag_at(r, c + colspan) in ("<lcel>", "<xcel>"):
                colspan += 1
            rowspan = 1
            while tag_at(r + rowspan, c) in ("<ucel>", "<xcel>"):
                rowspan += 1
            attrs = ""
            if colspan > 1:
                attrs += f' colspan="{colspan}"'
            if rowspan > 1:
                attrs += f' rowspan="{rowspan}"'
            text = html_lib.escape(content.strip(), quote=False) if tag == "<fcel>" else ""
            out.append(f"<td{attrs}>{text}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def fix_equation(text: str) -> str:
    """TeleOCR equation post-processing: strip a `\\[ \\]` wrapper, then `$$`-wrap."""
    s = text.strip()
    if s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s
    return f"$${s}$$"


def _otsl(raw: str, ctx: ParseContext) -> NormalizedPage:
    text = raw.strip()
    if not (is_otsl(text) or (text.startswith("<table") and text.endswith("</table>"))):
        return NormalizedPage(text=text, parse_error="no OTSL table in output")
    html = otsl_to_html(text)
    page = _tables_page(html)
    if not page.tables:
        return NormalizedPage(text=text, parse_error="OTSL could not be converted")
    return page


# --- plain markdown / HTML ------------------------------------------------------------------


def _markdown(raw: str, ctx: ParseContext) -> NormalizedPage:
    return _tables_page(raw.strip())


def _html_table(raw: str, ctx: ParseContext) -> NormalizedPage:
    page = _tables_page(raw.strip())
    if not page.tables:
        page.parse_error = "no table in output"
    return page


# --- JSON fields (kie / kie_map / classify fallback) ----------------------------------------


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    return candidates


def _json_fields(raw: str, ctx: ParseContext) -> NormalizedPage:
    text = raw.strip()
    if ctx.task == Task.classify:
        return NormalizedPage(text=text, label=text)
    last_error = "empty output"
    for candidate in _json_candidates(text):
        if not candidate:
            continue
        try:
            parsed = parse_answer_json(candidate)
        except ValueError as exc:
            last_error = str(exc)
            continue
        fields = {k: FieldValue(value=v, raw=v) for k, v in flatten_fields(parsed).items()}
        return NormalizedPage(text=text, fields=fields)
    return NormalizedPage(text=text, parse_error=f"invalid JSON: {last_error[:200]}")


# --- dots.ocr layout JSON -------------------------------------------------------------------

_FENCE_OPEN_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")

_DOTS_CATEGORY = {
    "Page-header": BlockType.header,
    "Page-footer": BlockType.footer,
    "Table": BlockType.table,
    "Picture": BlockType.figure,
}


def _salvage_json_list(text: str) -> list:
    """Complete top-level objects of a (possibly truncated) JSON list, in order.

    Salvage strategy for truncated dots.ocr output: decode objects one by one after the
    opening `[` and keep every object that decodes completely.
    """
    start = text.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    items: list = []
    pos = start + 1
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            break
        try:
            obj, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        items.append(obj)
    return items


def _load_dots_items(raw: str) -> tuple[list, str | None]:
    text = _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", raw.strip(), count=1))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        items = _salvage_json_list(text)
        if items:
            return items, f"truncated JSON: salvaged {len(items)} complete blocks"
        return [], "invalid layout JSON"
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if "bbox" not in data and lists else [data]
    if not isinstance(data, list):
        return [], "layout JSON is not a list"
    return data, None


def _scaled_bbox(value: object, ctx: ParseContext) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    sx, sy = ctx.scale
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def _dots_layout_json(raw: str, ctx: ParseContext) -> NormalizedPage:
    items, error = _load_dots_items(raw)
    if not items:
        return NormalizedPage(text=raw.strip(), parse_error=error or "no layout blocks")
    blocks: list[Block] = []
    tables: list[list[list[str]]] = []
    texts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "Text"))
        block_type = _DOTS_CATEGORY.get(category, BlockType.text)
        text = item.get("text")
        text = text if isinstance(text, str) else ""
        blocks.append(Block(type=block_type, text=text, bbox=_scaled_bbox(item.get("bbox"), ctx)))
        if block_type == BlockType.table:
            tables.extend(rows for _, rows in html_tables(text))
        if block_type != BlockType.figure and text:
            texts.append(text)
    return NormalizedPage(text="\n\n".join(texts), blocks=blocks, tables=tables, parse_error=error)


# --- TeleOCR layout -------------------------------------------------------------------------

_TELEOCR_LINE_RE = re.compile(r"^<box:([\d\s]+)><label:(\w+)><([^>]+)>$")
TELEOCR_LABELS = frozenset(
    {
        "text",
        "title",
        "table",
        "image",
        "code",
        "algorithm",
        "header",
        "footer",
        "page_number",
        "page_footnote",
        "aside_text",
        "equation",
        "equation_block",
        "ref_text",
        "list",
        "phonetic",
        "table_caption",
        "image_caption",
        "code_caption",
        "table_footnote",
        "image_footnote",
        "unknown",
        "seal",
        "char",
    }
)
_TELEOCR_ANGLES = (("up", 0), ("right", 90), ("down", 180), ("left", 270))
_TELEOCR_BLOCK_TYPE = {
    "table": BlockType.table,
    "image": BlockType.figure,
    "char": BlockType.figure,
    "header": BlockType.header,
    "footer": BlockType.footer,
    "page_number": BlockType.footer,
    "page_footnote": BlockType.footer,
}


@dataclass
class LayoutItem:
    index: int
    label: str
    points: list[tuple[int, int]]  # 0-1000 units of the original image
    angle: int | None

    def bbox_1000(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


def parse_teleocr_layout(raw: str) -> tuple[list[LayoutItem], list[str]]:
    """Parse TeleOCR layout lines `<box:x y ...><label:type><angle>`; returns (items, warnings)."""
    items: list[LayoutItem] = []
    warnings: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _TELEOCR_LINE_RE.match(line)
        if match is None:
            warnings.append(f"unparseable layout line: {line[:80]!r}")
            continue
        numbers = [int(n) for n in match.group(1).split()]
        if len(numbers) < 4 or len(numbers) % 2:
            warnings.append(f"bad box point count {len(numbers)}: {line[:80]!r}")
            continue
        if any(n < 0 or n > 1000 for n in numbers):
            warnings.append(f"box value outside 0-1000: {line[:80]!r}")
            continue
        label = match.group(2).lower()
        if label not in TELEOCR_LABELS:
            warnings.append(f"unknown layout label '{label}'")
            continue
        angle_text = match.group(3)
        angle = next((deg for word, deg in _TELEOCR_ANGLES if word in angle_text), None)
        points = list(zip(numbers[0::2], numbers[1::2], strict=True))
        items.append(LayoutItem(index=len(items), label=label, points=points, angle=angle))
    return items, warnings


def teleocr_page(
    items: list[LayoutItem],
    texts: dict[int, str],
    ctx: ParseContext,
    skip_types: list[str] | tuple[str, ...] = ("list", "equation_block"),
) -> NormalizedPage:
    """Assemble a TeleOCR page from layout items and (post-processed) block texts.

    Container types (`list`, `equation_block`) are dropped; `image` stays as a figure
    block without text. Table block texts are expected to be HTML already.
    """
    width, height = ctx.orig_size or (1000, 1000)
    blocks: list[Block] = []
    tables: list[list[list[str]]] = []
    page_texts: list[str] = []
    for item in items:
        if item.label in skip_types and item.label != "image":
            continue
        block_type = _TELEOCR_BLOCK_TYPE.get(item.label, BlockType.text)
        x0, y0, x1, y1 = item.bbox_1000()
        bbox = (x0 * width / 1000, y0 * height / 1000, x1 * width / 1000, y1 * height / 1000)
        text = texts.get(item.index, "")
        blocks.append(Block(type=block_type, text=text, bbox=bbox))
        if block_type == BlockType.table:
            tables.extend(rows for _, rows in html_tables(text))
        if block_type != BlockType.figure and text:
            page_texts.append(text)
    return NormalizedPage(text="\n\n".join(page_texts), blocks=blocks, tables=tables)


def _teleocr_layout(raw: str, ctx: ParseContext) -> NormalizedPage:
    items, warnings = parse_teleocr_layout(raw)
    page = teleocr_page(items, {}, ctx)
    if not items and raw.strip():
        page.parse_error = "no valid layout blocks" + (f": {warnings[0]}" if warnings else "")
    return page


# --- typhoon-ocr1.5 markdown ----------------------------------------------------------------

_FIGURE_RE = re.compile(r"<figure>(.*?)(?:</figure>|$)", re.DOTALL | re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(r"<page_number>(.*?)</page_number>", re.DOTALL | re.IGNORECASE)
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _typhoon_markdown(raw: str, ctx: ParseContext) -> NormalizedPage:
    extra: list[Block] = []
    figures = [m.group(1).strip() for m in _FIGURE_RE.finditer(raw)]
    text = _FIGURE_RE.sub("", raw)
    extra.extend(Block(type=BlockType.figure, text=f) for f in figures)
    page_numbers = [m.group(1).strip() for m in _PAGE_NUMBER_RE.finditer(text)]
    text = _PAGE_NUMBER_RE.sub(lambda m: m.group(1).strip(), text)
    extra.extend(Block(type=BlockType.footer, text=n) for n in page_numbers)
    text = _BLANK_LINES_RE.sub("\n\n", text).strip()
    return _tables_page(text, extra)


# --- OvisOCR2 markdown ----------------------------------------------------------------------

_OVIS_IMG_RE = re.compile(r'<img src="images/bbox_(\d+)_(\d+)_(\d+)_(\d+)\.jpg" />')


def clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """OvisOCR2's official post-processing (model card `OvisOCR2Parser`, Apache-2.0): cut a
    degenerate repeated tail of a long output down to one period."""
    n = len(text)
    if n < min_text_len:
        return text
    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue
        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1
        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]
    return text


def _ovis_markdown(raw: str, ctx: ParseContext) -> NormalizedPage:
    """Markdown with HTML tables; figures are `<img src="images/bbox_l_t_r_b.jpg" />` in 0-1000
    units of the sent image. As the official parser: drop the image-tag paragraphs, then cut
    a repeated tail. The tags become figure blocks (bboxes in original-image pixels)."""
    width, height = ctx.orig_size or (1000, 1000)
    figures = [
        Block(
            type=BlockType.figure,
            bbox=(
                int(m.group(1)) * width / 1000,
                int(m.group(2)) * height / 1000,
                int(m.group(3)) * width / 1000,
                int(m.group(4)) * height / 1000,
            ),
        )
        for m in _OVIS_IMG_RE.finditer(raw)
    ]
    text = "\n\n".join(
        block
        for block in raw.strip().split("\n\n")
        if not block.strip().startswith('<img src="images/bbox_')
    )
    return _tables_page(clean_truncated_repeats(text), figures)


# --- PaddleOCR-VL pipeline ------------------------------------------------------------------

# PP-DocLayoutV3 labels (PaddleX pipeline config PaddleOCR-VL-1.6.yaml); unlisted -> text.
_PADDLE_BLOCK_TYPE = {
    "table": BlockType.table,
    "image": BlockType.figure,
    "chart": BlockType.figure,
    "seal": BlockType.figure,
    "header_image": BlockType.figure,
    "footer_image": BlockType.figure,
    "header": BlockType.header,
    "footer": BlockType.footer,
    "footnote": BlockType.footer,
    "number": BlockType.footer,
}


def _paddle_layout(raw: str, ctx: ParseContext) -> NormalizedPage:
    """JSON of one `/layout-parsing` page (`PipelineClient`): `prunedResult.parsing_res_list`
    blocks `{block_label, block_content, block_bbox, block_order}` in pixels of the image
    sent. Table content is HTML (the pipeline converts OTSL). Page text joins every
    non-figure block in list order, headers and footers included, like the other layout
    parsers (the pipeline's own markdown drops them, and statements keep account details in
    headers); the pipeline markdown stays in the raw output."""
    data = json.loads(raw)
    items = (data.get("prunedResult") or {}).get("parsing_res_list")
    if not isinstance(items, list):
        markdown = ((data.get("markdown") or {}).get("text") or "").strip()
        return NormalizedPage(text=markdown, parse_error="no parsing_res_list in pipeline result")
    blocks: list[Block] = []
    tables: list[list[list[str]]] = []
    texts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        block_type = _PADDLE_BLOCK_TYPE.get(str(item.get("block_label", "")), BlockType.text)
        text = item.get("block_content")
        text = text.strip() if isinstance(text, str) else ""
        blocks.append(
            Block(type=block_type, text=text, bbox=_scaled_bbox(item.get("block_bbox"), ctx))
        )
        if block_type == BlockType.table:
            tables.extend(rows for _, rows in html_tables(text))
        if block_type != BlockType.figure and text:
            texts.append(text)
    return NormalizedPage(text="\n\n".join(texts), blocks=blocks, tables=tables)


# --- registry -------------------------------------------------------------------------------

PARSERS: dict[str, Parser] = {
    "markdown": _markdown,
    "html_table": _html_table,
    "json_fields": _json_fields,
    "dots_layout_json": _dots_layout_json,
    "teleocr_layout": _teleocr_layout,
    "otsl": _otsl,
    "typhoon_markdown": _typhoon_markdown,
    "ovis_markdown": _ovis_markdown,
    "paddle_layout": _paddle_layout,
}


def parse(name: str, raw: str, ctx: ParseContext) -> NormalizedPage:
    """Run parser `name`; never raises (failures become `parse_error`)."""
    parser = PARSERS.get(name)
    if parser is None:
        return NormalizedPage(text=raw.strip(), parse_error=f"unknown parser '{name}'")
    try:
        return parser(raw, ctx)
    except Exception as exc:  # a parser bug must not drop the prediction
        return NormalizedPage(
            text=raw.strip(), parse_error=f"parser {name} failed: {type(exc).__name__}: {exc}"
        )
