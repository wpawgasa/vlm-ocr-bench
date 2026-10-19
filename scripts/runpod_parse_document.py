#!/usr/bin/env python3
"""Parse a document (PDF or images) to Markdown with any of the Runpod OCR endpoints.

A standalone client sample: it imports nothing from this repo, so it can be copied into
another project. Requirements: `pip install httpx pillow pypdfium2` (plus
`opencv-python-headless numpy` for TeleOCR pages with polygon layout blocks).

    export DOTSOCR_BASE_URL=https://<endpoint-id>.api.runpod.ai/v1
    export DOTSOCR_API_KEY=<the endpoint's restricted Runpod key>
    python scripts/runpod_parse_document.py --model dotsocr statement.pdf -o statement.md
    python scripts/runpod_parse_document.py --model typhoon_ocr15 page1.png page2.jpg

Each model is called and answers differently; the script hides that and always writes
Markdown, one `<!-- page N -->` section per page:

- `dotsocr`: one chat request with the layout prompt; answers a JSON list of layout
  blocks (Markdown text, HTML tables, LaTeX formulas).
- `typhoon_ocr15`: one chat request with its one trained prompt; answers Markdown with
  HTML tables, `<figure>` (a Thai description) and `<page_number>` tags.
- `ovisocr2`: one chat request; answers Markdown with HTML tables and `<img>` figure tags.
- `teleocr`: one layout request on a 1036x1036 copy, then one request per block crop;
  answers layout lines, then each block's text (tables as OTSL tokens).
- `paddleocr_vl`: `POST /layout-parsing` on the endpoint root (PaddleX's pipeline); answers
  a JSON envelope of layout blocks.

In the Markdown, tables become pipe tables unless cells are merged (then they stay HTML,
which Markdown renders), formulas are `$$`-wrapped, pictures become `<!-- picture -->`
(typhoon's Thai figure descriptions become a quote), and `--drop-page-furniture` leaves out
page headers, footers and page numbers.

Pages are prepared exactly as the benchmark harness does (PDF at 200 dpi, then each model's
own resize), with the same prompts and sampling, so results match what was measured. The
endpoint is a Runpod load balancer: while no worker is ready it holds or drops requests, so
the script first waits on `/health` (up to `--wait-s`), then sends pages with bounded
concurrency and retries gateway errors and timeouts. Images never touch disk.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

PDF_DPI = 200
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
RETRY_STATUSES = {429, 500, 502, 503, 504}
STARTING_STATUSES = {204, 502, 503}
REQUEST_TIMEOUT_S = 330  # Runpod's per-request processing limit on load balancers is 5.5 min

Sleep = Callable[[float], Awaitable[None]]


class EndpointNotReady(RuntimeError):
    """No worker answered /health with 200 within the wait."""


# --- input pages ------------------------------------------------------------------------------


def load_pages(paths: Sequence[Path]) -> list[Image.Image]:
    """Every page of every input, in order: PDF pages at 200 dpi, image frames as they are."""
    pages: list[Image.Image] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            import pypdfium2  # only needed for PDFs

            pdf = pypdfium2.PdfDocument(str(path))
            try:
                for page in pdf:
                    pages.append(page.render(scale=PDF_DPI / 72).to_pil().convert("RGB"))
            finally:
                pdf.close()
        elif suffix in IMAGE_SUFFIXES:
            with Image.open(path) as img:
                for i in range(getattr(img, "n_frames", 1)):  # multi-page TIFF
                    img.seek(i)
                    pages.append(img.convert("RGB"))
        else:
            raise ValueError(f"{path}: not a PDF or image ({', '.join(sorted(IMAGE_SUFFIXES))})")
    return pages


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 11_289_600
) -> tuple[int, int]:
    """Qwen2-VL smart_resize: (height, width) in multiples of `factor`, within the pixel range."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def smart_resized(
    factor: int, min_pixels: int, max_pixels: int
) -> Callable[[Image.Image], Image.Image]:
    def prepare(image: Image.Image) -> Image.Image:
        w, h = image.size
        new_h, new_w = smart_resize(h, w, factor, min_pixels, max_pixels)
        return image if (new_w, new_h) == (w, h) else image.resize((new_w, new_h), Image.BICUBIC)

    return prepare


def max_side(limit: int) -> Callable[[Image.Image], Image.Image]:
    def prepare(image: Image.Image) -> Image.Image:
        w, h = image.size
        if max(w, h) <= limit:
            return image
        scale = limit / max(w, h)
        return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)

    return prepare


def unchanged(image: Image.Image) -> Image.Image:
    return image


def png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --- shared Markdown helpers ----------------------------------------------------------------


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.merged = False
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            spans = [v for k, v in attrs if k in ("rowspan", "colspan")]
            if any((v or "1").strip() not in ("", "1") for v in spans):
                self.merged = True
            if not self.rows:
                self.rows.append([])
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def table_to_markdown(html: str) -> str:
    """A pipe table, or the HTML unchanged when cells are merged (pipe tables cannot span)."""
    parser = _TableParser()
    parser.feed(html)
    rows = [r for r in parser.rows if r]
    if parser.merged or not rows:
        return html.strip()
    width = max(len(r) for r in rows)

    def line(cells: list[str]) -> str:
        cells = [c.replace("|", "\\|") for c in cells] + [""] * (width - len(cells))
        return "| " + " | ".join(cells) + " |"

    rule = "| " + " | ".join(["---"] * width) + " |"
    return "\n".join([line(rows[0]), rule, *map(line, rows[1:])])


_TABLE_RE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_BLANK_LINES_RE = re.compile(r"\n{3,}")
PICTURE = "<!-- picture -->"


def tables_to_markdown(text: str) -> str:
    return _TABLE_RE.sub(lambda m: table_to_markdown(m.group(0)), text)


def tidy(text: str) -> str:
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def display_math(text: str) -> str:
    return text if text.startswith("$") else f"$$\n{text}\n$$"


# --- dots.ocr: JSON layout blocks -----------------------------------------------------------

DOTS_PROMPT = """\
Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""  # noqa: E501
_FENCE_OPEN = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")


def load_layout(raw: str) -> tuple[list[Any], str | None]:
    """dots.ocr layout blocks and an error note. A truncated answer (it hit max_tokens)
    keeps every block that decodes completely."""
    text = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", raw.strip(), count=1))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        items = _salvage(text)
        if items:
            return items, f"truncated JSON: kept {len(items)} complete blocks"
        return [], "invalid layout JSON"
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if "bbox" not in data and lists else [data]
    if not isinstance(data, list):
        return [], "layout JSON is not a list"
    return data, None


def _salvage(text: str) -> list[Any]:
    start = text.find("[")
    if start < 0:
        return []
    decoder, items, pos = json.JSONDecoder(), [], start + 1
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


def dots_markdown(raw: str, keep_furniture: bool) -> tuple[str, str | None]:
    blocks, error = load_layout(raw)
    if not blocks:
        return raw.strip(), error or "no layout blocks"
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        category = str(block.get("category", "Text"))
        text = block.get("text")
        text = text.strip() if isinstance(text, str) else ""
        if category in ("Page-header", "Page-footer") and not keep_furniture:
            continue
        if category == "Picture":
            parts.append(PICTURE)
        elif category == "Table" and text:
            parts.append(table_to_markdown(text))
        elif category == "Formula" and text:
            parts.append(display_math(text))
        elif text:
            parts.append(text)
    return "\n\n".join(parts), error


# --- typhoon-ocr1.5: Markdown with figure / page-number tags --------------------------------

TYPHOON_PROMPT = (
    "Extract all text from the image.\n"
    "\n"
    "Instructions:\n"
    "- Only return the clean Markdown.\n"
    "- Do not include any explanation or extra text.\n"
    "- You must include all information on the page.\n"
    "\n"
    "Formatting Rules:\n"
    "- Tables: Render tables using <table>...</table> in clean HTML format.\n"
    "- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).\n"  # noqa: E501
    "- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:\n"  # noqa: E501
    "\n"
    "<figure>\n"
    "Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.\n"  # noqa: E501
    "</figure>\n"
    "\n"
    "- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).\n"  # noqa: E501
    "- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."
)
_FIGURE_RE = re.compile(r"<figure>(.*?)(?:</figure>|$)", re.DOTALL | re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(r"<page_number>(.*?)</page_number>", re.DOTALL | re.IGNORECASE)


def _figure_quote(match: re.Match[str]) -> str:
    lines = [ln.strip() for ln in match.group(1).strip().splitlines() if ln.strip()]
    if not lines:
        return PICTURE
    return "\n".join([f"> **Figure:** {lines[0]}", *(f"> {ln}" for ln in lines[1:])])


def typhoon_markdown(raw: str, keep_furniture: bool) -> tuple[str, str | None]:
    text = _FIGURE_RE.sub(_figure_quote, raw)
    text = _PAGE_NUMBER_RE.sub(lambda m: m.group(1).strip() if keep_furniture else "", text)
    return tidy(tables_to_markdown(text)), None


# --- OvisOCR2: Markdown with <img> figure tags ----------------------------------------------

OVIS_PROMPT = (
    "\n"
    'Extract all readable content from the image in natural human reading order and output the result as a single Markdown document. For charts or images, represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing.'  # noqa: E501
)


def clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """OvisOCR2's official post-processing (model card, Apache-2.0): cut a degenerate
    repeated tail of a long output down to one period."""
    n = len(text)
    if n < min_text_len:
        return text
    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue
        match_len, idx = 1, n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1
        total_len = match_len + unit_len
        repeat_times, tail_len = total_len // unit_len, total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]
    return text


def ovis_markdown(raw: str, keep_furniture: bool) -> tuple[str, str | None]:
    paragraphs = [
        PICTURE if p.strip().startswith('<img src="images/bbox_') else p
        for p in raw.strip().split("\n\n")
    ]
    return tidy(tables_to_markdown(clean_truncated_repeats("\n\n".join(paragraphs)))), None


# --- TeleOCR: layout request, then one request per block crop --------------------------------

TELEOCR_SYSTEM = "You are a helpful assistant."
TELEOCR_LAYOUT_PROMPT = "\nAnalyze the image layout."
TELEOCR_LAYOUT_SIZE = (1036, 1036)
TELEOCR_LAYOUT_SAMPLING: dict[str, Any] = {
    "max_tokens": 4096,
    "temperature": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 0,
    "top_p": 0.01,
    "top_k": 1,
    "repetition_penalty": 1.0,
    "vllm_xargs": {"no_repeat_ngram_size": 100},
}
_TELEOCR_TEXT_SAMPLING = {
    **TELEOCR_LAYOUT_SAMPLING,
    "presence_penalty": 1.0,
    "frequency_penalty": 0.05,
}
_TELEOCR_TABLE_SAMPLING = {**_TELEOCR_TEXT_SAMPLING, "frequency_penalty": 0.005}
TELEOCR_BLOCK_SAMPLING = {
    "text": _TELEOCR_TEXT_SAMPLING,
    "equation": _TELEOCR_TEXT_SAMPLING,
    "code": _TELEOCR_TEXT_SAMPLING,
    "default": _TELEOCR_TEXT_SAMPLING,
    "table": _TELEOCR_TABLE_SAMPLING,
    "char": _TELEOCR_TABLE_SAMPLING,
    "seal": _TELEOCR_TABLE_SAMPLING,
}
TELEOCR_BLOCK_PROMPTS = {
    "text": "\nPlease output the text content from the image.",
    "table": "\nThis is the image of a table. Please output the table in OTSL format.",
    "equation": "\nPlease write out the expression of the formula in the image using LaTeX format.",
    "code": "\nThe image contains a code snippet, please output the parsing result.",
    "seal": "\nSeal Recognition:",
    "char": "\nThis is a scientific figure. Please extract the table implied by this figure.",
    "default": "\nPlease output the text content from the image.",
}
TELEOCR_SKIP = ("image", "list", "equation_block")  # not sent; images stay as a picture
TELEOCR_MIN_EDGE, TELEOCR_MAX_RATIO = 28, 50
TELEOCR_MAX_PIXELS = 64_000_000
TELEOCR_BLOCK_CONCURRENCY = 8
TELEOCR_LABELS = frozenset(
    "text title table image code algorithm header footer page_number page_footnote aside_text"
    " equation equation_block ref_text list phonetic table_caption image_caption code_caption"
    " table_footnote image_footnote unknown seal char".split()
)
_TELEOCR_FURNITURE = {"header", "footer", "page_number", "page_footnote"}
_TELEOCR_LINE_RE = re.compile(r"^<box:([\d\s]+)><label:(\w+)><([^>]+)>$")
_TELEOCR_ANGLES = (("up", 0), ("right", 90), ("down", 180), ("left", 270))


@dataclass
class LayoutItem:
    index: int
    label: str
    points: list[tuple[int, int]]  # 0-1000 units of the page
    angle: int | None


def parse_teleocr_layout(raw: str) -> list[LayoutItem]:
    """TeleOCR layout lines `<box:x y ...><label:type><angle>`; malformed lines are skipped."""
    items: list[LayoutItem] = []
    for line in raw.splitlines():
        match = _TELEOCR_LINE_RE.match(line.strip())
        if match is None:
            continue
        numbers = [int(n) for n in match.group(1).split()]
        label = match.group(2).lower()
        if len(numbers) < 4 or len(numbers) % 2 or any(n < 0 or n > 1000 for n in numbers):
            continue
        if label not in TELEOCR_LABELS:
            continue
        angle = next((deg for word, deg in _TELEOCR_ANGLES if word in match.group(3)), None)
        points = list(zip(numbers[0::2], numbers[1::2], strict=True))
        items.append(LayoutItem(index=len(items), label=label, points=points, angle=angle))
    return items


def _region_pixels(region: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, ...]:
    w, h = size
    x1, y1, x2, y2 = region
    px1 = min(max(round(min(x1, x2) * w / 1000), 0), w - 1)
    py1 = min(max(round(min(y1, y2) * h / 1000), 0), h - 1)
    px2 = min(max(round(max(x1, x2) * w / 1000), px1 + 1), w)
    py2 = min(max(round(max(y1, y2) * h / 1000), py1 + 1), h)
    return px1, py1, px2, py2


def crop_block(image: Image.Image, points: list[tuple[int, int]]) -> Image.Image:
    """Two points are a rectangle; more are a polygon (bounding box, outside pixels black),
    exactly as TeleOCR's reference client crops."""
    if len(points) == 2:
        (ax, ay), (bx, by) = points
        return image.crop(_region_pixels((ax, ay, bx, by), image.size))
    import cv2  # polygons only: pip install opencv-python-headless numpy
    import numpy as np

    w, h = image.size
    poly = np.array([[round(x * w / 1000), round(y * h / 1000)] for x, y in points], np.int32)
    x, y, bw, bh = cv2.boundingRect(poly)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
    if x1 <= x0 or y1 <= y0:
        return image.crop((0, 0, 1, 1))
    arr = np.array(image.convert("RGB"))[y0:y1, x0:x1].copy()
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly - np.array([x0, y0], dtype=np.int32)], 255)
    arr[mask == 0] = 0
    return Image.fromarray(arr)


def resize_by_need(
    image: Image.Image, min_edge: int = TELEOCR_MIN_EDGE, max_ratio: float = TELEOCR_MAX_RATIO
) -> Image.Image:
    """Pad very elongated crops with white to `max_ratio`, then upscale a short edge to
    `min_edge` (TeleOCR's `resize_by_need`)."""
    w, h = image.size
    if min(w, h) > 0 and max(w, h) / min(w, h) > max_ratio:
        new_w, new_h = (w, math.ceil(w / max_ratio)) if w > h else (math.ceil(h / max_ratio), h)
        canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        canvas.paste(image.convert("RGB"), ((new_w - w) // 2, (new_h - h) // 2))
        image, (w, h) = canvas, (new_w, new_h)
    if min(w, h) < min_edge:
        scale = min_edge / min(w, h)
        image = image.resize((math.ceil(w * scale), math.ceil(h * scale)), Image.BICUBIC)
    return image


def downscale_if_huge(image: Image.Image, max_pixels: int = TELEOCR_MAX_PIXELS) -> Image.Image:
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


_OTSL_TAGS = ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>")
_OTSL_TOKEN_RE = re.compile(r"(<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)")


def otsl_to_html(otsl: str) -> str:
    """TeleOCR's OTSL table tokens to an HTML table (port of its `convert_otsl_to_html`)."""
    stripped = otsl.strip()
    if stripped.startswith("<table") and stripped.endswith("</table>"):
        return stripped
    stripped = stripped.replace("<otsl>", "").replace("</otsl>", "")
    grid: list[list[tuple[str, str]]] = []
    for line in stripped.split("<nl>"):
        parts = _OTSL_TOKEN_RE.split(line)
        row = [
            (parts[i], parts[i + 1] if i + 1 < len(parts) else "") for i in range(1, len(parts), 2)
        ]
        if row:
            grid.append(row)

    def tag_at(r: int, c: int) -> str | None:
        return grid[r][c][0] if 0 <= r < len(grid) and 0 <= c < len(grid[r]) else None

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
            attrs = (f' colspan="{colspan}"' if colspan > 1 else "") + (
                f' rowspan="{rowspan}"' if rowspan > 1 else ""
            )
            text = escape(content.strip(), quote=False) if tag == "<fcel>" else ""
            out.append(f"<td{attrs}>{text}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def fix_equation(text: str) -> str:
    """Strip a `\\[ \\]` wrapper, then `$$`-wrap (TeleOCR's equation post-processing)."""
    s = text.strip()
    if s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s
    return f"$${s}$$"


def teleocr_markdown(items: list[LayoutItem], texts: dict[int, str], keep_furniture: bool) -> str:
    parts: list[str] = []
    for item in items:
        if item.label in ("image", "char"):
            parts.append(PICTURE)
            continue
        if item.label in TELEOCR_SKIP or (item.label in _TELEOCR_FURNITURE and not keep_furniture):
            continue
        text = texts.get(item.index, "").strip()
        if not text:
            continue
        if item.label == "table":
            html = otsl_to_html(text) if any(t in text for t in _OTSL_TAGS) else text
            parts.append(table_to_markdown(html) if html.startswith("<table") else html)
        elif item.label == "equation":
            parts.append(fix_equation(text))
        elif item.label == "title":
            parts.append(text if text.startswith("#") else f"# {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts)


# --- PaddleOCR-VL: the PaddleX /layout-parsing pipeline --------------------------------------

PADDLE_PATH = "/layout-parsing"
PADDLE_REQUEST = {"visualize": False, "returnMarkdownImages": False}
PADDLE_TIMEOUT_S = 300
_PADDLE_PICTURES = {"image", "chart", "seal", "header_image", "footer_image"}
_PADDLE_FURNITURE = {"header", "footer", "footnote", "number"}
_PADDLE_HEADINGS = {"doc_title": "# ", "paragraph_title": "## "}


def paddle_markdown(page: dict[str, Any], keep_furniture: bool) -> str:
    """One page of the pipeline's result: its layout blocks in reading order, or the
    pipeline's own Markdown when the blocks are missing."""
    items = (page.get("prunedResult") or {}).get("parsing_res_list")
    if not isinstance(items, list):
        return ((page.get("markdown") or {}).get("text") or "").strip()
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("block_label", ""))
        text = item.get("block_content")
        text = text.strip() if isinstance(text, str) else ""
        if label in _PADDLE_PICTURES:
            parts.append(PICTURE)
        elif label in _PADDLE_FURNITURE and not keep_furniture:
            continue
        elif not text:
            continue
        elif label == "table":
            parts.append(tables_to_markdown(text))
        elif "formula" in label:
            parts.append(display_math(text))
        elif label in _PADDLE_HEADINGS and not text.startswith("#"):
            parts.append(_PADDLE_HEADINGS[label] + text)
        else:
            parts.append(text)
    return "\n\n".join(parts)


# --- the models ---------------------------------------------------------------------------------


def _chat_sampling(max_tokens: int) -> dict[str, Any]:
    return {
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": 0,
    }


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # chat | teleocr | paddle
    base_url_env: str
    served_model: str
    prepare: Callable[[Image.Image], Image.Image] = unchanged
    prompt: str = ""
    message_prefix: str = ""
    system_prompt: str | None = None
    sampling: dict[str, Any] = field(default_factory=dict)
    to_markdown: Callable[[str, bool], tuple[str, str | None]] | None = None

    @property
    def api_key_env(self) -> str:
        return self.base_url_env.removesuffix("_BASE_URL") + "_API_KEY"


MODELS: dict[str, ModelSpec] = {
    "dotsocr": ModelSpec(
        "dotsocr",
        "chat",
        "DOTSOCR_BASE_URL",
        "dotsocr",
        prepare=smart_resized(28, 3136, 4_000_000),
        prompt=DOTS_PROMPT,
        message_prefix="<|img|><|imgpad|><|endofimg|>",
        sampling=_chat_sampling(8192),
        to_markdown=dots_markdown,
    ),
    "typhoon_ocr15": ModelSpec(
        "typhoon_ocr15",
        "chat",
        "TYPHOON_BASE_URL",
        "typhoon",
        prepare=max_side(1800),
        prompt=TYPHOON_PROMPT,
        sampling=_chat_sampling(10000),
        to_markdown=typhoon_markdown,
    ),
    "ovisocr2": ModelSpec(
        "ovisocr2",
        "chat",
        "OVISOCR2_BASE_URL",
        "ovisocr2",
        prepare=smart_resized(32, 200_704, 8_294_400),
        prompt=OVIS_PROMPT,
        sampling=_chat_sampling(16384),
        to_markdown=ovis_markdown,
    ),
    "teleocr": ModelSpec(
        "teleocr", "teleocr", "TELEOCR_BASE_URL", "teleocr", system_prompt=TELEOCR_SYSTEM
    ),
    "paddleocr_vl": ModelSpec(
        "paddleocr_vl", "paddle", "PADDLEOCR_VL_BASE_URL", "PaddleOCR-VL-1.6-0.9B"
    ),
}


# --- the endpoint -------------------------------------------------------------------------------


def _root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


async def wait_for_worker(
    http: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    wait_s: float,
    poll_s: float = 5.0,
    sleep: Sleep = asyncio.sleep,
) -> None:
    """Poll /health until 200. 204/502/503 and timeouts mean a worker is still starting (a
    first start pulls the image and loads the model: several minutes)."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            r = await http.get(_root(base_url) + "/health", headers=headers, timeout=30)
            if r.status_code == 200:
                return
            if r.status_code not in STARTING_STATUSES:
                raise EndpointNotReady(f"/health answered {r.status_code}: {r.text[:200]}")
            last = f"/health answered {r.status_code}"
        except httpx.TimeoutException:
            last = "/health timed out"
        if time.monotonic() >= deadline:
            raise EndpointNotReady(f"no worker ready after {wait_s:.0f}s ({last})")
        await sleep(poll_s)


async def _post(
    http: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    sleep: Sleep,
    retries: int = 3,
) -> Any:
    """POST with retries on gateway errors, rate limits and timeouts; returns the JSON."""
    for attempt in range(retries + 1):
        try:
            r = await http.post(url, json=body, headers=headers, timeout=timeout)
        except httpx.TimeoutException:
            if attempt == retries:
                raise
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code not in RETRY_STATUSES or attempt == retries:
                r.raise_for_status()
        await sleep(min(2**attempt * 5, 30))
    raise AssertionError("unreachable")


async def chat(
    http: httpx.AsyncClient,
    spec: ModelSpec,
    base_url: str,
    headers: dict[str, str],
    image: Image.Image,
    prompt: str,
    sampling: dict[str, Any],
    sleep: Sleep,
) -> str:
    messages: list[dict[str, Any]] = []
    if spec.system_prompt is not None:
        messages.append({"role": "system", "content": spec.system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + png_b64(image)},
                },
                {"type": "text", "text": spec.message_prefix + prompt},
            ],
        }
    )
    body = {"model": spec.served_model, "messages": messages, **sampling}
    url = base_url.rstrip("/") + "/chat/completions"
    data = await _post(http, url, body, headers, REQUEST_TIMEOUT_S, sleep)
    return data["choices"][0]["message"]["content"] or ""


async def page_markdown(
    http: httpx.AsyncClient,
    spec: ModelSpec,
    base_url: str,
    headers: dict[str, str],
    page: Image.Image,
    keep_furniture: bool,
    sleep: Sleep,
) -> tuple[str, str | None]:
    """One page as Markdown, plus a note when the answer could not be fully parsed."""
    if spec.kind == "chat":
        raw = await chat(
            http, spec, base_url, headers, spec.prepare(page), spec.prompt, spec.sampling, sleep
        )
        assert spec.to_markdown is not None
        return spec.to_markdown(raw, keep_furniture)

    if spec.kind == "paddle":
        body = {**PADDLE_REQUEST, "file": png_b64(page), "fileType": 1}
        url = _root(base_url) + PADDLE_PATH
        envelope = await _post(http, url, body, headers, PADDLE_TIMEOUT_S, sleep)
        if envelope.get("errorCode", 0) != 0:
            return "", f"pipeline error {envelope.get('errorCode')}: {envelope.get('errorMsg')}"
        pages = (envelope.get("result") or {}).get("layoutParsingResults") or []
        if not pages:
            return "", "pipeline response has no layoutParsingResults"
        return paddle_markdown(pages[0], keep_furniture), None

    # TeleOCR: layout on a fixed-size copy, then each block cropped from the full page.
    original = downscale_if_huge(page)
    layout_image = original.resize(TELEOCR_LAYOUT_SIZE, Image.BICUBIC)
    layout = await chat(
        http,
        spec,
        base_url,
        headers,
        layout_image,
        TELEOCR_LAYOUT_PROMPT,
        TELEOCR_LAYOUT_SAMPLING,
        sleep,
    )
    items = parse_teleocr_layout(layout)
    if not items:
        return layout.strip(), "no valid layout blocks"
    limit = asyncio.Semaphore(TELEOCR_BLOCK_CONCURRENCY)

    async def block(item: LayoutItem) -> str:
        crop = crop_block(original, item.points)
        if item.angle in (90, 180, 270):
            crop = crop.rotate(item.angle, expand=True)
        crop = resize_by_need(crop)
        prompt = TELEOCR_BLOCK_PROMPTS.get(item.label, TELEOCR_BLOCK_PROMPTS["default"])
        sampling = TELEOCR_BLOCK_SAMPLING.get(item.label, TELEOCR_BLOCK_SAMPLING["default"])
        async with limit:
            return await chat(http, spec, base_url, headers, crop, prompt, sampling, sleep)

    targets = [i for i in items if i.label not in TELEOCR_SKIP]
    answers = await asyncio.gather(*(block(i) for i in targets))
    texts = {item.index: text for item, text in zip(targets, answers, strict=True)}
    return teleocr_markdown(items, texts, keep_furniture), None


async def parse_document(
    model: str,
    inputs: Sequence[Path],
    base_url: str,
    api_key: str,
    *,
    http: httpx.AsyncClient | None = None,
    concurrency: int = 4,
    wait_s: float = 600,
    poll_s: float = 5.0,
    keep_furniture: bool = True,
    sleep: Sleep = asyncio.sleep,
) -> str:
    """The whole input as Markdown, one `<!-- page N -->` section per page."""
    spec = MODELS[model]
    if base_url.startswith("http://") and base_url.split("/")[2].endswith(".api.runpod.ai"):
        raise ValueError("use https:// for Runpod endpoints")
    pages = load_pages(inputs)
    headers = {"Authorization": f"Bearer {api_key}"}
    own = http is None
    http = http or httpx.AsyncClient()
    try:
        await wait_for_worker(http, base_url, headers, wait_s, poll_s, sleep)
        limit = asyncio.Semaphore(concurrency)

        async def one(page: Image.Image) -> tuple[str, str | None]:
            async with limit:
                return await page_markdown(
                    http, spec, base_url, headers, page, keep_furniture, sleep
                )

        results = await asyncio.gather(*(one(p) for p in pages))
    finally:
        if own:
            await http.aclose()

    sections = []
    for n, (body, error) in enumerate(results, start=1):
        note = f"<!-- parse error: {error} -->\n\n" if error else ""
        sections.append(f"<!-- page {n} -->\n\n{note}{body}".rstrip())
    return "\n\n".join(sections) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse a PDF or images to Markdown with a Runpod OCR endpoint."
    )
    ap.add_argument("--model", required=True, choices=sorted(MODELS), help="which endpoint")
    ap.add_argument("inputs", nargs="+", type=Path, help="PDF and/or image files, in page order")
    ap.add_argument("-o", "--output", type=Path, help="Markdown file (default: stdout)")
    ap.add_argument(
        "--base-url", help="https://<endpoint-id>.api.runpod.ai/v1 (default: $<MODEL>_BASE_URL)"
    )
    ap.add_argument(
        "--api-key", help="Runpod key (default: $<MODEL>_API_KEY, then $RUNPOD_API_KEY)"
    )
    ap.add_argument("--concurrency", type=int, default=4, help="pages in flight (default: 4)")
    ap.add_argument(
        "--wait-s", type=float, default=600, help="how long to wait for a worker (default: 600)"
    )
    ap.add_argument(
        "--drop-page-furniture",
        action="store_true",
        help="leave out page headers, footers and page numbers",
    )
    args = ap.parse_args(argv)
    spec = MODELS[args.model]
    base_url = args.base_url or os.environ.get(spec.base_url_env)
    api_key = args.api_key or os.environ.get(spec.api_key_env) or os.environ.get("RUNPOD_API_KEY")
    if not base_url or not api_key:
        ap.error(
            f"set --base-url/--api-key or ${spec.base_url_env} and "
            f"${spec.api_key_env} (or $RUNPOD_API_KEY)"
        )
    markdown = asyncio.run(
        parse_document(
            args.model,
            args.inputs,
            base_url,
            api_key,
            concurrency=args.concurrency,
            wait_s=args.wait_s,
            keep_furniture=not args.drop_page_furniture,
        )
    )
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
