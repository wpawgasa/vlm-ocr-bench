#!/usr/bin/env python3
"""Parse a document (PDF or images) to Markdown with the dots.ocr Runpod endpoint.

A standalone client sample: it imports nothing from this repo, so it can be copied into
another project. Requirements: `pip install httpx pillow pypdfium2`.

    export DOTSOCR_BASE_URL=https://<endpoint-id>.api.runpod.ai/v1
    export DOTSOCR_API_KEY=<the endpoint's restricted Runpod key>
    python scripts/runpod_parse_document.py statement.pdf -o statement.md
    python scripts/runpod_parse_document.py page1.png page2.jpg > pages.md

What dots.ocr returns: with the layout prompt below (the one the benchmark scored), the
model answers with a JSON list of layout blocks, `{"bbox", "category", "text"}`, in reading
order. `text` is Markdown for text blocks, HTML for `Table`, LaTeX for `Formula`, and absent
for `Picture`. This script turns that list into one Markdown document: tables without merged
cells become pipe tables (with merged cells they stay HTML, which Markdown renders), formulas
are wrapped in `$$`, pictures become a comment, and each page starts with `<!-- page N -->`.

Pages are rendered and resized exactly as the benchmark harness does (PDF at 200 dpi, then
Qwen2-VL smart_resize to at most 4 MP), so results match what was measured. The endpoint is a
Runpod load balancer: while no worker is ready it holds or drops requests, so the script first
waits on `/health` (up to `--wait-s`), then sends pages with bounded concurrency and retries
gateway errors and timeouts. Images never touch disk.
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
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

# --- the model contract (kept identical to configs/models/dotsocr.yaml by a test) -------------

SERVED_MODEL = "dotsocr"
MESSAGE_PREFIX = "<|img|><|imgpad|><|endofimg|>"
MAX_PIXELS = 4_000_000
MAX_TOKENS = 8192
LAYOUT_PROMPT = """\
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

PDF_DPI = 200
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PAGE_FURNITURE = {"Page-header", "Page-footer"}
RETRY_STATUSES = {429, 500, 502, 503, 504}
STARTING_STATUSES = {204, 502, 503}


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
    height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = MAX_PIXELS
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


def image_data_url(image: Image.Image) -> str:
    w, h = image.size
    new_h, new_w = smart_resize(h, w)
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.BICUBIC)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --- the model's answer to Markdown -----------------------------------------------------------

_FENCE_OPEN = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")


def load_layout(raw: str) -> tuple[list[dict[str, Any]], str | None]:
    """The layout blocks and an error note. A truncated answer (it hit max_tokens) keeps
    every block that decodes completely."""
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
            spans = {k: v for k, v in attrs if k in ("rowspan", "colspan")}
            if any((v or "1").strip() not in ("", "1") for v in spans.values()):
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


def blocks_to_markdown(blocks: list[dict[str, Any]], keep_page_furniture: bool = True) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        category = str(block.get("category", "Text"))
        text = block.get("text")
        text = text.strip() if isinstance(text, str) else ""
        if category in PAGE_FURNITURE and not keep_page_furniture:
            continue
        if category == "Picture":
            parts.append("<!-- picture -->")
        elif category == "Table" and text:
            parts.append(table_to_markdown(text))
        elif category == "Formula" and text:
            parts.append(text if text.startswith("$") else f"$$\n{text}\n$$")
        elif text:
            parts.append(text)
    return "\n\n".join(parts)


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
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Poll /health until 200. 204/502/503 and timeouts mean a worker is still starting (a
    first start pulls the image and loads the model, several minutes)."""
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


async def parse_page(
    http: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    image: Image.Image,
    retries: int = 3,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    body = {
        "model": SERVED_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                    {"type": "text", "text": MESSAGE_PREFIX + LAYOUT_PROMPT},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 0,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    for attempt in range(retries + 1):
        try:
            # Runpod allows 5.5 min of processing per request on a load-balancing endpoint.
            r = await http.post(url, json=body, headers=headers, timeout=330)
        except httpx.TimeoutException:
            if attempt == retries:
                raise
        else:
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"] or ""
            if r.status_code not in RETRY_STATUSES or attempt == retries:
                r.raise_for_status()
        await sleep(min(2**attempt * 5, 30))
    raise AssertionError("unreachable")


async def parse_document(
    inputs: Sequence[Path],
    base_url: str,
    api_key: str,
    *,
    http: httpx.AsyncClient | None = None,
    concurrency: int = 4,
    wait_s: float = 600,
    poll_s: float = 5.0,
    keep_page_furniture: bool = True,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    """The whole input as Markdown, one `<!-- page N -->` section per page."""
    if base_url.startswith("http://") and base_url.split("/")[2].endswith(".api.runpod.ai"):
        raise ValueError("use https:// for Runpod endpoints")
    pages = load_pages(inputs)
    headers = {"Authorization": f"Bearer {api_key}"}
    own = http is None
    http = http or httpx.AsyncClient()
    try:
        await wait_for_worker(http, base_url, headers, wait_s, poll_s, sleep)
        limit = asyncio.Semaphore(concurrency)

        async def one(image: Image.Image) -> str:
            async with limit:
                return await parse_page(http, base_url, headers, image, sleep=sleep)

        answers = await asyncio.gather(*(one(p) for p in pages))
    finally:
        if own:
            await http.aclose()

    sections = []
    for n, raw in enumerate(answers, start=1):
        blocks, error = load_layout(raw)
        body = blocks_to_markdown(blocks, keep_page_furniture) if blocks else raw.strip()
        note = f"<!-- parse error: {error} -->\n\n" if error else ""
        sections.append(f"<!-- page {n} -->\n\n{note}{body}".rstrip())
    return "\n\n".join(sections) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse a PDF or images to Markdown with the dots.ocr Runpod endpoint."
    )
    ap.add_argument("inputs", nargs="+", type=Path, help="PDF and/or image files, in page order")
    ap.add_argument("-o", "--output", type=Path, help="Markdown file (default: stdout)")
    ap.add_argument(
        "--base-url",
        default=os.environ.get("DOTSOCR_BASE_URL"),
        help="https://<endpoint-id>.api.runpod.ai/v1 (default: $DOTSOCR_BASE_URL)",
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("DOTSOCR_API_KEY") or os.environ.get("RUNPOD_API_KEY"),
        help="Runpod key (default: $DOTSOCR_API_KEY, then $RUNPOD_API_KEY)",
    )
    ap.add_argument("--concurrency", type=int, default=4, help="pages in flight (default: 4)")
    ap.add_argument(
        "--wait-s", type=float, default=600, help="how long to wait for a worker (default: 600)"
    )
    ap.add_argument(
        "--drop-page-furniture", action="store_true", help="leave out page headers and footers"
    )
    args = ap.parse_args(argv)
    if not args.base_url or not args.api_key:
        ap.error("set --base-url/--api-key or $DOTSOCR_BASE_URL/$DOTSOCR_API_KEY")
    markdown = asyncio.run(
        parse_document(
            args.inputs,
            args.base_url,
            args.api_key,
            concurrency=args.concurrency,
            wait_s=args.wait_s,
            keep_page_furniture=not args.drop_page_furniture,
        )
    )
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
