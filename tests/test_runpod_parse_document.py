"""scripts/runpod_parse_document.py: the standalone dots.ocr client sample."""

import asyncio
import base64
import importlib.util
import io
import json
from pathlib import Path

import httpx
import pytest
import yaml
from PIL import Image

from ocr_bench.models.imaging import smart_resize as harness_smart_resize
from ocr_bench.models.parsers import _load_dots_items
from tests.conftest import _write_digital_pdf

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "runpod_parse_document", ROOT / "scripts" / "runpod_parse_document.py"
)
rpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rpd)  # type: ignore[union-attr]

BASE = "https://abcdefghij12.api.runpod.ai/v1"


# --- stays identical to the harness -----------------------------------------------------------


def test_prompt_prefix_and_sampling_match_the_harness_config():
    cfg = yaml.safe_load((ROOT / "configs" / "models" / "dotsocr.yaml").read_text())
    assert rpd.LAYOUT_PROMPT == cfg["plans"]["ocr_fullpage"]["prompt"]
    assert rpd.MESSAGE_PREFIX == cfg["message_prefix"]
    assert rpd.SERVED_MODEL == cfg["served_model_name"]
    assert rpd.MAX_PIXELS == cfg["max_pixels"]
    assert rpd.MAX_TOKENS == cfg["sampling"]["max_tokens"]


@pytest.mark.parametrize("size", [(1654, 2339), (3000, 4000), (40, 30), (2480, 3508)])
def test_smart_resize_matches_the_harness(size):
    w, h = size
    assert rpd.smart_resize(h, w) == harness_smart_resize(h, w, max_pixels=rpd.MAX_PIXELS)


@pytest.mark.parametrize(
    "raw",
    [
        '[{"category": "Text", "text": "a"}]',
        '```json\n[{"category": "Text", "text": "a"}]\n```',
        '{"layout": [{"category": "Text", "text": "a"}]}',
        '[{"category": "Text", "text": "a"}, {"category": "Text", "text": "b',  # truncated
    ],
)
def test_layout_json_loading_matches_the_harness(raw):
    assert rpd.load_layout(raw)[0] == _load_dots_items(raw)[0]


# --- blocks to Markdown -------------------------------------------------------------------------


def test_blocks_become_markdown_in_reading_order():
    blocks = [
        {"category": "Page-header", "text": "Bank X"},
        {"category": "Title", "text": "# Statement"},
        {"category": "Text", "text": "Account **123**"},
        {"category": "Formula", "text": "a^2 + b^2"},
        {"category": "Picture", "bbox": [0, 0, 1, 1]},
        {
            "category": "Table",
            "text": "<table><tr><th>Date</th><th>Amount</th></tr>"
            "<tr><td>1 Oct</td><td>1,000.00</td></tr></table>",
        },
        {"category": "Page-footer", "text": "Page 1"},
    ]
    md = rpd.blocks_to_markdown(blocks)
    assert md.split("\n\n") == [
        "Bank X",
        "# Statement",
        "Account **123**",
        "$$\na^2 + b^2\n$$",
        "<!-- picture -->",
        "| Date | Amount |\n| --- | --- |\n| 1 Oct | 1,000.00 |",
        "Page 1",
    ]
    no_furniture = rpd.blocks_to_markdown(blocks, keep_page_furniture=False)
    assert "Bank X" not in no_furniture and "Page 1" not in no_furniture


def test_table_with_merged_cells_stays_html():
    html = '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'
    assert rpd.table_to_markdown(html) == html


def test_table_cells_escape_pipes_and_newlines():
    html = "<table><tr><td>a|b</td><td>x<br>y</td></tr></table>"
    assert rpd.table_to_markdown(html) == "| a\\|b | x y |\n| --- | --- |"


def test_formula_already_delimited_is_kept():
    assert rpd.blocks_to_markdown([{"category": "Formula", "text": "$$x$$"}]) == "$$x$$"


# --- end to end against a fake endpoint ---------------------------------------------------------


class _Endpoint:
    """A fake load-balancing endpoint: /health starts at 204, chat returns one layout block."""

    def __init__(self, health=(204, 200)):
        self.health = list(health)
        self.chats: list[dict] = []
        self.auth: set[str | None] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.auth.add(request.headers.get("authorization"))
        if request.url.path.endswith("/health"):
            return httpx.Response(self.health.pop(0) if len(self.health) > 1 else self.health[0])
        body = json.loads(request.content)
        self.chats.append(body)
        n = len(self.chats)
        content = json.dumps([{"bbox": [0, 0, 10, 10], "category": "Text", "text": f"page {n}"}])
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _png(path: Path, size=(1240, 1754)) -> Path:
    Image.new("RGB", size, "white").save(path)
    return path


def _run(endpoint, inputs, **kw):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as http:
            return await rpd.parse_document(
                inputs, BASE, "key-1", http=http, poll_s=0, sleep=_no_sleep, **kw
            )

    return asyncio.run(go())


async def _no_sleep(_s: float) -> None:
    return None


def test_images_are_parsed_after_waiting_for_a_worker(tmp_path):
    ep = _Endpoint()
    md = _run(ep, [_png(tmp_path / "a.png"), _png(tmp_path / "b.png")])
    assert md == "<!-- page 1 -->\n\npage 1\n\n<!-- page 2 -->\n\npage 2\n"
    assert ep.auth == {"Bearer key-1"}
    body = ep.chats[0]
    assert body["model"] == "dotsocr"
    assert body["temperature"] == 0.0
    content = body["messages"][0]["content"]
    assert content[1]["text"] == rpd.MESSAGE_PREFIX + rpd.LAYOUT_PROMPT
    sent = Image.open(io.BytesIO(base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1])))
    h, w = rpd.smart_resize(1754, 1240)
    assert sent.size == (w, h)


def test_sent_image_is_smart_resized(tmp_path):
    ep = _Endpoint(health=(200,))
    _run(ep, [_png(tmp_path / "big.png", size=(3000, 4000))])
    url = ep.chats[0]["messages"][0]["content"][0]["image_url"]["url"]
    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    h, w = rpd.smart_resize(4000, 3000)
    assert sent.size == (w, h)
    assert w * h <= rpd.MAX_PIXELS


def test_pdf_pages_are_rendered_at_200_dpi(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _write_digital_pdf(pdf, n_pages=2)  # 4 x 3 in pages
    ep = _Endpoint(health=(200,))
    md = _run(ep, [pdf])
    assert md.count("<!-- page ") == 2
    url = ep.chats[0]["messages"][0]["content"][0]["image_url"]["url"]
    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    h, w = rpd.smart_resize(600, 800)  # 3 x 4 in at 200 dpi
    assert sent.size == (w, h)


def test_unparseable_output_falls_back_to_raw_text(tmp_path):
    def handler(request):
        if request.url.path.endswith("/health"):
            return httpx.Response(200)
        return httpx.Response(200, json={"choices": [{"message": {"content": "plain words"}}]})

    md = _run(handler, [_png(tmp_path / "a.png")])
    assert "plain words" in md
    assert "<!-- parse error" in md


def test_no_worker_within_the_wait_raises(tmp_path):
    ep = _Endpoint(health=(204,))
    with pytest.raises(rpd.EndpointNotReady):
        _run(ep, [_png(tmp_path / "a.png")], wait_s=0)


def test_gateway_error_is_retried(tmp_path):
    calls = {"n": 0}

    def handler(request):
        if request.url.path.endswith("/health"):
            return httpx.Response(200)
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, text="no workers available")
        content = json.dumps([{"category": "Text", "text": "ok"}])
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    assert "ok" in _run(handler, [_png(tmp_path / "a.png")])
    assert calls["n"] == 2


def test_cli_writes_markdown_and_reads_env(tmp_path, monkeypatch):
    out = tmp_path / "out.md"
    monkeypatch.setenv("DOTSOCR_BASE_URL", BASE)
    monkeypatch.setenv("DOTSOCR_API_KEY", "env-key")
    seen = {}

    async def fake_parse(inputs, base_url, api_key, **kw):
        seen.update(inputs=inputs, base_url=base_url, api_key=api_key, **kw)
        return "# hi\n"

    monkeypatch.setattr(rpd, "parse_document", fake_parse)
    assert rpd.main([str(_png(tmp_path / "a.png")), "-o", str(out), "--drop-page-furniture"]) == 0
    assert out.read_text() == "# hi\n"
    assert seen["base_url"] == BASE and seen["api_key"] == "env-key"
    assert seen["keep_page_furniture"] is False
