"""clients/runpod-ocr-client/runpod_parse_document.py: the standalone client for every Runpod model.

The script imports nothing from this repo, so these tests pin it to the harness: the same
prompts, sampling, image preparation, crops and output parsing as `ocr_bench.models`.
"""

import asyncio
import base64
import importlib.util
import io
import json
import sys
from pathlib import Path

import httpx
import pytest
from PIL import Image

from ocr_bench.config import load_model_config
from ocr_bench.models import imaging, parsers
from ocr_bench.models.vllm_client import request_params
from tests.conftest import _write_digital_pdf

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "runpod_parse_document", ROOT / "clients" / "runpod-ocr-client" / "runpod_parse_document.py"
)
rpd = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rpd  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(rpd)  # type: ignore[union-attr]

MODELS = ["dotsocr", "typhoon_ocr15", "ovisocr2", "teleocr", "paddleocr_vl"]
CHAT_MODELS = ["dotsocr", "typhoon_ocr15", "ovisocr2"]
BASE = "https://abcdefghij12.api.runpod.ai/v1"


def _cfg(model):
    return load_model_config(ROOT / "configs" / "models" / f"{model}.yaml", model)


def _harness_body(sampling) -> dict:
    """The JSON body fields the harness's OpenAI call sends for `sampling` (minus logprobs)."""
    params = request_params(sampling)
    extra = params.pop("extra_body", {})
    for key in ("logprobs", "top_logprobs"):
        params.pop(key)
    return {**params, **extra}


def _decode(url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


# --- every model's contract matches the harness config --------------------------------------


def test_every_runpod_model_has_a_spec():
    shipped = {p.stem for p in (ROOT / "configs" / "runpod").glob("*.yaml")}
    assert set(MODELS) <= shipped
    assert set(rpd.MODELS) == set(MODELS)


@pytest.mark.parametrize("model", MODELS)
def test_spec_names_env_and_served_model_match(model):
    cfg, spec = _cfg(model), rpd.MODELS[model]
    assert spec.base_url_env == cfg.endpoint_env
    assert spec.served_model == cfg.served_model_name


@pytest.mark.parametrize("model", CHAT_MODELS)
def test_chat_prompt_prefix_and_sampling_match(model):
    cfg, spec = _cfg(model), rpd.MODELS[model]
    plan = cfg.plans["ocr_fullpage"]
    assert plan.kind == "single"
    assert spec.prompt == plan.prompt
    assert spec.message_prefix == cfg.message_prefix
    assert spec.system_prompt == cfg.system_prompt
    assert spec.sampling == _harness_body(plan.sampling or cfg.sampling)


def test_teleocr_two_stage_matches():
    cfg, spec = _cfg("teleocr"), rpd.MODELS["teleocr"]
    ts = cfg.plans["ocr_fullpage"].two_stage
    assert spec.system_prompt == cfg.system_prompt
    assert rpd.TELEOCR_LAYOUT_PROMPT == ts.layout_prompt
    assert rpd.TELEOCR_LAYOUT_SIZE == tuple(ts.layout_size)
    assert rpd.TELEOCR_LAYOUT_SAMPLING == _harness_body(ts.layout_sampling)
    assert rpd.TELEOCR_BLOCK_PROMPTS == ts.block_prompts
    assert rpd.TELEOCR_BLOCK_SAMPLING == {
        label: _harness_body(s) for label, s in ts.block_sampling.items()
    }
    assert set(rpd.TELEOCR_SKIP) == set(ts.skip_types)
    assert (rpd.TELEOCR_MIN_EDGE, rpd.TELEOCR_MAX_RATIO) == (ts.min_edge, ts.max_edge_ratio)
    assert rpd.TELEOCR_MAX_PIXELS == ts.max_pixels
    assert rpd.TELEOCR_LABELS == parsers.TELEOCR_LABELS


def test_paddle_pipeline_request_matches():
    cfg = _cfg("paddleocr_vl")
    assert cfg.plans["ocr_fullpage"].kind == "pipeline"
    assert rpd.PADDLE_PATH == cfg.pipeline.path
    assert rpd.PADDLE_REQUEST == cfg.pipeline.request


# --- image preparation ------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("size", [(1654, 2339), (3000, 4000), (40, 30), (2480, 3508)])
def test_page_preparation_matches_the_harness(model, size):
    cfg = _cfg(model)
    image = Image.new("RGB", size, "white")
    if cfg.image_resize == "smart_resize":
        h, w = imaging.smart_resize(
            size[1], size[0], cfg.resize_factor, cfg.min_pixels or 3136, cfg.max_pixels
        )
        expected = (w, h)
    elif cfg.image_resize == "max_side":
        expected = imaging.resize_max_side(image, cfg.max_side).size
    else:
        expected = size
    assert rpd.MODELS[model].prepare(image).size == expected


@pytest.mark.parametrize(
    "points",
    [
        [(100, 100), (500, 300)],
        [(100, 100), (500, 120), (480, 400), (90, 380)],  # polygon
        [(0, 0), (1000, 1000)],
        [(990, 990), (1000, 1000)],
    ],
)
def test_teleocr_crop_matches_the_harness(points):
    image = Image.new("RGB", (1240, 1754))
    for x in range(0, 1240, 40):
        image.paste((x % 255, 80, 160), (x, 0, x + 20, 1754))
    ours = rpd.crop_block(image, points)
    theirs = imaging.crop_block(image, points)
    assert ours.size == theirs.size
    assert ours.tobytes() == theirs.tobytes()


@pytest.mark.parametrize("size", [(10, 600), (600, 10), (5, 5), (300, 200)])
def test_teleocr_resize_by_need_matches(size):
    image = Image.new("RGB", size, "gray")
    assert rpd.resize_by_need(image).size == imaging.resize_by_need(image).size


# --- output parsing -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '[{"category": "Text", "text": "a"}]',
        '```json\n[{"category": "Text", "text": "a"}]\n```',
        '{"layout": [{"category": "Text", "text": "a"}]}',
        '[{"category": "Text", "text": "a"}, {"category": "Text", "text": "b',  # truncated
    ],
)
def test_dots_layout_json_matches(raw):
    assert rpd.load_layout(raw)[0] == parsers._load_dots_items(raw)[0]


def test_teleocr_layout_lines_match():
    raw = (
        "<box:10 20 300 80><label:title><up>\n"
        "<box:10 100 990 600><label:table><up>\n"
        "<box:10 100 500 120 480 400 90 380><label:text><right>\n"
        "garbage line\n"
        "<box:1 2 3><label:text><up>\n"
        "<box:10 700 990 950><label:image><up>\n"
    )
    ours = rpd.parse_teleocr_layout(raw)
    theirs, _ = parsers.parse_teleocr_layout(raw)
    assert [(i.label, i.points, i.angle) for i in ours] == [
        (i.label, i.points, i.angle) for i in theirs
    ]


@pytest.mark.parametrize(
    "otsl",
    [
        "<fcel>A<fcel>B<nl><fcel>1<fcel>2<nl>",
        "<otsl><fcel>H<lcel><nl><fcel>a<fcel>b<nl></otsl>",
        "<fcel>x<ecel><nl><ucel><fcel>y<nl>",
        "<table><tr><td>ok</td></tr></table>",
    ],
)
def test_otsl_to_html_matches(otsl):
    assert rpd.otsl_to_html(otsl) == parsers.otsl_to_html(otsl)


@pytest.mark.parametrize("text", ["x^2", "\\[ x^2 \\]", "$$x$$", "$x$"])
def test_fix_equation_matches(text):
    assert rpd.fix_equation(text) == parsers.fix_equation(text)


def test_ovis_repeat_cleanup_matches():
    text = "intro " * 1500 + "ab" * 400
    assert rpd.clean_truncated_repeats(text) == parsers.clean_truncated_repeats(text)


# --- to Markdown ----------------------------------------------------------------------------

TABLE_HTML = (
    "<table><tr><th>Date</th><th>Amount</th></tr><tr><td>1 Oct</td><td>1,000.00</td></tr></table>"
)
TABLE_MD = "| Date | Amount |\n| --- | --- |\n| 1 Oct | 1,000.00 |"


def test_dots_blocks_to_markdown():
    blocks = [
        {"category": "Page-header", "text": "Bank X"},
        {"category": "Title", "text": "# Statement"},
        {"category": "Formula", "text": "a^2"},
        {"category": "Picture", "bbox": [0, 0, 1, 1]},
        {"category": "Table", "text": TABLE_HTML},
        {"category": "Page-footer", "text": "Page 1"},
    ]
    md, error = rpd.dots_markdown(json.dumps(blocks), keep_furniture=True)
    assert error is None
    assert md.split("\n\n") == [
        "Bank X",
        "# Statement",
        "$$\na^2\n$$",
        "<!-- picture -->",
        TABLE_MD,
        "Page 1",
    ]
    assert "Bank X" not in rpd.dots_markdown(json.dumps(blocks), keep_furniture=False)[0]


def test_typhoon_markdown():
    raw = (
        f"# Statement\n\n{TABLE_HTML}\n\n<figure>\nรูปโลโก้ธนาคาร\n</figure>\n\n\n\n"
        "<page_number>3</page_number>"
    )
    md, _ = rpd.typhoon_markdown(raw, keep_furniture=True)
    assert md == f"# Statement\n\n{TABLE_MD}\n\n> **Figure:** รูปโลโก้ธนาคาร\n\n3"
    assert rpd.typhoon_markdown(raw, keep_furniture=False)[0].endswith("รูปโลโก้ธนาคาร")


def test_ovis_markdown():
    raw = f'# T\n\n<img src="images/bbox_1_2_3_4.jpg" />\n\n{TABLE_HTML}'
    md, _ = rpd.ovis_markdown(raw, keep_furniture=True)
    assert md == f"# T\n\n<!-- picture -->\n\n{TABLE_MD}"


def test_teleocr_markdown():
    items = rpd.parse_teleocr_layout(
        "<box:0 0 1000 50><label:header><up>\n"
        "<box:0 60 1000 100><label:title><up>\n"
        "<box:0 110 1000 400><label:table><up>\n"
        "<box:0 410 1000 500><label:equation><up>\n"
        "<box:0 510 1000 700><label:image><up>\n"
        "<box:0 710 1000 900><label:list><up>\n"
    )
    texts = {
        0: "Bank X",
        1: "Statement",
        2: "<fcel>Date<fcel>Amount<nl><fcel>1 Oct<fcel>1,000.00<nl>",
        3: "x^2",
    }
    md = rpd.teleocr_markdown(items, texts, keep_furniture=True)
    assert md.split("\n\n") == ["Bank X", "# Statement", TABLE_MD, "$$x^2$$", "<!-- picture -->"]
    assert not rpd.teleocr_markdown(items, texts, keep_furniture=False).startswith("Bank X")


def test_paddle_markdown():
    page = {
        "prunedResult": {
            "parsing_res_list": [
                {"block_label": "header", "block_content": "Bank X"},
                {"block_label": "doc_title", "block_content": "Statement"},
                {"block_label": "paragraph_title", "block_content": "Summary"},
                {"block_label": "table", "block_content": TABLE_HTML},
                {"block_label": "display_formula", "block_content": "x^2"},
                {"block_label": "image", "block_content": ""},
                {"block_label": "number", "block_content": "1"},
            ]
        },
        "markdown": {"text": "pipeline markdown"},
    }
    md = rpd.paddle_markdown(page, keep_furniture=True)
    assert md.split("\n\n") == [
        "Bank X",
        "# Statement",
        "## Summary",
        TABLE_MD,
        "$$\nx^2\n$$",
        "<!-- picture -->",
        "1",
    ]
    assert rpd.paddle_markdown({"markdown": {"text": "only md"}}, True) == "only md"


def test_table_with_merged_cells_stays_html():
    html = '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'
    assert rpd.table_to_markdown(html) == html


def test_table_cells_escape_pipes_and_newlines():
    html = "<table><tr><td>a|b</td><td>x<br>y</td></tr></table>"
    assert rpd.table_to_markdown(html) == "| a\\|b | x y |\n| --- | --- |"


# --- end to end against fake endpoints ----------------------------------------------------------


async def _no_sleep(_s: float) -> None:
    return None


def _png(path: Path, size=(1240, 1754)) -> Path:
    Image.new("RGB", size, "white").save(path)
    return path


class _Endpoint:
    """A fake load balancer. /health starts at 204; chat and /layout-parsing answer per model."""

    def __init__(self, answer, health=(204, 200)):
        self.answer = answer
        self.health = list(health)
        self.requests: list[tuple[str, dict]] = []
        self.auth: set[str | None] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.auth.add(request.headers.get("authorization"))
        if request.url.path.endswith("/health"):
            return httpx.Response(self.health.pop(0) if len(self.health) > 1 else self.health[0])
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        return self.answer(request.url.path, body, len(self.requests))


def _chat(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _run(model, endpoint, inputs, base_url=BASE, **kw):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as http:
            return await rpd.parse_document(
                model, inputs, base_url, "key-1", http=http, poll_s=0, sleep=_no_sleep, **kw
            )

    return asyncio.run(go())


def test_dotsocr_end_to_end(tmp_path):
    def answer(path, body, n):
        return _chat(json.dumps([{"category": "Text", "text": f"page {n}"}]))

    ep = _Endpoint(answer)
    md = _run("dotsocr", ep, [_png(tmp_path / "a.png"), _png(tmp_path / "b.png")], concurrency=1)
    assert md == "<!-- page 1 -->\n\npage 1\n\n<!-- page 2 -->\n\npage 2\n"
    assert ep.auth == {"Bearer key-1"}
    path, body = ep.requests[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "dotsocr"
    spec = rpd.MODELS["dotsocr"]
    content = body["messages"][0]["content"]
    assert content[1]["text"] == spec.message_prefix + spec.prompt
    h, w = rpd.smart_resize(1754, 1240, 28, 3136, 4_000_000)
    assert _decode(content[0]["image_url"]["url"]).size == (w, h)


def test_typhoon_end_to_end(tmp_path):
    ep = _Endpoint(lambda p, b, n: _chat("# Title\n\n<page_number>1</page_number>"))
    md = _run("typhoon_ocr15", ep, [_png(tmp_path / "a.png", size=(3000, 4000))])
    assert md == "<!-- page 1 -->\n\n# Title\n\n1\n"
    body = ep.requests[0][1]
    assert body["model"] == "typhoon"
    assert max(_decode(body["messages"][0]["content"][0]["image_url"]["url"]).size) == 1800


def test_teleocr_end_to_end(tmp_path):
    layout = "<box:0 0 1000 100><label:title><up>\n<box:0 200 1000 600><label:table><up>\n"

    def answer(path, body, n):
        text = body["messages"][1]["content"][1]["text"]
        if text == rpd.TELEOCR_LAYOUT_PROMPT:
            return _chat(layout)
        if "OTSL" in text:
            return _chat("<fcel>A<fcel>B<nl>")
        return _chat("Statement")

    ep = _Endpoint(answer)
    md = _run("teleocr", ep, [_png(tmp_path / "a.png")])
    assert md == "<!-- page 1 -->\n\n# Statement\n\n| A | B |\n| --- | --- |\n"
    first = ep.requests[0][1]
    assert first["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert _decode(first["messages"][1]["content"][0]["image_url"]["url"]).size == (1036, 1036)
    assert first["vllm_xargs"] == {"no_repeat_ngram_size": 100}
    assert len(ep.requests) == 3  # layout + two blocks


def test_paddle_end_to_end(tmp_path):
    def answer(path, body, n):
        result = {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "parsing_res_list": [{"block_label": "text", "block_content": "hello"}]
                    },
                    "markdown": {"text": "hello"},
                }
            ]
        }
        return httpx.Response(200, json={"errorCode": 0, "result": result})

    ep = _Endpoint(answer)
    md = _run("paddleocr_vl", ep, [_png(tmp_path / "a.png")])
    assert md == "<!-- page 1 -->\n\nhello\n"
    path, body = ep.requests[0]
    assert path == "/layout-parsing"  # the endpoint root, not /v1
    assert body["fileType"] == 1 and body["visualize"] is False
    assert Image.open(io.BytesIO(base64.b64decode(body["file"]))).size == (1240, 1754)


def test_paddle_pipeline_error_is_reported(tmp_path):
    ep = _Endpoint(lambda p, b, n: httpx.Response(200, json={"errorCode": 500, "errorMsg": "bad"}))
    md = _run("paddleocr_vl", ep, [_png(tmp_path / "a.png")])
    assert "<!-- parse error: pipeline error 500: bad -->" in md


def test_pdf_pages_are_rendered_at_200_dpi(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _write_digital_pdf(pdf, n_pages=2)  # 4 x 3 in pages
    ep = _Endpoint(lambda p, b, n: _chat("text"), health=(200,))
    md = _run("ovisocr2", ep, [pdf])
    assert md.count("<!-- page ") == 2
    sent = _decode(ep.requests[0][1]["messages"][0]["content"][0]["image_url"]["url"])
    h, w = rpd.smart_resize(600, 800, 32, 200704, 8294400)
    assert sent.size == (w, h)


def test_unparseable_dots_output_falls_back_to_raw_text(tmp_path):
    ep = _Endpoint(lambda p, b, n: _chat("plain words"), health=(200,))
    md = _run("dotsocr", ep, [_png(tmp_path / "a.png")])
    assert "plain words" in md and "<!-- parse error" in md


def test_no_worker_within_the_wait_raises(tmp_path):
    ep = _Endpoint(lambda p, b, n: _chat("x"), health=(204,))
    with pytest.raises(rpd.EndpointNotReady):
        _run("typhoon_ocr15", ep, [_png(tmp_path / "a.png")], wait_s=0)


def test_gateway_error_is_retried(tmp_path):
    def answer(path, body, n):
        return httpx.Response(502, text="no workers available") if n == 1 else _chat("ok")

    ep = _Endpoint(answer, health=(200,))
    assert "ok" in _run("typhoon_ocr15", ep, [_png(tmp_path / "a.png")])
    assert len(ep.requests) == 2


def test_plain_http_to_runpod_is_refused(tmp_path):
    ep = _Endpoint(lambda p, b, n: _chat("x"))
    with pytest.raises(ValueError, match="https"):
        _run("dotsocr", ep, [], base_url="http://abcdefghij12.api.runpod.ai/v1")


def test_cli_reads_the_models_env(tmp_path, monkeypatch):
    out = tmp_path / "out.md"
    monkeypatch.setenv("TYPHOON_BASE_URL", BASE)
    monkeypatch.setenv("TYPHOON_API_KEY", "typhoon-key")
    seen = {}

    async def fake_parse(model, inputs, base_url, api_key, **kw):
        seen.update(model=model, base_url=base_url, api_key=api_key, **kw)
        return "# hi\n"

    monkeypatch.setattr(rpd, "parse_document", fake_parse)
    args = ["--model", "typhoon_ocr15", str(_png(tmp_path / "a.png")), "-o", str(out)]
    assert rpd.main([*args, "--drop-page-furniture"]) == 0
    assert out.read_text() == "# hi\n"
    assert seen["model"] == "typhoon_ocr15"
    assert seen["base_url"] == BASE and seen["api_key"] == "typhoon-key"
    assert seen["keep_furniture"] is False


def test_cli_falls_back_to_the_management_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DOTSOCR_BASE_URL", BASE)
    monkeypatch.delenv("DOTSOCR_API_KEY", raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY", "mgmt")
    seen = {}

    async def fake_parse(model, inputs, base_url, api_key, **kw):
        seen["api_key"] = api_key
        return ""

    monkeypatch.setattr(rpd, "parse_document", fake_parse)
    rpd.main(["--model", "dotsocr", str(_png(tmp_path / "a.png")), "-o", str(tmp_path / "o")])
    assert seen["api_key"] == "mgmt"


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (rpd.EndpointNotReady("no worker ready after 600s"), 3, "no worker ready"),
        (ValueError("x.docx: not a PDF or image"), 2, "not a PDF or image"),
        (
            httpx.HTTPStatusError(
                "denied",
                request=httpx.Request("POST", BASE),
                response=httpx.Response(401, request=httpx.Request("POST", BASE)),
            ),
            4,
            "check the API key",
        ),
    ],
)
def test_cli_reports_errors_without_a_traceback(
    tmp_path, monkeypatch, capsys, error, code, message
):
    monkeypatch.setenv("DOTSOCR_BASE_URL", BASE)
    monkeypatch.setenv("DOTSOCR_API_KEY", "k")

    async def fail(*args, **kw):
        raise error

    monkeypatch.setattr(rpd, "parse_document", fail)
    assert rpd.main(["--model", "dotsocr", str(_png(tmp_path / "a.png"))]) == code
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err
