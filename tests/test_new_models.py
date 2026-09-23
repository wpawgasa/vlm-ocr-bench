"""OvisOCR2 and PaddleOCR-VL-1.6: configs, parsers, the pipeline plan kind and client."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from ocr_bench.config import ModelConfig, PipelineEndpoint, load_model_config
from ocr_bench.models.parsers import ParseContext, clean_truncated_repeats, parse
from ocr_bench.models.pipeline_client import PipelineClient, page_result
from ocr_bench.models.registry import build_model
from ocr_bench.models.vllm_client import EndpointUnavailable
from ocr_bench.schemas import BlockType, Condition, GtKind, ManifestRow, Source, Task
from tests.fakes import FakeChatClient, FakePipelineClient

MODELS = Path(__file__).resolve().parents[1] / "configs" / "models"
FG = "Fine-grained text recognition"
FG_QUESTION = "ช่วยดึงข้อความที่อยู่ในพิกัด [274, 385, 573, 501] ของรูปภาพออกมาให้หน่อย"
IMG = Image.new("RGB", (2000, 1000), (255, 255, 255))


def cfg(name: str):
    return load_model_config(MODELS / f"{name}.yaml", name)


def row(task: Task, subtask: str | None = None, question: str = ""):
    return ManifestRow(
        sample_id="s1",
        source=Source.thaiocrbench,
        task=task,
        subtask=subtask,
        condition=Condition.clean,
        image_path="img/s1.png",
        gt_kind=GtKind.text,
        question=question,
    )


def run(m, r, image=IMG):
    return asyncio.run(m.predict(r, image))


def paddle_json(blocks, markdown="md"):
    return json.dumps(
        {"prunedResult": {"parsing_res_list": blocks}, "markdown": {"text": markdown}},
        ensure_ascii=False,
    )


BLOCKS = [
    {"block_label": "header", "block_content": "ธนาคารกรุงเทพ", "block_bbox": [10, 10, 500, 60]},
    {
        "block_label": "text",
        "block_content": "เลขที่บัญชี 123-4-56789-0",
        "block_bbox": [10, 80, 900, 120],
    },
    {"block_label": "image", "block_content": "", "block_bbox": [1500, 10, 1900, 200]},
    {
        "block_label": "table",
        "block_content": "".join(
            [
                "<table>",
                "<tr><td>วันที่</td><td>ยอด</td></tr>",
                "<tr><td>01/01</td><td>100.00</td></tr>",
                "</table>",
            ]
        ),
        "block_bbox": [10, 200, 1900, 900],
    },
]


def paddle(responder):
    fake = FakePipelineClient(responder)
    chat = FakeChatClient(lambda *_: '{"answer": "x"}')
    return build_model(cfg("paddleocr_vl"), client=chat, pipeline=fake), fake, chat


# --- configs --------------------------------------------------------------------------------


def test_ovis_prompt_and_sampling_follow_model_card():
    c = cfg("ovisocr2")
    prompt = c.plans["ocr_fullpage"].prompt
    assert prompt.startswith("\nExtract all readable content from the image")
    assert prompt.endswith("Preserve the original text without translation or paraphrasing.")
    assert c.sampling.max_tokens == 16384 and c.sampling.temperature == 0.0
    assert (c.resize_factor, c.min_pixels, c.max_pixels) == (32, 448 * 448, 2880 * 2880)
    assert c.system_prompt is None


def test_paddle_ocr_tasks_use_pipeline_and_questions_use_vllm():
    c = cfg("paddleocr_vl")
    assert c.served_model_name == "PaddleOCR-VL-1.6-0.9B"
    for task in ("ocr_fullpage", "docparse", "statement", "table", "ocr_line", "handwriting"):
        assert c.plans[task].kind == "pipeline"
    assert c.plans[f"ocr_line@{FG}"].kind == "crop_pipeline"
    for task in ("kie", "kie_map", "classify"):
        assert c.plans[task].kind == "question"


def test_pipeline_plan_without_endpoint_is_rejected():
    data = cfg("paddleocr_vl").model_dump()
    data["pipeline"] = None
    with pytest.raises(ValueError, match="no `pipeline` endpoint"):
        ModelConfig.model_validate(data)


def test_build_paddle_without_pipeline_env_raises(monkeypatch):
    monkeypatch.delenv("PADDLE_PIPELINE_URL", raising=False)
    with pytest.raises(EndpointUnavailable, match="PADDLE_PIPELINE_URL"):
        build_model(cfg("paddleocr_vl"), client=FakeChatClient(lambda *_: ""))


# --- OvisOCR2 -------------------------------------------------------------------------------


def test_ovis_resizes_to_factor_32_within_bounds():
    m = build_model(cfg("ovisocr2"), client=(client := FakeChatClient(lambda *_: "ข้อความ")))
    run(m, row(Task.ocr_fullpage), Image.new("RGB", (4000, 3000)))
    w, h = client.calls[0]["image"].size
    assert w % 32 == 0 and h % 32 == 0 and w * h <= 2880 * 2880
    run(m, row(Task.ocr_fullpage), Image.new("RGB", (100, 50)))
    w, h = client.calls[1]["image"].size
    assert w % 32 == 0 and h % 32 == 0 and w * h >= 448 * 448


def test_ovis_markdown_figures_tables_and_img_tags_dropped():
    raw = (
        "# ใบแจ้งยอด\n\n"
        '<img src="images/bbox_100_0_500_250.jpg" />\n\n'
        "<table><tr><td>A</td><td>B</td></tr></table>\n\nท้ายหน้า"
    )
    page = parse("ovis_markdown", raw, ParseContext(orig_size=(2000, 1000)))
    assert "images/bbox_" not in page.text
    assert page.text.startswith("# ใบแจ้งยอด") and page.text.endswith("ท้ายหน้า")
    figures = [b for b in page.blocks if b.type == BlockType.figure]
    assert figures[0].bbox == (200.0, 0.0, 1000.0, 250.0)
    assert page.tables == [[["A", "B"]]]


def test_clean_truncated_repeats_cuts_long_repeated_tail():
    head = "ก" * 8000
    looped = head + "นิติวิธี " * 50
    cleaned = clean_truncated_repeats(looped)
    assert cleaned == head + "นิติวิธี "
    assert clean_truncated_repeats("สั้น " * 10) == "สั้น " * 10  # under 8000 chars: untouched


# --- PaddleOCR-VL pipeline ------------------------------------------------------------------


def test_paddle_layout_keeps_headers_in_text_and_parses_tables():
    page = parse("paddle_layout", paddle_json(BLOCKS), ParseContext(orig_size=(2000, 1000)))
    assert page.text.split("\n\n")[:2] == ["ธนาคารกรุงเทพ", "เลขที่บัญชี 123-4-56789-0"]
    assert [b.type for b in page.blocks] == [
        BlockType.header,
        BlockType.text,
        BlockType.figure,
        BlockType.table,
    ]
    assert page.blocks[3].bbox == (10.0, 200.0, 1900.0, 900.0)
    assert page.tables == [[["วันที่", "ยอด"], ["01/01", "100.00"]]]


def test_paddle_layout_without_blocks_falls_back_to_markdown():
    raw = json.dumps({"prunedResult": {}, "markdown": {"text": "ข้อความ"}})
    page = parse("paddle_layout", raw, ParseContext())
    assert page.text == "ข้อความ"
    assert page.parse_error == "no parsing_res_list in pipeline result"


def test_paddle_pipeline_plan_sends_page_once_and_records_call():
    m, fake, chat = paddle(lambda image: paddle_json(BLOCKS))
    pred = run(m, row(Task.statement))
    assert len(fake.calls) == 1 and fake.calls[0].size == IMG.size and not chat.calls
    assert pred.prompt_version == "paddle-vl16-pipeline-v1"
    assert pred.n_requests == 1 and pred.raw.calls[0].kind == "pipeline"
    assert pred.raw.token_logprobs == [] and pred.raw.error is None
    assert pred.normalized.tables and pred.latency_ms >= 0


def test_paddle_fine_grained_crops_region():
    m, fake, _ = paddle(lambda image: paddle_json([{"block_label": "text", "block_content": "ก"}]))
    pred = run(m, row(Task.ocr_line, FG, question=FG_QUESTION))
    assert fake.calls[0].size == (598, 116)
    assert pred.prompt_version == "paddle-vl16-pipeline-crop-v1"
    assert pred.normalized.text == "ก"


def test_paddle_pipeline_failure_gives_error_row():
    m, _, _ = paddle(lambda image: RuntimeError("HTTP 500: boom"))
    pred = run(m, row(Task.ocr_fullpage))
    assert pred.raw.error == "HTTP 500: boom" and pred.normalized.text == ""


def test_paddle_question_goes_to_vllm_not_pipeline():
    m, fake, chat = paddle(lambda image: paddle_json(BLOCKS))
    run(m, row(Task.kie, question="ชื่อธนาคารคืออะไร"))
    assert not fake.calls and chat.calls[0]["prompt"] == "ชื่อธนาคารคืออะไร"


def test_paddle_health_checks_pipeline_too():
    fake = FakePipelineClient(lambda image: "", healthy=False)
    m = build_model(cfg("paddleocr_vl"), client=FakeChatClient(lambda *_: ""), pipeline=fake)
    with pytest.raises(EndpointUnavailable):
        asyncio.run(m.health())


# --- PipelineClient -------------------------------------------------------------------------

OK_ENVELOPE = {
    "logId": "1",
    "errorCode": 0,
    "errorMsg": "Success",
    "result": {
        "layoutParsingResults": [
            {
                "prunedResult": {"parsing_res_list": BLOCKS},
                "markdown": {"text": "md", "images": {"a.jpg": "base64..."}},
                "outputImages": {"layout": "base64..."},
            }
        ],
        "dataInfo": {},
    },
}


def client_with(handler) -> tuple[PipelineClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(len(seen))

    http = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    endpoint = PipelineEndpoint(endpoint_env="X", request={"visualize": False})
    return PipelineClient("http://pipe:8080/", endpoint, http_client=http), seen


def test_page_result_strips_images_and_rejects_errors():
    page = page_result(OK_ENVELOPE)
    assert page == {"prunedResult": {"parsing_res_list": BLOCKS}, "markdown": {"text": "md"}}
    with pytest.raises(ValueError, match="pipeline error 500"):
        page_result({"errorCode": 500, "errorMsg": "boom"})
    with pytest.raises(ValueError, match="no layoutParsingResults"):
        page_result({"errorCode": 0, "result": {"layoutParsingResults": []}})


def test_pipeline_client_posts_image_and_options():
    client, seen = client_with(lambda n: httpx.Response(200, json=OK_ENVELOPE))
    result = asyncio.run(client.layout(Image.new("RGB", (20, 10))))
    assert result.error is None and result.attempts == 1
    assert json.loads(result.text)["markdown"] == {"text": "md"}
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://pipe:8080/layout-parsing"
    assert body["fileType"] == 1 and body["visualize"] is False and len(body["file"]) > 20


def test_pipeline_client_retries_5xx_once_then_succeeds():
    client, seen = client_with(
        lambda n: (
            httpx.Response(503, text="busy") if n == 1 else httpx.Response(200, json=OK_ENVELOPE)
        )
    )
    result = asyncio.run(client.layout(IMG))
    assert result.error is None and result.attempts == 2 and len(seen) == 2


def test_pipeline_client_does_not_retry_4xx_or_pipeline_errors():
    client, seen = client_with(lambda n: httpx.Response(422, text="bad file"))
    result = asyncio.run(client.layout(IMG))
    assert result.error.startswith("HTTP 422") and len(seen) == 1
    client, seen = client_with(
        lambda n: httpx.Response(200, json={"errorCode": 1, "errorMsg": "x"})
    )
    result = asyncio.run(client.layout(IMG))
    assert result.error == "pipeline error 1: x" and len(seen) == 1
