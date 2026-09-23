"""Tests for ocr_bench.models.base and the per-model adapters (fake client, no network)."""

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from ocr_bench.config import load_model_config
from ocr_bench.models.base import PlannedModel, parse_region
from ocr_bench.models.registry import build_model
from ocr_bench.schemas import BlockType, Condition, GtKind, ManifestRow, Source, Task
from tests.fakes import FakeChatClient, FakeOpenAI, completion

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "configs" / "models"
FG = "Fine-grained text recognition"
FG_QUESTION = (
    "แบ่งความยาวและความสูงของรูปภาพออกเป็น 1000 ส่วน แล้วช่วยดึงข้อความที่อยู่ในพิกัด "
    "[274, 385, 573, 501] ของรูปภาพออกมาให้หน่อย"
)


def cfg(name: str):
    return load_model_config(MODELS / f"{name}.yaml", name)


def row(task: Task, subtask: str | None = None, question: str = "", sample_id: str = "s1"):
    return ManifestRow(
        sample_id=sample_id,
        source=Source.thaiocrbench,
        task=task,
        subtask=subtask,
        condition=Condition.clean,
        image_path=f"img/{sample_id}.png",
        gt_kind=GtKind.text,
        question=question,
    )


def model(name: str, responder, **kwargs) -> tuple[PlannedModel, FakeChatClient]:
    client = FakeChatClient(responder, **kwargs)
    return build_model(cfg(name), client=client), client


def run(m, r, image):
    return asyncio.run(m.predict(r, image))


IMG = Image.new("RGB", (2000, 1000), (255, 255, 255))


# --- region parsing -------------------------------------------------------------------------


def test_parse_region_takes_last_bracket():
    assert parse_region(FG_QUESTION) == (274, 385, 573, 501)
    assert parse_region("[1, 2, 3, 4] แล้ว [10, 20, 30, 40]") == (10, 20, 30, 40)
    assert parse_region("ไม่มีพิกัด") is None
    assert parse_region("[1, 2, 3]") is None


# --- request plans --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["teleocr", "dotsocr", "typhoon_ocr15", "ovisocr2"])
@pytest.mark.parametrize("task", [Task.kie, Task.kie_map, Task.classify])
def test_fallback_sends_benchmark_question_verbatim(name, task):
    question = "จากรูปภาพกรุณาดึงข้อมูล ชื่อร้าน เป็นรูปแบบ JSON"
    m, client = model(name, lambda *_: '{"ชื่อร้าน": "ร้านป้า"}')
    pred = run(m, row(task, question=question), IMG)
    assert [c["prompt"] for c in client.calls] == [question]
    assert pred.prompt_version == "question-v1"
    assert pred.n_requests == 1


def test_kie_unparseable_reply_is_kept_with_parse_error():
    m, _ = model("dotsocr", lambda *_: "ไม่ใช่ JSON")
    pred = run(m, row(Task.kie, question="q"), IMG)
    assert pred.raw.error is None
    assert pred.normalized.fields == {}
    assert pred.normalized.parse_error is not None


def test_dots_fullpage_uses_prompt_layout_all_en_with_prefix():
    dots = cfg("dotsocr")
    fake = FakeOpenAI([completion("[]", logprobs=[("[]", -0.01)])])
    m = build_model(dots, openai_client=fake, base_url="http://dots:8000/v1")
    pred = asyncio.run(m.predict(row(Task.ocr_fullpage, "Full-page OCR", question="Q?"), IMG))
    kwargs = fake.completions.calls[0]
    text = kwargs["messages"][-1]["content"][1]["text"]
    layout_prompt = dots.plan_for(Task.ocr_fullpage, None).prompt
    assert text == "<|img|><|imgpad|><|endofimg|>" + layout_prompt
    assert "Q?" not in text
    assert kwargs["max_tokens"] == 8192
    assert [m_["role"] for m_ in kwargs["messages"]] == ["user"]  # no system prompt
    assert pred.prompt_version == "dots-layout-all-v1"


def test_dots_image_is_smart_resized_and_bboxes_mapped_back():
    reply = '[{"bbox": [0, 0, 994, 504], "category": "Text", "text": "สวัสดี"}]'
    m, client = model("dotsocr", lambda *_: reply)
    pred = run(m, row(Task.ocr_fullpage), IMG)
    assert client.calls[0]["image"].size == (1988, 1008)
    x0, y0, x1, y1 = pred.normalized.blocks[0].bbox
    assert (x0, y0) == (0.0, 0.0)
    assert x1 == pytest.approx(1000.0) and y1 == pytest.approx(500.0)
    assert pred.normalized.text == "สวัสดี"


def test_dots_grounding_maps_region_to_resized_pixels():
    m, client = model("dotsocr", lambda *_: "ข้อความ")
    pred = run(m, row(Task.ocr_line, FG, question=FG_QUESTION), IMG)
    prompt = client.calls[0]["prompt"]
    assert prompt.startswith("Extract text from the given bounding box on the image")
    assert prompt.endswith("Bounding Box:\n[545, 388, 1139, 505]")
    assert client.calls[0]["image"].size == (1988, 1008)
    assert pred.prompt_version == "dots-grounding-ocr-v1"
    assert pred.normalized.text == "ข้อความ"


def test_teleocr_fine_grained_crops_region():
    m, client = model("teleocr", lambda *_: "ข้อความ")
    pred = run(m, row(Task.ocr_line, FG, question=FG_QUESTION), IMG)
    assert client.calls[0]["image"].size == (598, 116)
    assert client.calls[0]["prompt"] == "\nPlease output the text content from the image."
    assert pred.prompt_version == "teleocr-crop-text-v1"


def test_typhoon_fine_grained_crops_region_with_official_prompt():
    m, client = model("typhoon_ocr15", lambda *_: "ข้อความ")
    run(m, row(Task.ocr_line, FG, question=FG_QUESTION), IMG)
    assert client.calls[0]["image"].size == (598, 116)
    assert client.calls[0]["prompt"].startswith("Extract all text from the image.")


def test_missing_region_falls_back_to_whole_image():
    m, client = model("teleocr", lambda *_: "ข้อความ")
    pred = run(m, row(Task.ocr_line, FG, question="ไม่มีพิกัด"), IMG)
    assert client.calls[0]["image"].size == IMG.size
    assert pred.normalized.parse_error == "no region in question"
    assert pred.normalized.text == "ข้อความ"


def test_text_recognition_uses_native_prompt_on_whole_image():
    m, client = model("dotsocr", lambda *_: "ข้อความ")
    run(m, row(Task.ocr_line, "Text recognition", question="อ่านข้อความ"), IMG)
    assert client.calls[0]["prompt"] == "Extract the text content from this image."


def test_typhoon_resizes_longest_side_to_1800():
    m, client = model("typhoon_ocr15", lambda *_: "ข้อความ")
    run(m, row(Task.ocr_fullpage), Image.new("RGB", (3600, 1800)))
    assert client.calls[0]["image"].size == (1800, 900)
    assert client.calls[0]["sampling"].max_tokens == 10000


def test_teleocr_table_single_call_converts_otsl():
    m, client = model("teleocr", lambda *_: "<fcel>A<lcel><nl><fcel>1<fcel>2<nl>")
    pred = run(m, row(Task.table), IMG)
    assert "OTSL" in client.calls[0]["prompt"]
    assert client.calls[0]["sampling"].frequency_penalty == 0.005
    assert pred.normalized.tables == [[["A"], ["1", "2"]]]
    assert pred.prompt_version == "teleocr-otsl-v1"


def test_single_call_error_gives_error_row_with_empty_output():
    m, _ = model("dotsocr", lambda *_: TimeoutError("timeout twice"))
    pred = run(m, row(Task.ocr_fullpage), IMG)
    assert pred.raw.error is not None
    assert pred.normalized.text == "" and pred.normalized.blocks == []
    assert pred.n_requests == 1


def test_single_call_row_records_tokens_and_call():
    m, _ = model("dotsocr", lambda *_: "ข้อความยาวพอสมควร")
    pred = run(m, row(Task.handwriting), IMG)
    assert pred.prompt_tokens == 100
    assert pred.completion_tokens == len(pred.raw.token_logprobs) > 0
    assert len(pred.raw.calls) == 1 and pred.raw.calls[0].kind == "single"
    assert pred.raw.text == "ข้อความยาวพอสมควร"


# --- TeleOCR two-stage ----------------------------------------------------------------------

LAYOUT_6 = "\n".join(
    [
        "<box:100 50 900 100><label:header><rotate_up>",
        "<box:100 120 900 200><label:text><rotate_up>",
        "<box:100 220 900 500><label:table><rotate_up>",
        "<box:600 520 900 700><label:image><rotate_up>",
        "<box:100 520 300 600 250 700 50 600><label:seal><rotate_up>",
        "<box:100 720 900 780><label:list><rotate_up>",
        "<box:700 800 760 980><label:text><rotate_right>",
        "<box:100 900 500 950><label:equation><rotate_up>",
    ]
)
BLOCK_PROMPTS = {
    "\nPlease output the text content from the image.": "ข้อความ",
    "\nThis is the image of a table. Please output the table in OTSL format.": (
        "<fcel>A<lcel><nl><fcel>1<fcel>2<nl>"
    ),
    "\nSeal Recognition:": "ตราประทับ",
    "\nPlease write out the expression of the formula in the image using LaTeX format.": (
        "\\[ x^2 \\]"
    ),
}


def two_stage_responder(fail_prompt: str | None = None, layout=LAYOUT_6):
    def respond(image, prompt, sampling):
        if prompt == "\nAnalyze the image layout.":
            return layout
        if prompt == fail_prompt:
            return TimeoutError("timeout twice")
        return BLOCK_PROMPTS[prompt]

    return respond


def test_two_stage_page_sends_7_requests_and_writes_one_row():
    m, client = model("teleocr", two_stage_responder(), delay_s=0.02)
    pred = run(m, row(Task.ocr_fullpage), IMG)
    # 8 layout blocks: image and list are skipped -> 6 block calls + 1 layout call
    assert len(client.calls) == 7
    assert pred.n_requests == 7
    assert len(pred.raw.calls) == 7
    assert pred.raw.calls[0].kind == "layout"
    assert [c.kind for c in pred.raw.calls[1:]] == ["block"] * 6
    assert [c.block_index for c in pred.raw.calls[1:]] == [0, 1, 2, 4, 6, 7]
    assert pred.raw.error is None
    assert pred.prompt_version == "teleocr-2stage-v1"
    # latency covers the layout call plus the (concurrent) block calls
    assert pred.latency_ms >= 40
    assert pred.prompt_tokens == 700
    assert pred.completion_tokens == sum(c.completion_tokens for c in pred.raw.calls)
    assert len(pred.raw.token_logprobs) == sum(len(c.token_logprobs) for c in pred.raw.calls)


def test_two_stage_layout_request_shape():
    m, client = model("teleocr", two_stage_responder())
    run(m, row(Task.statement), IMG)
    layout = client.calls[0]
    assert layout["image"].size == (1036, 1036)
    assert layout["sampling"].presence_penalty == 0.0
    blocks = client.calls[1:]
    table = next(c for c in blocks if "OTSL" in c["prompt"])
    assert table["sampling"].presence_penalty == 1.0
    assert table["sampling"].frequency_penalty == 0.005
    assert table["sampling"].vllm_xargs == {"no_repeat_ngram_size": 100}
    text = next(c for c in blocks if c["prompt"].startswith("\nPlease output the text"))
    assert text["sampling"].frequency_penalty == 0.05
    # the rotate_right block crops to 120x180 px on a 2000x1000 page; rotated by 90 -> 180x120
    assert any(c["image"].size == (180, 120) for c in blocks)
    assert all(min(c["image"].size) >= 28 for c in blocks)


def test_two_stage_normalized_page_postprocessed():
    m, _ = model("teleocr", two_stage_responder())
    pred = run(m, row(Task.ocr_fullpage), IMG)
    page = pred.normalized
    assert page.tables == [[["A"], ["1", "2"]]]
    table_block = next(b for b in page.blocks if b.type == BlockType.table)
    assert 'colspan="2"' in table_block.text
    assert "$$x^2$$" in page.text
    assert page.blocks[0].type == BlockType.header
    assert page.blocks[0].bbox == (200.0, 50.0, 1800.0, 100.0)
    # raw.text keeps the verbatim block texts in reading order
    assert "<fcel>A<lcel>" in pred.raw.text
    assert pred.raw.text.startswith("ข้อความ")


def test_two_stage_partial_block_failure():
    m, _ = model("teleocr", two_stage_responder(fail_prompt="\nSeal Recognition:"))
    pred = run(m, row(Task.ocr_fullpage), IMG)
    assert pred.raw.error == "1 of 6 blocks failed"
    assert pred.n_requests == 7
    failed = [c for c in pred.raw.calls if c.error]
    assert len(failed) == 1 and failed[0].block_type == "seal"
    seal_blocks = [b for b in pred.normalized.blocks if b.text == ""]
    assert seal_blocks  # the failed block is kept, empty
    assert pred.normalized.tables  # the rest of the page is still there
    assert "ตราประทับ" not in pred.normalized.text


def test_two_stage_layout_failure_gives_error_row():
    def respond(image, prompt, sampling):
        return TimeoutError("timeout twice")

    m, client = model("teleocr", respond)
    pred = run(m, row(Task.ocr_fullpage), IMG)
    assert len(client.calls) == 1
    assert pred.n_requests == 1
    assert pred.raw.error is not None and "layout" in pred.raw.error
    assert pred.normalized.text == "" and pred.normalized.blocks == []


def test_two_stage_block_concurrency_bounded():
    many = "\n".join(
        f"<box:0 {i * 10} 1000 {i * 10 + 9}><label:text><rotate_up>" for i in range(20)
    )
    m, client = model("teleocr", two_stage_responder(layout=many), delay_s=0.01)
    pred = run(m, row(Task.ocr_fullpage), IMG)
    assert pred.n_requests == 21
    assert client.max_in_flight <= 8


def test_two_stage_unparseable_layout_is_parse_error_not_error():
    m, _ = model("teleocr", two_stage_responder(layout="no layout here"))
    pred = run(m, row(Task.ocr_fullpage), IMG)
    assert pred.raw.error is None
    assert pred.n_requests == 1
    assert pred.normalized.parse_error is not None


def test_build_model_without_endpoint_env_raises(monkeypatch):
    from ocr_bench.models.vllm_client import EndpointUnavailable

    monkeypatch.delenv("DOTSOCR_BASE_URL", raising=False)
    with pytest.raises(EndpointUnavailable) as exc_info:
        build_model(cfg("dotsocr"))
    assert "DOTSOCR_BASE_URL" in str(exc_info.value)


def test_build_model_picks_adapter_classes():
    from ocr_bench.models.dotsocr import DotsOCRModel
    from ocr_bench.models.teleocr import TeleOCRModel
    from ocr_bench.models.typhoon import TyphoonModel

    fake = FakeChatClient(lambda *_: "")
    assert isinstance(build_model(cfg("teleocr"), client=fake), TeleOCRModel)
    assert isinstance(build_model(cfg("dotsocr"), client=fake), DotsOCRModel)
    assert isinstance(build_model(cfg("typhoon_ocr15"), client=fake), TyphoonModel)


def test_teleocr_postprocess_keeps_non_otsl_table_text():
    from ocr_bench.models.teleocr import TeleOCRModel

    m = TeleOCRModel(cfg("teleocr"), FakeChatClient(lambda *_: ""))
    assert m.postprocess_block("table", "ตารางที่อ่านไม่ได้") == "ตารางที่อ่านไม่ได้"
    assert m.postprocess_block("char", "<fcel>a<nl>") == "<table><tr><td>a</td></tr></table>"
    assert m.postprocess_block("text", "<fcel>a<nl>") == "<fcel>a<nl>"
