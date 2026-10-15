"""`ocrbench runpod smoke`: health, served name, one page, Paddle layout, auth rejection."""

import asyncio
import itertools

import httpx
from PIL import Image

from ocr_bench.config import load_model_config
from ocr_bench.models.base import PlannedModel
from ocr_bench.runpod.config import load_runpod_config
from ocr_bench.runpod.smoke import SMOKE_IMAGE, run_smoke
from tests.fakes import FakeChatClient, FakePipelineClient
from tests.runpod_helpers import ROOT, pinned_configs

BASE = "https://api.runpod.ai/v2/abc/v1"
LAYOUT_JSON = '{"layoutParsingResults": []}'


def _ctx(tmp_path, name):
    configs = pinned_configs(tmp_path / "c")
    rcfg = load_runpod_config(configs / f"{name}.yaml")
    mcfg = load_model_config(ROOT / "configs" / "models" / f"{name}.yaml", name)
    return rcfg, mcfg


def _http(served, unauth_status=401, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append((request.url.path, request.headers.get("authorization")))
        if request.url.path != "/v2/abc/v1/models":
            return httpx.Response(404)
        if request.headers.get("authorization") != "Bearer k":
            return httpx.Response(unauth_status)
        return httpx.Response(200, json={"data": [{"id": served}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _clock(step):
    ticks = itertools.count()
    return lambda: next(ticks) * step


def _run(rcfg, mcfg, http, *, text="สวัสดี", step=1.0, chat=None, pipeline=None):
    client = chat or FakeChatClient(lambda *a: text)
    model = PlannedModel(mcfg, client, pipeline)
    return asyncio.run(
        run_smoke(rcfg, mcfg, model, base_url=BASE, api_key="k", http=http, clock=_clock(step))
    )


def test_fixture_loads():
    assert Image.open(SMOKE_IMAGE).size[0] > 0
    assert "statement" not in SMOKE_IMAGE.name


def test_pass(tmp_path):
    rcfg, mcfg = _ctx(tmp_path, "typhoon_ocr15")
    report = _run(rcfg, mcfg, _http(mcfg.served_model_name))
    assert report.failures == []
    assert report.ok
    assert report.request_s > 0
    assert report.cold_start_s >= 0


def test_unauthenticated_request_carries_no_key(tmp_path):
    rcfg, mcfg = _ctx(tmp_path, "typhoon_ocr15")
    seen = []
    _run(rcfg, mcfg, _http(mcfg.served_model_name, seen=seen))
    assert ("/v2/abc/v1/models", None) in seen


def test_wrong_served_name_fails_before_any_page(tmp_path):
    rcfg, mcfg = _ctx(tmp_path, "typhoon_ocr15")
    chat = FakeChatClient(lambda *a: "text")
    report = _run(rcfg, mcfg, _http("other-model"), chat=chat)
    assert not report.ok
    assert mcfg.served_model_name in report.failures[0] and "other-model" in report.failures[0]
    assert chat.calls == []


def test_slow_request_fails(tmp_path):
    rcfg, mcfg = _ctx(tmp_path, "typhoon_ocr15")
    report = _run(rcfg, mcfg, _http(mcfg.served_model_name), step=rcfg.slo.page_p95_s + 1)
    assert not report.ok
    assert any("page_p95_s" in f for f in report.failures)


def test_unauthenticated_accepted_fails(tmp_path):
    rcfg, mcfg = _ctx(tmp_path, "typhoon_ocr15")

    def handler(request):
        return httpx.Response(200, json={"data": [{"id": mcfg.served_model_name}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    report = _run(rcfg, mcfg, http)
    assert not report.ok
    assert any("without a key" in f for f in report.failures)


def test_request_error_fails(tmp_path):
    rcfg, mcfg = _ctx(tmp_path, "typhoon_ocr15")
    chat = FakeChatClient(lambda *a: RuntimeError("boom"))
    report = _run(rcfg, mcfg, _http(mcfg.served_model_name), chat=chat)
    assert not report.ok


def test_empty_text_fails(tmp_path):
    rcfg, mcfg = _ctx(tmp_path, "typhoon_ocr15")
    report = _run(rcfg, mcfg, _http(mcfg.served_model_name), text="")
    assert not report.ok


def test_unhealthy_fails_and_skips_rest(tmp_path):
    rcfg, mcfg = _ctx(tmp_path, "typhoon_ocr15")
    chat = FakeChatClient(lambda *a: "x", healthy=False)
    report = _run(rcfg, mcfg, _http(mcfg.served_model_name), chat=chat)
    assert not report.ok
    assert chat.calls == []


def test_paddle_also_calls_layout_parsing(tmp_path):
    rcfg, mcfg = _ctx(tmp_path, "paddleocr_vl")
    pipeline = FakePipelineClient(lambda image: LAYOUT_JSON)
    report = _run(rcfg, mcfg, _http(mcfg.served_model_name), pipeline=pipeline)
    assert report.failures == []
    assert len(pipeline.calls) == 2  # the full-page plan, then the explicit layout check
    assert report.layout_s is not None


def test_paddle_layout_error_fails(tmp_path):
    rcfg, mcfg = _ctx(tmp_path, "paddleocr_vl")
    pipeline = FakePipelineClient(lambda image: RuntimeError("down"))
    report = _run(rcfg, mcfg, _http(mcfg.served_model_name), pipeline=pipeline)
    assert not report.ok
