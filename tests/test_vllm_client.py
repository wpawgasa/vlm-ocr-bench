"""Tests for ocr_bench.models.vllm_client (fake OpenAI client, no network)."""

import asyncio
import base64
import io

import httpx
import openai
import pytest
from PIL import Image

from ocr_bench.config import Sampling
from ocr_bench.models.vllm_client import EndpointUnavailable, VllmClient, health_url
from tests.fakes import FakeOpenAI, completion, sleeper, status_error

IMG = Image.new("RGB", (40, 30), color=(255, 255, 255))


def _client(script, **kwargs) -> tuple[VllmClient, FakeOpenAI]:
    fake = FakeOpenAI(script)
    client = VllmClient(
        "http://fake:8000/v1",
        "served",
        system_prompt=kwargs.pop("system_prompt", None),
        message_prefix=kwargs.pop("message_prefix", ""),
        client=fake,
        **kwargs,
    )
    return client, fake


def _chat(client, prompt="p", sampling=None):
    return asyncio.run(client.chat(IMG, prompt, sampling or Sampling()))


def test_success_with_logprobs():
    client, fake = _client([completion("สวัสดี", logprobs=[("สวั", -0.1), ("สดี", -0.2)])])
    result = _chat(client)
    assert result.error is None
    assert result.text == "สวัสดี"
    assert result.tokens == ["สวั", "สดี"]
    assert result.token_logprobs == [-0.1, -0.2]
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 2
    assert result.latency_ms >= 0
    assert result.ended >= result.started
    assert result.attempts == 1
    assert len(fake.completions.calls) == 1


def test_request_shape_and_deterministic_params():
    client, fake = _client(
        [completion("x", logprobs=[("x", 0.0)])],
        system_prompt="You are a helpful assistant.",
        message_prefix="<|img|><|imgpad|><|endofimg|>",
    )
    _chat(client, prompt="hello")
    kwargs = fake.completions.calls[0]
    assert kwargs["model"] == "served"
    assert kwargs["temperature"] == 0.0
    assert kwargs["seed"] == 0
    assert kwargs["max_tokens"] == 4096
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] == 0
    messages = kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    content = messages[1]["content"]
    assert content[0]["type"] == "image_url"
    url = content[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert decoded.format == "PNG" and decoded.size == (40, 30)
    assert content[1] == {"type": "text", "text": "<|img|><|imgpad|><|endofimg|>hello"}


def test_no_system_prompt_means_single_user_message():
    client, fake = _client([completion("x")])
    _chat(client)
    messages = fake.completions.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["user"]


def test_vllm_extensions_go_to_extra_body():
    sampling = Sampling(
        max_tokens=123,
        top_p=0.01,
        top_k=1,
        repetition_penalty=1.0,
        presence_penalty=1.0,
        frequency_penalty=0.05,
        vllm_xargs={"no_repeat_ngram_size": 100},
    )
    client, fake = _client([completion("x")])
    _chat(client, sampling=sampling)
    kwargs = fake.completions.calls[0]
    assert kwargs["max_tokens"] == 123
    assert kwargs["top_p"] == 0.01
    assert kwargs["presence_penalty"] == 1.0
    assert kwargs["frequency_penalty"] == 0.05
    assert "top_k" not in kwargs
    assert kwargs["extra_body"] == {
        "top_k": 1,
        "repetition_penalty": 1.0,
        "vllm_xargs": {"no_repeat_ngram_size": 100},
    }


def test_optional_params_omitted_when_unset():
    client, fake = _client([completion("x")])
    _chat(client)
    kwargs = fake.completions.calls[0]
    assert "top_p" not in kwargs
    assert "extra_body" not in kwargs or kwargs["extra_body"] == {}


def test_timeout_then_success_retries_once():
    client, fake = _client(
        [sleeper(5, completion("late")), completion("ok", logprobs=[("ok", -0.5)])],
        timeout_s=0.05,
    )
    result = _chat(client)
    assert result.error is None
    assert result.text == "ok"
    assert result.attempts == 2
    assert len(fake.completions.calls) == 2


def test_double_timeout_returns_error_and_empty_text():
    client, fake = _client(
        [sleeper(5, completion("late")), sleeper(5, completion("late"))], timeout_s=0.05
    )
    result = _chat(client)
    assert result.error is not None
    assert "timeout" in result.error.lower()
    assert result.text == ""
    assert result.tokens == [] and result.token_logprobs == []
    assert len(fake.completions.calls) == 2


def test_sdk_timeout_error_is_retried():
    request = httpx.Request("POST", "http://fake/v1/chat/completions")
    client, fake = _client([openai.APITimeoutError(request=request), completion("ok")])
    result = _chat(client)
    assert result.error is None and result.text == "ok"
    assert len(fake.completions.calls) == 2


def test_missing_logprobs_gives_empty_list_and_no_error():
    client, _ = _client([completion("no lp", logprobs=None, completion_tokens=3)])
    result = _chat(client)
    assert result.error is None
    assert result.text == "no lp"
    assert result.token_logprobs == []
    assert result.tokens == []
    assert result.completion_tokens == 3


def test_4xx_is_not_retried():
    client, fake = _client([status_error(400), completion("never")])
    result = _chat(client)
    assert result.error is not None and "400" in result.error
    assert result.text == ""
    assert len(fake.completions.calls) == 1


def test_5xx_is_retried_then_error():
    client, fake = _client([status_error(503), status_error(500)])
    result = _chat(client)
    assert result.error is not None
    assert len(fake.completions.calls) == 2


def test_unexpected_exception_never_raises():
    client, _ = _client([RuntimeError("boom"), RuntimeError("boom again")])
    result = _chat(client)
    assert "boom" in result.error


def test_health_url_strips_v1():
    assert health_url("http://teleocr:8000/v1") == "http://teleocr:8000/health"
    assert health_url("http://teleocr:8000/v1/") == "http://teleocr:8000/health"
    assert health_url("http://teleocr:8000") == "http://teleocr:8000/health"


def _http(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_health_ok():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200)

    client = VllmClient(
        "http://t:8000/v1", "teleocr", client=FakeOpenAI([]), http_client=_http(handler)
    )
    asyncio.run(client.health())
    assert seen == ["http://t:8000/health"]


def test_health_failure_raises_endpoint_unavailable():
    client = VllmClient(
        "http://t:8000/v1",
        "teleocr",
        client=FakeOpenAI([]),
        http_client=_http(lambda r: httpx.Response(503)),
    )
    with pytest.raises(EndpointUnavailable) as exc_info:
        asyncio.run(client.health())
    assert "teleocr" in str(exc_info.value)
    assert "http://t:8000/v1" in str(exc_info.value)


def test_health_connection_error_raises_endpoint_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = VllmClient(
        "http://t:8000/v1", "dotsocr", client=FakeOpenAI([]), http_client=_http(handler)
    )
    with pytest.raises(EndpointUnavailable):
        asyncio.run(client.health())


def test_version_reads_server_version():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"version": "0.22.1"})

    client = VllmClient(
        "http://t:8000/v1", "ovisocr2", client=FakeOpenAI([]), http_client=_http(handler)
    )
    assert asyncio.run(client.version()) == "0.22.1"
    assert seen == ["http://t:8000/version"]


def test_version_unknown_on_error_or_bad_body():
    for handler in (lambda r: httpx.Response(404), lambda r: httpx.Response(200, text="x")):
        client = VllmClient(
            "http://t:8000/v1", "dotsocr", client=FakeOpenAI([]), http_client=_http(handler)
        )
        assert asyncio.run(client.version()) == "unknown"


def test_streaming_records_ttft_and_text():
    from openai.types.chat import ChatCompletionChunk

    def chunk(content, lp=None, usage=None):
        choices = []
        if content is not None:
            choices = [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                    "logprobs": (
                        {"content": [{"token": content, "logprob": lp, "top_logprobs": []}]}
                        if lp is not None
                        else None
                    ),
                }
            ]
        return ChatCompletionChunk.model_validate(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": choices,
                "usage": usage,
            }
        )

    async def stream():
        async def gen():
            yield chunk("ab", -0.1)
            yield chunk("cd", -0.2)
            yield chunk(None, usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9})

        return gen()

    client, fake = _client([stream])
    result = asyncio.run(client.chat(IMG, "p", Sampling(), stream=True))
    assert result.error is None
    assert result.text == "abcd"
    assert result.token_logprobs == [-0.1, -0.2]
    assert result.prompt_tokens == 7 and result.completion_tokens == 2
    assert result.ttft_ms is not None and result.ttft_ms <= result.latency_ms
    assert fake.completions.calls[0]["stream"] is True
