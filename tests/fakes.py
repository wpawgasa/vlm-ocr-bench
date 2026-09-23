"""Test doubles for the OpenAI SDK and for `VllmClient` (no network, no GPU)."""

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import openai
from openai.types.chat import ChatCompletion


def completion(
    text: str,
    *,
    logprobs: list[tuple[str, float]] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int | None = None,
) -> ChatCompletion:
    """A real `ChatCompletion` object as vLLM would return it."""
    lp = None
    if logprobs is not None:
        lp = {
            "content": [
                {"token": t, "logprob": v, "bytes": None, "top_logprobs": []} for t, v in logprobs
            ]
        }
    n_completion = completion_tokens if completion_tokens is not None else len(logprobs or [])
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                    "logprobs": lp,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": n_completion,
                "total_tokens": prompt_tokens + n_completion,
            },
        }
    )


def status_error(code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://fake/v1/chat/completions")
    response = httpx.Response(code, request=request)
    cls = openai.InternalServerError if code >= 500 else openai.BadRequestError
    return cls(f"HTTP {code}", response=response, body=None)


class _ScriptedCompletions:
    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return await item()
        return item


class FakeOpenAI:
    """`openai.AsyncOpenAI`-compatible: `chat.completions.create` is scripted per call."""

    def __init__(self, script: list[Any]):
        self.completions = _ScriptedCompletions(script)
        self.chat = SimpleNamespace(completions=self.completions)


def sleeper(seconds: float, then: Any) -> Callable[[], Any]:
    async def _run() -> Any:
        await asyncio.sleep(seconds)
        return then

    return _run


class FakeChatClient:
    """Stands in for `VllmClient`: `responder(image, prompt, sampling)` returns either a
    string (success), an Exception instance (the call failed twice) or a `CallResult`."""

    def __init__(self, responder: Callable[..., Any], delay_s: float = 0.0, healthy: bool = True):
        self.responder = responder
        self.delay_s = delay_s
        self.healthy = healthy
        self.calls: list[dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def health(self) -> None:
        from ocr_bench.models.vllm_client import EndpointUnavailable

        if not self.healthy:
            raise EndpointUnavailable("fake", "http://fake:8000/v1")

    async def chat(self, image, prompt, sampling, *, stream: bool = False):
        import time

        from ocr_bench.models.vllm_client import CallResult

        self.calls.append({"image": image, "prompt": prompt, "sampling": sampling})
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        started = time.perf_counter()
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            out = self.responder(image, prompt, sampling)
        finally:
            self.in_flight -= 1
        ended = time.perf_counter()
        if isinstance(out, CallResult):
            out.started, out.ended = started, ended
            return out
        if isinstance(out, BaseException):
            return CallResult(error=str(out), attempts=2, started=started, ended=ended)
        n = max(1, len(out) // 4)
        return CallResult(
            text=out,
            tokens=[out[i::n] for i in range(n)],
            token_logprobs=[-0.1] * n,
            prompt_tokens=100,
            completion_tokens=n,
            latency_ms=(ended - started) * 1000,
            attempts=1,
            started=started,
            ended=ended,
        )


class FakePipelineClient:
    """Stands in for `PipelineClient`: `responder(image)` returns the page-result JSON text
    (success) or an Exception instance (the request failed for good)."""

    def __init__(self, responder: Callable[..., Any], healthy: bool = True):
        self.responder = responder
        self.healthy = healthy
        self.calls: list[Any] = []

    async def health(self) -> None:
        from ocr_bench.models.vllm_client import EndpointUnavailable

        if not self.healthy:
            raise EndpointUnavailable("fake-pipeline", "http://fake:8080")

    async def layout(self, image):
        import time

        from ocr_bench.models.pipeline_client import PipelineResult

        self.calls.append(image)
        started = time.perf_counter()
        out = self.responder(image)
        ended = time.perf_counter()
        if isinstance(out, BaseException):
            return PipelineResult(error=str(out), attempts=2, started=started, ended=ended)
        return PipelineResult(
            text=out,
            latency_ms=(ended - started) * 1000,
            attempts=1,
            started=started,
            ended=ended,
        )


def make_run(data_dir, run_id: str, rows: list) -> Any:
    """Write `rows` (ManifestRow) as a run's manifest plus a small PNG per image_path."""
    from PIL import Image

    from ocr_bench.jsonl import write_rows_atomic

    root = data_dir / "runs" / run_id
    for r in rows:
        path = root / r.image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            Image.new("RGB", (200, 280), (255, 255, 255)).save(path)
    write_rows_atomic(root / "manifest.jsonl", rows)
    return root


class FakeOcrModel:
    """An `OcrModel` whose output per call is chosen by `behaviour(n, row)`:
    "ok", "error" (row with raw.error), "wrong_id" (row for another sample) or an exception."""

    def __init__(self, name: str, behaviour=None, healthy: bool = True):
        self.name = name
        self.behaviour = behaviour or (lambda n, row: "ok")
        self.healthy = healthy
        self.requested: list[str] = []

    async def health(self) -> None:
        from ocr_bench.models.vllm_client import EndpointUnavailable

        if not self.healthy:
            raise EndpointUnavailable(self.name, "http://down:8000/v1", "GET -> 503")

    async def predict(self, row, image):
        from ocr_bench.schemas import NormalizedPage, PredictionRow, RawPrediction

        self.requested.append(row.sample_id)
        outcome = self.behaviour(len(self.requested), row)
        if isinstance(outcome, BaseException):
            raise outcome
        await asyncio.sleep(0)
        return PredictionRow(
            sample_id=row.sample_id + ("-x" if outcome == "wrong_id" else ""),
            condition=row.condition,
            model=self.name,
            prompt_version="fake-v1",
            raw=RawPrediction(text="t", error="timeout" if outcome == "error" else None),
            normalized=NormalizedPage(text="" if outcome == "error" else "t"),
            latency_ms=float(len(self.requested)),
            prompt_tokens=1,
            completion_tokens=1,
        )
