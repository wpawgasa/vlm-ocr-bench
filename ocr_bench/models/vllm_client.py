"""Async client for one vLLM OpenAI-compatible server (design D4).

Every request carries the image as a base64 PNG data URL, `seed=0` and token logprobs.
Each request times out after `timeout_s` (120 s) and is retried once on a timeout, a
transport error or a 5xx; a 4xx is not retried. `chat` never raises: a request that
fails for good returns a `CallResult` with `error` set and empty text.
"""

import asyncio
import base64
import io
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai
from PIL import Image

from ocr_bench.config import Sampling

DEFAULT_TIMEOUT_S = 120.0
HEALTH_TIMEOUT_S = 10.0


class EndpointUnavailable(RuntimeError):
    def __init__(self, model: str, url: str, reason: str = ""):
        self.model = model
        self.url = url
        detail = f": {reason}" if reason else ""
        super().__init__(f"model '{model}' endpoint {url} is unavailable{detail}")


@dataclass
class CallResult:
    text: str = ""
    tokens: list[str] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0  # wall clock of the whole call, retry included
    ttft_ms: float | None = None
    error: str | None = None
    attempts: int = 0
    started: float = 0.0  # time.perf_counter() at the call's start
    ended: float = 0.0  # time.perf_counter() at the call's end


def health_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/health"


def image_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def request_params(sampling: Sampling) -> dict[str, Any]:
    """Map `Sampling` to OpenAI chat-completion kwargs; vLLM extensions go in `extra_body`."""
    params: dict[str, Any] = {
        "max_tokens": sampling.max_tokens,
        "temperature": sampling.temperature,
        "presence_penalty": sampling.presence_penalty,
        "frequency_penalty": sampling.frequency_penalty,
        "seed": 0,
        "logprobs": True,
        "top_logprobs": 0,
    }
    if sampling.top_p is not None:
        params["top_p"] = sampling.top_p
    extra: dict[str, Any] = {}
    if sampling.top_k is not None:
        extra["top_k"] = sampling.top_k
    if sampling.repetition_penalty is not None:
        extra["repetition_penalty"] = sampling.repetition_penalty
    if sampling.vllm_xargs:
        extra["vllm_xargs"] = dict(sampling.vllm_xargs)
    if extra:
        params["extra_body"] = extra
    return params


class _NoRetry(Exception):
    pass


def _describe(exc: BaseException) -> str:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, openai.APITimeoutError)):
        return f"timeout: {type(exc).__name__}"
    if isinstance(exc, openai.APIStatusError):
        return f"HTTP {exc.status_code}: {exc.message}"
    return f"{type(exc).__name__}: {exc}"


class VllmClient:
    def __init__(
        self,
        base_url: str,
        served_model_name: str,
        system_prompt: str | None = None,
        message_prefix: str = "",
        *,
        client: Any | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        model_label: str | None = None,
    ):
        self.base_url = base_url
        self.served_model_name = served_model_name
        self.system_prompt = system_prompt
        self.message_prefix = message_prefix
        self.timeout_s = timeout_s
        self.model_label = model_label or served_model_name
        if client is None:
            # The SDK's own retries are off: the retry policy lives in `chat`.
            client = openai.AsyncOpenAI(
                base_url=base_url,
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                max_retries=0,
                timeout=timeout_s + 10,
            )
        self._client = client
        self._http_client = http_client

    async def health(self) -> None:
        url = health_url(self.base_url)
        try:
            if self._http_client is not None:
                response = await self._http_client.get(url, timeout=HEALTH_TIMEOUT_S)
            else:
                async with httpx.AsyncClient() as http:
                    response = await http.get(url, timeout=HEALTH_TIMEOUT_S)
        except httpx.HTTPError as exc:
            raise EndpointUnavailable(self.model_label, self.base_url, str(exc)) from exc
        if response.status_code != 200:
            raise EndpointUnavailable(
                self.model_label, self.base_url, f"GET {url} -> {response.status_code}"
            )

    async def version(self) -> str:
        """The server's vLLM version from `GET /version`, or "unknown" if unavailable."""
        url = health_url(self.base_url)[: -len("/health")] + "/version"
        try:
            if self._http_client is not None:
                response = await self._http_client.get(url, timeout=HEALTH_TIMEOUT_S)
            else:
                async with httpx.AsyncClient() as http:
                    response = await http.get(url, timeout=HEALTH_TIMEOUT_S)
            if response.status_code == 200:
                return str(response.json().get("version") or "unknown")
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        return "unknown"

    def messages(self, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                    {"type": "text", "text": self.message_prefix + prompt},
                ],
            }
        )
        return messages

    async def chat(
        self, image: Image.Image, prompt: str, sampling: Sampling, *, stream: bool = False
    ) -> CallResult:
        kwargs: dict[str, Any] = {
            "model": self.served_model_name,
            "messages": self.messages(image, prompt),
            **request_params(sampling),
        }
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}

        started = time.perf_counter()
        last_error = "no attempt made"
        attempts = 0
        for _ in range(2):
            attempts += 1
            try:
                result = await asyncio.wait_for(self._attempt(kwargs, stream), self.timeout_s)
            except asyncio.CancelledError:
                raise
            except openai.APIStatusError as exc:
                last_error = _describe(exc)
                if exc.status_code < 500:
                    break
                continue
            except Exception as exc:  # timeouts, transport errors, malformed responses
                last_error = _describe(exc)
                continue
            ended = time.perf_counter()
            call = result.to_call_result()
            call.attempts = attempts
            call.started, call.ended = started, ended
            call.latency_ms = (ended - started) * 1000.0
            if call.ttft_ms is not None:  # measured from the call's start, like latency_ms
                call.ttft_ms += (result.started_attempt - started) * 1000.0
            return call
        ended = time.perf_counter()
        return CallResult(
            error=last_error,
            attempts=attempts,
            started=started,
            ended=ended,
            latency_ms=(ended - started) * 1000.0,
        )

    async def _attempt(self, kwargs: dict[str, Any], stream: bool) -> "_AttemptResult":
        attempt_start = time.perf_counter()
        response = await self._client.chat.completions.create(**kwargs)
        if stream:
            return await self._consume_stream(response, attempt_start)
        choice = response.choices[0]
        tokens: list[str] = []
        logprobs: list[float] = []
        content = getattr(choice.logprobs, "content", None) if choice.logprobs else None
        for item in content or []:
            tokens.append(item.token)
            logprobs.append(float(item.logprob))
        usage = response.usage
        return _AttemptResult(
            text=choice.message.content or "",
            tokens=tokens,
            token_logprobs=logprobs,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            started_attempt=attempt_start,
        )

    async def _consume_stream(self, response: Any, attempt_start: float) -> "_AttemptResult":
        parts: list[str] = []
        tokens: list[str] = []
        logprobs: list[float] = []
        ttft_ms: float | None = None
        prompt_tokens = completion_tokens = 0
        async for chunk in response:
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
            for choice in chunk.choices:
                piece = choice.delta.content if choice.delta else None
                if piece:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - attempt_start) * 1000.0
                    parts.append(piece)
                content = getattr(choice.logprobs, "content", None) if choice.logprobs else None
                for item in content or []:
                    tokens.append(item.token)
                    logprobs.append(float(item.logprob))
        return _AttemptResult(
            text="".join(parts),
            tokens=tokens,
            token_logprobs=logprobs,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            started_attempt=attempt_start,
        )


@dataclass
class _AttemptResult(CallResult):
    started_attempt: float = 0.0

    def to_call_result(self) -> CallResult:
        return CallResult(
            text=self.text,
            tokens=self.tokens,
            token_logprobs=self.token_logprobs,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            latency_ms=self.latency_ms,
            ttft_ms=self.ttft_ms,
            error=self.error,
            attempts=self.attempts,
            started=self.started,
            ended=self.ended,
        )
