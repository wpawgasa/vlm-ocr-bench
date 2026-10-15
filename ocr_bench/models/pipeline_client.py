"""Async client for a vendor document-parsing pipeline served over HTTP (design D15).

Today: PaddleX's official serving app for PaddleOCR-VL (`POST /layout-parsing`). The request
carries the page as base64 PNG (`fileType` 1 = image) plus the pipeline options from the
model config; the response envelope is `{logId, errorCode, errorMsg, result}` with
`result.layoutParsingResults[i] = {prunedResult, markdown: {text, images}, ...}`.

Same contract as `VllmClient.chat`: one retry on a timeout, a transport error or a 5xx; a
4xx is not retried; `layout` never raises. The pipeline returns no tokens or logprobs.
"""

import asyncio
import base64
import io
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from PIL import Image

from ocr_bench.config import PipelineEndpoint
from ocr_bench.models.vllm_client import (
    HEALTH_TIMEOUT_S,
    auth_headers,
    require_tls_for_runpod,
    wait_until_healthy,
)


@dataclass
class PipelineResult:
    text: str = ""  # JSON of the page's layout result (images stripped), for the parser
    latency_ms: float = 0.0
    error: str | None = None
    attempts: int = 0
    started: float = 0.0
    ended: float = 0.0


def image_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def page_result(envelope: dict[str, Any]) -> dict[str, Any]:
    """The first page of a `/layout-parsing` response, without base64 image payloads.

    Raises `ValueError` on a pipeline error or an unexpected shape."""
    if envelope.get("errorCode", 0) != 0:
        raise ValueError(f"pipeline error {envelope.get('errorCode')}: {envelope.get('errorMsg')}")
    pages = (envelope.get("result") or {}).get("layoutParsingResults")
    if not isinstance(pages, list) or not pages:
        raise ValueError("pipeline response has no layoutParsingResults")
    page = pages[0]
    markdown = page.get("markdown") or {}
    return {
        "prunedResult": page.get("prunedResult") or {},
        "markdown": {"text": markdown.get("text", "")},
    }


class _NoRetry(Exception):
    pass


class PipelineClient:
    def __init__(
        self,
        base_url: str,
        endpoint: PipelineEndpoint,
        *,
        http_client: httpx.AsyncClient | None = None,
        model_label: str = "pipeline",
        api_key: str | None = None,
        wait_ready_s: float = 0.0,
        base_url_env: str | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        require_tls_for_runpod(base_url, base_url_env)
        self.api_key = api_key
        self.wait_ready_s = wait_ready_s
        self._sleep = sleep
        self._clock = clock
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.model_label = model_label
        self._http_client = http_client

    async def _post(self, http: httpx.AsyncClient, body: dict[str, Any]) -> httpx.Response:
        return await http.post(
            self.base_url + self.endpoint.path,
            json=body,
            headers=auth_headers(self.api_key),
            timeout=self.endpoint.timeout_s,
        )

    async def _get(self, url: str) -> httpx.Response:
        headers = auth_headers(self.api_key)
        if self._http_client is not None:
            return await self._http_client.get(url, headers=headers, timeout=HEALTH_TIMEOUT_S)
        async with httpx.AsyncClient() as http:
            return await http.get(url, headers=headers, timeout=HEALTH_TIMEOUT_S)

    async def health(self) -> None:
        url = self.base_url + self.endpoint.health_path
        await wait_until_healthy(
            lambda: self._get(url),
            model=self.model_label,
            base_url=self.base_url,
            url=url,
            wait_ready_s=self.wait_ready_s,
            sleep=self._sleep,
            clock=self._clock,
        )

    async def layout(self, image: Image.Image) -> PipelineResult:
        body = {**self.endpoint.request, "file": image_b64(image), "fileType": 1}
        started = time.perf_counter()
        last_error = "no attempt made"
        attempts = 0
        for _ in range(2):
            attempts += 1
            try:
                if self._http_client is not None:
                    response = await self._post(self._http_client, body)
                else:
                    async with httpx.AsyncClient() as http:
                        response = await self._post(http, body)
                if response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    continue
                if response.status_code >= 400:
                    raise _NoRetry(f"HTTP {response.status_code}: {response.text[:200]}")
                page = page_result(response.json())
            except asyncio.CancelledError:
                raise
            except _NoRetry as exc:
                last_error = str(exc)
                break
            except ValueError as exc:  # pipeline-reported error or bad shape: not transient
                last_error = str(exc)
                break
            except Exception as exc:  # timeouts, transport errors
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            ended = time.perf_counter()
            return PipelineResult(
                text=json.dumps(page, ensure_ascii=False),
                latency_ms=(ended - started) * 1000.0,
                attempts=attempts,
                started=started,
                ended=ended,
            )
        ended = time.perf_counter()
        return PipelineResult(
            error=last_error,
            latency_ms=(ended - started) * 1000.0,
            attempts=attempts,
            started=started,
            ended=ended,
        )
