"""PaddleOCR-VL worker front (design D3): one public port for vLLM and the PaddleX pipelines.

- `POST /layout-parsing` goes round-robin to the pipeline processes on 127.0.0.1:8081...;
- `/v1/*` and `/version` are reverse-proxied to the loopback vLLM, bodies streamed both ways;
- `GET /health` is 200 only when vLLM and every pipeline answer 200 on /health, else 204, so
  Runpod's load balancer sees one readiness signal.

The access log is one line per request, `METHOD PATH STATUS BYTES MSms`: no query string, no
headers, no bodies (they may hold client documents).
"""

import asyncio
import itertools
import logging
import os
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

LOGGER_NAME = "paddle_front"
log = logging.getLogger(LOGGER_NAME)

VLLM_URL = "http://127.0.0.1:8000"
PIPELINE_BASE_PORT = 8081
# Runpod cuts a load-balancer request at 5.5 min; give the upstream a little less.
_TIMEOUT = httpx.Timeout(320.0, connect=5.0)
_HEALTH_TIMEOUT = httpx.Timeout(3.0)
# Hop-by-hop headers (RFC 9110 7.6.1) plus Host, which httpx sets from the upstream URL.
_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
}


def pipeline_urls(env: Mapping[str, str]) -> list[str]:
    n = int(env.get("PADDLE_PIPELINES", "2"))
    return [f"http://127.0.0.1:{PIPELINE_BASE_PORT + i}" for i in range(n)]


def _headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP}


class AccessLog:
    """ASGI middleware: method, path, status, response bytes, elapsed ms; nothing else."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        t0 = time.perf_counter()
        status, size = 500, 0

        async def _send(message):
            nonlocal status, size
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                size += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            ms = (time.perf_counter() - t0) * 1000
            log.info("%s %s %d %d %.1fms", scope["method"], scope["path"], status, size, ms)


def create_app(vllm_url: str, pipelines: list[str]) -> FastAPI:
    if not pipelines:
        raise ValueError("at least one pipeline URL is required")
    rotation = itertools.cycle(pipelines)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(AccessLog)

    async def proxy(request: Request, base: str) -> Response:
        client: httpx.AsyncClient = request.app.state.client
        url = base + request.url.path
        if request.url.query:
            url += "?" + request.url.query
        upstream = client.build_request(
            request.method, url, headers=_headers(request.headers), content=request.stream()
        )
        try:
            resp = await client.send(upstream, stream=True)
        except httpx.HTTPError as e:
            # Exception class only: messages can echo the URL, never the body, but keep it terse.
            log.warning("upstream %s unavailable: %s", base, type(e).__name__)
            return JSONResponse({"error": "upstream unavailable"}, status_code=502)
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=_headers(resp.headers),
            background=BackgroundTask(resp.aclose),
        )

    @app.post("/layout-parsing")
    async def layout_parsing(request: Request) -> Response:
        return await proxy(request, next(rotation))

    @app.api_route("/v1/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def v1(request: Request, rest: str) -> Response:
        return await proxy(request, vllm_url)

    @app.get("/version")
    async def version(request: Request) -> Response:
        return await proxy(request, vllm_url)

    @app.get("/health")
    async def health(request: Request) -> Response:
        client: httpx.AsyncClient = request.app.state.client

        async def ok(base: str) -> bool:
            try:
                r = await client.get(base + "/health", timeout=_HEALTH_TIMEOUT)
            except httpx.HTTPError:
                return False
            return r.status_code == 200

        results = await asyncio.gather(*(ok(u) for u in (vllm_url, *pipelines)))
        return Response(status_code=200 if all(results) else 204)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    app = create_app(os.environ.get("VLLM_URL", VLLM_URL), pipeline_urls(os.environ))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
