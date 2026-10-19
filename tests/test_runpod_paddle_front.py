"""runpod/paddle_front.py: one public port for vLLM and the PaddleX pipelines (design D3).

The upstreams are real HTTP servers on loopback ports (stdlib, one thread each) so the front's
httpx client, streaming and health fan-out run exactly as in the worker.
"""

import importlib.util
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("paddle_front", ROOT / "runpod" / "paddle_front.py")
paddle_front = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paddle_front)  # type: ignore[union-attr]

SECRET = "ACCT-0000-SECRET-BODY-MARKER"


class _Fake(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, name: str):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.name = name
        self.healthy = True
        self.bodies: list[bytes] = []
        threading.Thread(target=self.serve_forever, args=(0.02,), daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    server: _Fake

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, status: int, payload: dict | None = None) -> None:
        body = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send(200 if self.server.healthy else 503, {})
        elif path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": "PaddleOCR-VL-1.6-0.9B"}]})
        elif path == "/version":
            self._send(200, {"version": "0.22.1"})
        else:
            self._send(404, {"path": self.path})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.server.bodies.append(body)
        if self.path == "/v1/chat/completions":
            chunks = [b'data: {"n": 1}\n\n', b'data: {"n": 2}\n\n', b"data: [DONE]\n\n"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for c in chunks:
                self.wfile.write(f"{len(c):x}\r\n".encode() + c + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        elif self.path == "/layout-parsing":
            self._send(200, {"server": self.server.name, "bytes": len(body)})
        else:
            self._send(404, {"path": self.path})


@pytest.fixture
def upstreams():
    vllm = _Fake("vllm")
    pipes = [_Fake("p0"), _Fake("p1")]
    yield vllm, pipes
    for s in (vllm, *pipes):
        s.shutdown()
        s.server_close()


@pytest.fixture
def client(upstreams):
    vllm, pipes = upstreams
    app = paddle_front.create_app(vllm.url, [p.url for p in pipes])
    with TestClient(app) as c:
        yield c


def test_layout_parsing_round_robins_over_pipelines(client, upstreams):
    _, pipes = upstreams
    served = [
        client.post("/layout-parsing", json={"file": "x", "fileType": 1}).json()["server"]
        for _ in range(4)
    ]
    assert served == ["p0", "p1", "p0", "p1"]
    assert all(len(p.bodies) == 2 for p in pipes)


def test_layout_parsing_forwards_body_unchanged(client, upstreams):
    _, pipes = upstreams
    payload = {"file": "data:image/png;base64,AAAA", "fileType": 1, "visualize": False}
    r = client.post("/layout-parsing", json=payload)
    assert r.status_code == 200
    assert json.loads(pipes[0].bodies[0]) == payload


def test_v1_models_proxied_to_vllm(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "PaddleOCR-VL-1.6-0.9B"


def test_version_proxied_to_vllm(client):
    assert client.get("/version").json() == {"version": "0.22.1"}


def test_v1_post_streams_response_body(client, upstreams):
    vllm, _ = upstreams
    with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    assert text.endswith("data: [DONE]\n\n")
    assert json.loads(vllm.bodies[0]) == {"stream": True}


# Runpod's load balancer polls /ping (live worker logs, 2026-10-19); /health is the same.
READINESS = pytest.mark.parametrize("path", ["/ping", "/health"])


@READINESS
def test_health_200_when_vllm_and_all_pipelines_healthy(client, path):
    assert client.get(path).status_code == 200


@READINESS
def test_health_204_while_one_pipeline_down(client, upstreams, path):
    _, pipes = upstreams
    pipes[1].healthy = False
    r = client.get(path)
    assert r.status_code == 204
    assert r.content == b""


@READINESS
def test_health_204_while_vllm_down(client, upstreams, path):
    vllm, _ = upstreams
    vllm.healthy = False
    assert client.get(path).status_code == 204


def test_health_204_when_pipeline_unreachable(upstreams):
    vllm, pipes = upstreams
    dead = _Fake("dead")
    dead_url = dead.url
    dead.shutdown()
    dead.server_close()
    app = paddle_front.create_app(vllm.url, [pipes[0].url, dead_url])
    with TestClient(app) as c:
        assert c.get("/health").status_code == 204


def test_upstream_down_gives_502_not_a_crash(upstreams):
    vllm, _ = upstreams
    dead = _Fake("dead")
    dead_url = dead.url
    dead.shutdown()
    dead.server_close()
    app = paddle_front.create_app(vllm.url, [dead_url])
    with TestClient(app) as c:
        assert c.post("/layout-parsing", json={"file": SECRET}).status_code == 502


def test_access_log_has_method_path_status_size_ms_only(client, caplog):
    caplog.set_level(logging.INFO, logger=paddle_front.LOGGER_NAME)
    r = client.get("/v1/models?secret=" + SECRET)
    lines = [rec.getMessage() for rec in caplog.records if rec.name == paddle_front.LOGGER_NAME]
    assert len(lines) == 1
    method, path, status, size, ms = lines[0].split()
    assert (method, path, status) == ("GET", "/v1/models", "200")
    assert int(size) == len(r.content)
    assert ms.endswith("ms") and float(ms[:-2]) >= 0


def test_request_body_never_logged(client, upstreams, caplog, capfd):
    caplog.set_level(logging.DEBUG)
    client.post("/layout-parsing", json={"file": SECRET})
    client.post("/v1/chat/completions", json={"messages": [{"content": SECRET}]})
    vllm, pipes = upstreams
    pipes[0].shutdown()
    pipes[0].server_close()
    client.post("/layout-parsing", json={"file": SECRET})  # p1
    client.post("/layout-parsing", json={"file": SECRET})  # p0, now down -> 502
    out, err = capfd.readouterr()
    assert SECRET not in caplog.text
    assert SECRET not in out + err


def test_pipeline_urls_from_env():
    assert paddle_front.pipeline_urls({"PADDLE_PIPELINES": "3"}) == [
        "http://127.0.0.1:8081",
        "http://127.0.0.1:8082",
        "http://127.0.0.1:8083",
    ]
    assert len(paddle_front.pipeline_urls({})) == 2
