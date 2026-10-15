"""A recorded fake of the Runpod REST v2 API (templates, serverless endpoints) for deploy tests.

Served through `httpx.MockTransport`, so the real `RunpodAPI` HTTP code runs. Response shapes
follow the request bodies `render.py` sends (the v2 reference pages are unverified, D4).
"""

import itertools
import json

import httpx


class FakeRunpod:
    def __init__(self, templates=None, endpoints=None, list_wrapper=None):
        self.templates: list[dict] = list(templates or [])
        self.endpoints: list[dict] = list(endpoints or [])
        self.calls: list[tuple[str, str, dict | None]] = []
        self.auth: list[str | None] = []
        self.list_wrapper = list_wrapper  # e.g. "data" to answer lists as {"data": [...]}
        self._ids = itertools.count(1)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    @property
    def writes(self) -> list[tuple[str, str, dict | None]]:
        return [c for c in self.calls if c[0] != "GET"]

    def _list(self, items: list[dict]) -> httpx.Response:
        return httpx.Response(200, json={self.list_wrapper: items} if self.list_wrapper else items)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.calls.append((request.method, path, body))
        self.auth.append(request.headers.get("authorization"))
        if path == "/v2/templates" and request.method == "GET":
            return self._list(self.templates)
        if path == "/v2/templates" and request.method == "POST":
            tmpl = {"id": f"tmpl{next(self._ids)}", **body}
            self.templates.append(tmpl)
            return httpx.Response(200, json=tmpl)
        if path == "/v2/serverless" and request.method == "GET":
            return self._list(self.endpoints)
        if path == "/v2/serverless" and request.method == "POST":
            ep = {"id": f"ep{next(self._ids)}", **body}
            self.endpoints.append(ep)
            return httpx.Response(200, json=ep)
        if path.startswith("/v2/serverless/") and request.method == "PATCH":
            ep_id = path.rsplit("/", 1)[1]
            for ep in self.endpoints:
                if ep["id"] == ep_id:
                    ep.update(body)
                    return httpx.Response(200, json=ep)
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(404, json={"error": f"no route {request.method} {path}"})
