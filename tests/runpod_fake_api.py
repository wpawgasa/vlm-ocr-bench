"""A recorded fake of the Runpod REST v2 API (templates, serverless endpoints) for deploy tests.

Served through `httpx.MockTransport`, so the real `RunpodAPI` HTTP code runs. Shapes follow
the live v2 API (checked 2026-10-19): `templateId` is applied once (the template's container
settings are copied onto the endpoint and no link is kept), kinds are `QUEUE`/`LOAD_BALANCER`,
and `GET /v2/catalog/gpus` lists each GPU type with its serverless pool.
"""

import itertools
import json

import httpx

BLACKWELL_MIG = "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"
CATALOG = [
    {"id": "NVIDIA GeForce RTX 3090", "pool": "AMPERE_24", "memory": 24},
    {"id": "NVIDIA L4", "pool": "AMPERE_24", "memory": 24},
    {"id": "NVIDIA RTX A5000", "pool": "AMPERE_24", "memory": 24},
    {"id": BLACKWELL_MIG, "pool": "AMPERE_24", "memory": 24},
    {"id": "NVIDIA A40", "pool": "AMPERE_48", "memory": 48},
    {"id": "NVIDIA GeForce RTX 3080", "pool": None, "memory": 10},
]
_CONTAINER_KEYS = ("image", "args", "disk", "ports", "env", "registry")


class FakeRunpod:
    def __init__(self, templates=None, endpoints=None, list_wrapper=None, catalog=None):
        self.templates: list[dict] = list(templates or [])
        self.catalog: list[dict] = list(CATALOG if catalog is None else catalog)
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
        if path == "/v2/catalog/gpus" and request.method == "GET":
            return httpx.Response(200, json={"gpus": self.catalog})  # the live wrapper
        if path == "/v2/serverless" and request.method == "POST":
            ep = {"id": f"ep{next(self._ids)}", **self._apply_template(body)}
            self.endpoints.append(ep)
            return httpx.Response(200, json=ep)
        if path.startswith("/v2/serverless/") and request.method == "PATCH":
            ep_id = path.rsplit("/", 1)[1]
            for ep in self.endpoints:
                if ep["id"] == ep_id:
                    ep.update(self._apply_template(body))
                    return httpx.Response(200, json=ep)
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(404, json={"error": f"no route {request.method} {path}"})

    def _apply_template(self, body: dict) -> dict:
        """Copy the template's container settings onto the body; v2 keeps no template link."""
        body = dict(body)
        tmpl_id = body.pop("templateId", None)
        if tmpl_id is not None:
            tmpl = next(t for t in self.templates if t["id"] == tmpl_id)
            for key in _CONTAINER_KEYS:
                if key in tmpl:
                    body.setdefault(key, tmpl[key])
        return body
