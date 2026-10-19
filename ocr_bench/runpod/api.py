"""A thin client for the Runpod REST v2 management API (design D4).

Covers what `deploy` needs: list/create templates and list/create/patch serverless endpoints
under `https://api.runpod.io/v2`, with the management key from `RUNPOD_API_KEY` as a bearer
token. The v2 list responses are unverified, so a bare list and `{"data"|"items": [...]}` are
both accepted. Errors name the status and path, never the key.
"""

import os
from typing import Any

import httpx

BASE_URL = "https://api.runpod.io/v2"
KEY_ENV = "RUNPOD_API_KEY"


class RunpodAPIError(RuntimeError):
    """A Runpod API call failed, or no management key is configured."""


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "endpoints", "templates"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise RunpodAPIError(f"unexpected list response shape: {type(payload).__name__}")


class RunpodAPI:
    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
            timeout=timeout,
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "RunpodAPI":
        key = os.environ.get(KEY_ENV)
        if not key:
            raise RunpodAPIError(f"{KEY_ENV} is not set (a Runpod management key)")
        return cls(key, **kwargs)

    @property
    def base_url(self) -> httpx.URL:
        return self._http.base_url.copy_with(path=self._http.base_url.path.rstrip("/"))

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        try:
            resp = self._http.request(method, path, json=body)
        except httpx.HTTPError as e:
            raise RunpodAPIError(f"{method} {path}: {type(e).__name__}") from e
        if resp.status_code >= 400:
            raise RunpodAPIError(f"{method} {path}: HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json() if resp.content else None

    def list_templates(self) -> list[dict[str, Any]]:
        return _items(self._call("GET", "/templates"))

    def create_template(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/templates", body)

    def gpu_catalog(self) -> list[dict[str, Any]]:
        """GPU types with their serverless `pool` (null when not offered on serverless)."""
        return _items(self._call("GET", "/catalog/gpus"))

    def list_endpoints(self) -> list[dict[str, Any]]:
        return _items(self._call("GET", "/serverless"))

    def create_endpoint(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/serverless", body)

    def patch_endpoint(self, endpoint_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("PATCH", f"/serverless/{endpoint_id}", body)
