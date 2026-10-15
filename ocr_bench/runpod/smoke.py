"""`ocrbench runpod smoke <model>` (design D5): is the endpoint up, serving the right model, and
fast enough, and does it reject unauthenticated callers? The page is a public fixture."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from PIL import Image

from ocr_bench.config import ModelConfig
from ocr_bench.models.base import PlannedModel
from ocr_bench.models.vllm_client import EndpointUnavailable, auth_headers
from ocr_bench.runpod.config import RunpodModelConfig
from ocr_bench.schemas import Condition, GtKind, ManifestRow, Source, Task

SMOKE_IMAGE = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "runpod" / "smoke_page.png"
)
_HTTP_TIMEOUT_S = 30.0


@dataclass
class SmokeReport:
    cold_start_s: float = 0.0
    request_s: float = 0.0
    layout_s: float | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def models_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root + "/models" if root.endswith("/v1") else root + "/v1/models"


async def run_smoke(
    rcfg: RunpodModelConfig,
    mcfg: ModelConfig,
    model: PlannedModel,
    *,
    base_url: str,
    api_key: str | None,
    http: httpx.AsyncClient,
    clock: Callable[[], float] = time.monotonic,
    image_path: Path = SMOKE_IMAGE,
) -> SmokeReport:
    report = SmokeReport()
    url = models_url(base_url)

    started = clock()
    try:
        await model.health()
    except EndpointUnavailable as exc:
        report.failures.append(f"health: {exc}")
        return report
    report.cold_start_s = clock() - started

    try:
        response = await http.get(url, headers=auth_headers(api_key), timeout=_HTTP_TIMEOUT_S)
        response.raise_for_status()
        served = [m.get("id") for m in response.json().get("data", [])]
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        report.failures.append(f"GET {url}: {exc}")
        return report
    if mcfg.served_model_name not in served:
        report.failures.append(
            f"endpoint serves {served}, expected served_model_name {mcfg.served_model_name!r}"
        )
        return report

    image = Image.open(image_path).convert("RGB")
    row = ManifestRow(
        sample_id="runpod-smoke",
        source=Source.thaiocrbench,
        task=Task.ocr_fullpage,
        condition=Condition.clean,
        image_path=image_path.name,
        gt_kind=GtKind.none,
    )
    plan = mcfg.plan_for(row.task, row.subtask)
    started = clock()
    pred = await model.predict_with_plan(row, image, plan)
    report.request_s = clock() - started
    if pred.raw.error is not None:
        report.failures.append(f"ocr_fullpage request failed: {pred.raw.error}")
    elif not pred.raw.text.strip():
        report.failures.append("ocr_fullpage request returned no text")
    if report.request_s > rcfg.slo.page_p95_s:
        report.failures.append(
            f"page latency {report.request_s:.1f}s exceeds slo.page_p95_s {rcfg.slo.page_p95_s:g}s"
        )

    if model.pipeline is not None:
        started = clock()
        result = await model.pipeline.layout(image)
        report.layout_s = clock() - started
        if result.error is not None:
            report.failures.append(f"/layout-parsing request failed: {result.error}")

    try:
        bare = await http.get(url, timeout=_HTTP_TIMEOUT_S)
    except httpx.HTTPError as exc:
        report.failures.append(f"unauthenticated GET {url}: {exc}")
    else:
        if bare.status_code not in (401, 403):
            report.failures.append(
                f"GET {url} without a key returned {bare.status_code}, expected 401 or 403"
            )
    return report
