"""Task 6.2: measure a Runpod endpoint's cold start, page latency and cost per page.

    uv run python scripts/runpod_measure.py dotsocr --out data/runs/runpod-measure-<date>

1. `--cold-starts` times (default 5): wait until the endpoint has no running or initializing
   worker (scale to zero), then run the smoke check (`ocrbench runpod smoke`), which times
   health-ready (the cold start) and one page.
2. Right after the last cold start, while a worker is warm: the 60-page parity set
   (`configs/runpod/parity_samples.v1.yaml`, clean rows of its run) through the harness
   adapter at `client.max_concurrency`, polling the worker count every few seconds.

Writes `<out>/<model>.json` (all samples and the summary) and `<out>/<model>.predictions.jsonl`.
Cost per page = billed worker-seconds during the parity run x the pool's serverless $/h / pages;
running and initializing workers count as billed. Needs `RUNPOD_API_KEY` (management),
`<MODEL>_BASE_URL`, `<MODEL>_API_KEY` and, for Paddle, `PADDLE_PIPELINE_URL`.
"""

import argparse
import asyncio
import json
import math
import os
import subprocess
import time
from datetime import date
from pathlib import Path

import httpx

from ocr_bench.config import load_model_config
from ocr_bench.jsonl import read_rows
from ocr_bench.models.registry import build_model
from ocr_bench.models.runner import infer_model
from ocr_bench.paths import RunPaths
from ocr_bench.runpod import state as runpod_state
from ocr_bench.runpod.api import RunpodAPI
from ocr_bench.runpod.config import load_runpod_config
from ocr_bench.runpod.deploy import endpoint_name
from ocr_bench.runpod.parity_set import load_parity_set
from ocr_bench.runpod.releases import latest
from ocr_bench.runpod.render import gpu_selection
from ocr_bench.runpod.smoke import run_smoke
from ocr_bench.schemas import Condition, ManifestRow, PredictionRow

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "runpod"
BILLED = ("running", "initializing")


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in 0..100) of a non-empty list."""
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def worker_seconds(samples: list[tuple[float, int]], end: float) -> float:
    """Integrate (time, billed workers) samples, each held until the next sample or `end`."""
    total = 0.0
    for (t, n), nxt in zip(samples, [s[0] for s in samples[1:]] + [end], strict=True):
        total += n * (nxt - t)
    return total


def cost_per_page(billed_s: float, pages: int, usd_per_hour: float) -> float:
    return billed_s * usd_per_hour / 3600 / pages


def billed_workers(api: RunpodAPI, endpoint_id: str) -> int:
    summary = api._call("GET", f"/serverless/{endpoint_id}/workers").get("summary", {})
    return sum(int(summary.get(k, 0)) for k in BILLED)


def serverless_price(api: RunpodAPI, pools: list[str], excluded: list[str]) -> float:
    """The highest serverless $/h among the endpoint's GPU types (upper bound)."""
    prices = [
        g["price"]["serverless"]
        for g in api.gpu_catalog()
        if g.get("pool") in pools and g.get("id") not in excluded
    ]
    return max(prices)


def wait_scaled_to_zero(api: RunpodAPI, endpoint_id: str, budget_s: float) -> float:
    started = time.monotonic()
    while billed_workers(api, endpoint_id):
        if time.monotonic() - started > budget_s:
            raise SystemExit(f"endpoint still has billed workers after {budget_s:.0f}s")
        time.sleep(10)
    return time.monotonic() - started


def parity_rows(run_id: str) -> list[ManifestRow]:
    rows = list(read_rows(RunPaths.for_run(run_id).manifest, ManifestRow))
    ps = load_parity_set(CONFIGS / "parity_samples.v1.yaml", rows)
    wanted = set(ps.sample_ids)
    return [r for r in rows if r.sample_id in wanted and r.condition == Condition.clean]


async def measure(model: str, out: Path, cold_starts: int, scale_wait_s: float) -> dict:
    rcfg = load_runpod_config(CONFIGS / f"{model}.yaml")
    mcfg = load_model_config(ROOT / "configs" / "models" / f"{model}.yaml", model)
    os.environ["OCRBENCH_ENDPOINT_WAIT_S"] = str(rcfg.client.cold_start_wait_s)
    api = RunpodAPI.from_env()
    endpoint_id = runpod_state.load_state()["endpoints"][endpoint_name(rcfg)]["id"]
    release = latest(model) or {}
    gpu = gpu_selection(rcfg, api.gpu_catalog())
    usd_h = serverless_price(api, gpu.pools, gpu.excluded_types)

    colds = []
    for i in range(cold_starts):
        waited = wait_scaled_to_zero(api, endpoint_id, scale_wait_s)
        built = build_model(mcfg)
        async with httpx.AsyncClient() as http:
            rep = await run_smoke(
                rcfg,
                mcfg,
                built,
                base_url=built.client.base_url,
                api_key=getattr(built.client, "api_key", None),
                http=http,
            )
        colds.append(
            {
                "cold_start_s": round(rep.cold_start_s, 2),
                "page_s": round(rep.request_s, 2),
                "layout_s": rep.layout_s and round(rep.layout_s, 2),
                "scale_down_wait_s": round(waited, 1),
                "failures": rep.failures,
            }
        )
        print(f"{model} cold start {i + 1}/{cold_starts}: {colds[-1]}", flush=True)

    rows = parity_rows("2026-09-22-a")
    preds_path = out / f"{model}.predictions.jsonl"
    preds_path.unlink(missing_ok=True)
    built = build_model(mcfg)
    samples: list[tuple[float, int]] = []
    done = asyncio.Event()

    async def poll() -> None:
        while not done.is_set():
            n = await asyncio.to_thread(billed_workers, api, endpoint_id)
            samples.append((time.monotonic(), n))
            try:
                await asyncio.wait_for(done.wait(), timeout=5)
            except TimeoutError:
                pass

    poller = asyncio.create_task(poll())
    await asyncio.sleep(0.5)
    t0 = time.monotonic()
    await infer_model(
        built, rows, RunPaths.for_run("2026-09-22-a").root, preds_path, rcfg.client.max_concurrency
    )
    t1 = time.monotonic()
    done.set()
    await poller
    preds = list(read_rows(preds_path, PredictionRow))
    lat = [p.latency_ms / 1000 for p in preds if p.raw.error is None]
    billed_s = worker_seconds([s for s in samples if s[0] >= t0] or samples[:1], t1)

    cold = [c["cold_start_s"] for c in colds if not c["failures"]]
    summary = {
        "model": model,
        "date": date.today().isoformat(),
        "image_digest": release.get("image_digest"),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip(),
        "cold_start_s": {
            "n": len(cold),
            "p50": round(percentile(cold, 50), 1) if cold else None,
            "p95": round(percentile(cold, 95), 1) if cold else None,
        },
        "parity_run": {
            "pages": len(rows),
            "errors": sum(p.raw.error is not None for p in preds),
            "concurrency": rcfg.client.max_concurrency,
            "wall_s": round(t1 - t0, 1),
            "page_p50_s": round(percentile(lat, 50), 2) if lat else None,
            "page_p95_s": round(percentile(lat, 95), 2) if lat else None,
            "max_billed_workers": max((n for _, n in samples), default=0),
            "billed_worker_s": round(billed_s, 1),
            "usd_per_hour": usd_h,
            "usd_per_page": round(cost_per_page(billed_s, len(rows), usd_h), 5),
        },
        "slo": {
            "page_p50_s": rcfg.slo.page_p50_s,
            "page_p95_s": rcfg.slo.page_p95_s,
            "cold_start_p95_s": rcfg.slo.cold_start_p95_s,
        },
        "cold_starts": colds,
    }
    met = (
        bool(cold)
        and lat
        and summary["cold_start_s"]["p95"] <= rcfg.slo.cold_start_p95_s
        and summary["parity_run"]["page_p50_s"] <= rcfg.slo.page_p50_s
        and summary["parity_run"]["page_p95_s"] <= rcfg.slo.page_p95_s
    )
    summary["slo_result"] = "met" if met else "missed"
    (out / f"{model}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cold-starts", type=int, default=5)
    ap.add_argument("--scale-wait-s", type=float, default=900)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s = asyncio.run(measure(args.model, args.out, args.cold_starts, args.scale_wait_s))
    print(json.dumps({k: s[k] for k in ("cold_start_s", "parity_run", "slo_result")}, indent=2))


if __name__ == "__main__":
    main()
