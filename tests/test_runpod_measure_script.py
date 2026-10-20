"""The pure helpers of scripts/runpod_measure.py (task 6.2)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "runpod_measure.py"
_spec = importlib.util.spec_from_file_location("runpod_measure", _PATH)
measure = importlib.util.module_from_spec(_spec)
sys.modules["runpod_measure"] = measure
_spec.loader.exec_module(measure)


def test_percentile_interpolates_linearly():
    xs = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert measure.percentile(xs, 50) == 3.0
    assert measure.percentile(xs, 95) == pytest.approx(4.8)
    assert measure.percentile([7.0], 95) == 7.0


def test_worker_seconds_holds_each_sample_until_the_next():
    samples = [(0.0, 1), (10.0, 2), (15.0, 0)]
    assert measure.worker_seconds(samples, end=20.0) == 10 * 1 + 5 * 2 + 5 * 0


def test_cost_per_page_uses_hourly_price():
    assert measure.cost_per_page(3600.0, 60, 0.69) == pytest.approx(0.69 / 60)


def test_parity_rows_are_sixty_clean_rows_with_local_images():
    if not measure.RunPaths.for_run("2026-09-22-a").manifest.exists():
        pytest.skip("run 2026-09-22-a not pulled (dvc pull)")
    rows = measure.parity_rows("2026-09-22-a")
    assert len(rows) == 60
    assert len({r.sample_id for r in rows}) == 60
    root = measure.RunPaths.for_run("2026-09-22-a").root
    missing = [r.image_path for r in rows if not (root / r.image_path).exists()]
    if missing:
        pytest.skip(f"{len(missing)} parity images not pulled (dvc pull)")


def test_billed_workers_counts_running_and_initializing_capped_at_workers_max():
    class FakeAPI:
        def _call(self, method, path):
            return {"summary": {"idle": 2, "initializing": 4, "running": 1, "throttled": 3}}

    assert measure.billed_workers(FakeAPI(), "ep") == 5
    assert measure.billed_workers(FakeAPI(), "ep", cap=2) == 2


def _pred(n_tokens, error=None):
    from ocr_bench.schemas import RawPrediction

    return type("P", (), {"raw": RawPrediction(tokens=["x"] * n_tokens, error=error)})()


def test_loop_page_is_one_whose_output_reached_max_tokens():
    assert measure.loop_page(_pred(8192), 8192)
    assert not measure.loop_page(_pred(8191), 8192)


def test_slo_result_judges_cold_start_and_non_loop_page_latency():
    from ocr_bench.runpod.config import SloConfig

    slo = SloConfig(page_p50_s=10, page_p95_s=30, cold_start_p95_s=120)
    assert measure.slo_result(115.0, 9.4, 22.6, slo) == "met"
    assert measure.slo_result(121.0, 9.4, 22.6, slo) == "missed"
    assert measure.slo_result(115.0, 10.5, 22.6, slo) == "missed"
    assert measure.slo_result(115.0, 9.4, 31.0, slo) == "missed"
    assert measure.slo_result(None, 9.4, 22.6, slo) == "missed"
