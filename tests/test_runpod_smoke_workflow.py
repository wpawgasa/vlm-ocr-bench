"""Static checks of .github/workflows/runpod-smoke.yml (no network)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "runpod-smoke.yml"
MODELS = ("dotsocr", "ovisocr2", "paddleocr_vl", "teleocr", "typhoon_ocr15")


def _load() -> dict:
    doc = yaml.safe_load(WORKFLOW.read_text())
    doc["on"] = doc.pop(True, doc.get("on"))  # YAML 1.1 reads the key `on` as True
    return doc


def test_triggers_are_weekday_business_hours_cron_and_dispatch():
    on = _load()["on"]
    assert "workflow_dispatch" in on
    assert "models" in on["workflow_dispatch"]["inputs"]
    (entry,) = on["schedule"]
    # 08:00-20:00 Asia/Bangkok (UTC+7) is 01:00-13:00 UTC, Monday-Friday, every 30 minutes.
    assert entry["cron"] == "0,30 1-13 * * 1-5"


def test_runs_smoke_per_model_and_fails_on_nonzero_exit():
    doc = _load()
    job = doc["jobs"]["smoke"]
    assert job["strategy"]["fail-fast"] is False
    text = WORKFLOW.read_text()
    assert "ocrbench runpod smoke" in text
    assert "continue-on-error" not in text


def test_models_and_secrets_cover_every_model():
    text = WORKFLOW.read_text()
    assert "RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}" in text
    for var in (
        "DOTSOCR",
        "OVISOCR2",
        "PADDLEOCR_VL",
        "TELEOCR",
        "TYPHOON",
    ):
        assert f"secrets.{var}_BASE_URL" in text
        assert f"secrets.{var}_API_KEY" in text
    assert "secrets.PADDLE_PIPELINE_URL" in text
    for m in MODELS:
        assert m in text
