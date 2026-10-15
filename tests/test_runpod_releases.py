import pytest
import yaml

from ocr_bench import cli
from ocr_bench.config import ModelConfig
from ocr_bench.paths import RunPaths
from ocr_bench.runpod import releases
from ocr_bench.runpod.releases import ReleaseRecordError


def _record(model="dotsocr", vllm="0.11.0", date="2026-10-01"):
    return {"model": model, "date": date, "vllm_version": vllm, "parity": "pending"}


def test_append_then_latest_returns_newest_record_for_model(tmp_path):
    path = tmp_path / "releases.yaml"
    releases.append(_record(vllm="0.11.0"), path)
    releases.append(_record(model="teleocr", vllm="0.22.1"), path)
    releases.append(_record(vllm="0.12.0", date="2026-10-02"), path)
    assert len(releases.load_releases(path)) == 3
    assert releases.latest("dotsocr", path)["vllm_version"] == "0.12.0"
    assert releases.latest("other", path) is None


def test_load_releases_missing_file_is_empty(tmp_path):
    assert releases.load_releases(tmp_path / "none.yaml") == []


@pytest.mark.parametrize(
    "secret",
    ["rpa_" + "A1b2C3d4E5f6G7h8I9j0K", "abcdefghij12.api.runpod.ai"],
)
def test_append_refuses_key_or_endpoint_id(tmp_path, secret):
    path = tmp_path / "releases.yaml"
    with pytest.raises(ReleaseRecordError):
        releases.append({**_record(), "deployer": {"note": ["x " + secret]}}, path)
    assert not path.exists()


def _cfg(env):
    # plans are irrelevant here; skip the validator that demands every plan key
    return ModelConfig.model_construct(
        name="dotsocr", endpoint_env=env, served_model_name="m", plans={}
    )


def _snapshot(tmp_path, url, version, monkeypatch):
    monkeypatch.setenv("DOTSOCR_BASE_URL", url)
    rp = RunPaths(root=tmp_path)
    cli._write_infer_config(rp, {"dotsocr": _cfg("DOTSOCR_BASE_URL")}, 1, {"dotsocr": version})
    return yaml.safe_load(rp.config_infer.read_text())["models"]["dotsocr"]


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("https://abcdefghij12.api.runpod.ai/v1", "runpod"),
        ("https://abcdefghij12.api.runpod.ai", "runpod"),
        ("http://localhost:8000/v1", "self_hosted"),
    ],
)
def test_endpoint_kind_recorded(tmp_path, monkeypatch, url, kind):
    assert _snapshot(tmp_path, url, "0.11.0", monkeypatch)["endpoint_kind"] == kind


def test_runpod_version_falls_back_to_latest_release(tmp_path, monkeypatch):
    path = tmp_path / "releases.yaml"
    releases.append(_record(vllm="0.11.0"), path)
    monkeypatch.setattr(releases, "DEFAULT_PATH", path)
    entry = _snapshot(tmp_path, "https://abcdefghij12.api.runpod.ai/v1", "unknown", monkeypatch)
    assert entry["vllm_version"] == "0.11.0"


def test_self_hosted_unknown_version_has_no_fallback(tmp_path, monkeypatch):
    path = tmp_path / "releases.yaml"
    releases.append(_record(vllm="0.11.0"), path)
    monkeypatch.setattr(releases, "DEFAULT_PATH", path)
    entry = _snapshot(tmp_path, "http://localhost:8000/v1", "unknown", monkeypatch)
    assert entry["vllm_version"] == "unknown"
