import json
import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ocr_bench.cli import app
from tests.runpod_helpers import REV, pinned_configs

runner = CliRunner()
IMAGE_DIGEST = "ghcr.io/wpawgasa/vlm-ocr-bench/runpod-teleocr@sha256:" + "d" * 64


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    """A `docker` on PATH that logs its argv and answers `inspect` with a digest."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "docker.log"
    script = bindir / "docker"
    script.write_text(
        f'#!/bin/sh\necho "$@" >> {log}\nif [ "$1" = inspect ]; then echo "{IMAGE_DIGEST}"; fi\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return log


def _calls(log: Path) -> list[str]:
    return log.read_text().splitlines()


def test_build_runs_docker_build_with_pinned_args(fake_docker, tmp_path):
    configs = pinned_configs(tmp_path / "c")
    result = runner.invoke(app, ["runpod", "build", "teleocr", "--configs-dir", str(configs)])
    assert result.exit_code == 0, result.output
    (call,) = _calls(fake_docker)
    assert call.startswith("build ")
    assert "--target teleocr" in call
    assert "--build-arg VLLM_IMAGE=vllm/vllm-openai@sha256:014a95f2" in call
    assert "--build-arg MODEL=teleocr" in call
    assert "--build-arg WEIGHTS_REPO=StarDoc-AI/TeleOCR" in call
    assert f"--build-arg WEIGHTS_REVISION={REV}" in call
    assert "--secret id=HF_TOKEN,env=HF_TOKEN" in call
    assert "-f runpod/Dockerfile" in call


def test_smoke_unset_endpoint_exits_2(tmp_path, monkeypatch):
    configs = pinned_configs(tmp_path / "c")
    monkeypatch.delenv("TYPHOON_BASE_URL", raising=False)
    result = runner.invoke(app, ["runpod", "smoke", "typhoon_ocr15", "--configs-dir", str(configs)])
    assert result.exit_code == 2
    assert "TYPHOON_BASE_URL" in result.output


def test_build_push_pushes_then_prints_digest(fake_docker, tmp_path):
    configs = pinned_configs(tmp_path / "c")
    result = runner.invoke(
        app, ["runpod", "build", "teleocr", "--push", "--configs-dir", str(configs)]
    )
    assert result.exit_code == 0, result.output
    verbs = [c.split()[0] for c in _calls(fake_docker)]
    assert verbs == ["build", "push", "inspect"]
    assert IMAGE_DIGEST in result.output


def test_build_unpinned_revision_exits_nonzero_without_calling_docker(fake_docker):
    # the shipped configs still carry `revision: REQUIRED`
    result = runner.invoke(app, ["runpod", "build", "teleocr"])
    assert result.exit_code != 0
    assert not fake_docker.exists()


def test_render_prints_argv_and_bodies(tmp_path):
    configs = pinned_configs(tmp_path / "c")
    digest = "sha256:" + "c" * 64
    result = runner.invoke(
        app, ["runpod", "render", "dotsocr", "--digest", digest, "--configs-dir", str(configs)]
    )
    assert result.exit_code == 0, result.output
    assert "vllm serve" in result.output
    assert "--served-model-name" in result.output
    assert "--max-log-len 0" in result.output
    bodies = json.loads(result.output[result.output.index("\n{\n") :])
    assert bodies["template"]["image"].endswith(f"runpod-dotsocr@{digest}")
    assert bodies["endpoint"]["workers"]["max"] == 3
