"""Static checks on runpod/Dockerfile (design D2); the real build is the `live` test below."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "runpod" / "Dockerfile"


def _stages() -> dict[str, str]:
    """Stage name -> its instructions (line continuations joined)."""
    text = DOCKERFILE.read_text().replace("\\\n", " ")
    stages: dict[str, str] = {}
    name = None
    for line in text.splitlines():
        m = re.match(r"FROM\s+\S+\s+AS\s+(\w+)", line, re.IGNORECASE)
        if m:
            name = m.group(1)
            stages[name] = ""
        elif name:
            stages[name] += line + "\n"
    return stages


def test_stages_are_plain_teleocr_paddle():
    assert list(_stages()) == ["plain", "teleocr", "paddle"]


def test_args_declared_before_first_from():
    text = DOCKERFILE.read_text()
    assert "ARG VLLM_IMAGE" in text.split("FROM", 1)[0]
    assert "ARG MODEL" in text


def test_plain_copies_entrypoint_and_configs():
    plain = _stages()["plain"]
    assert "runpod/entrypoint.py" in plain
    assert "configs/runpod" in plain
    assert "configs/models" in plain


def test_plain_downloads_weights_with_secret_and_pinned_revision():
    plain = _stages()["plain"]
    assert "--mount=type=secret,id=HF_TOKEN" in plain
    assert "snapshot_download" in plain
    assert "revision" in plain
    assert "/models/" in plain


def test_no_stage_bakes_hf_token_into_env_or_arg():
    for body in _stages().values():
        assert not re.search(r"^\s*(ENV|ARG)\s+HF_TOKEN", body, re.MULTILINE)


def test_plain_env_defaults():
    plain = _stages()["plain"]
    for item in (
        "HF_HUB_OFFLINE=1",
        "PORT=8000",
        "OCRBENCH_MODEL=${MODEL}",
        "HEALTH_CHECK_PATH=/health",
    ):
        assert item in plain


def test_teleocr_repeats_devcontainer_install_with_same_commit():
    dev = (ROOT / ".devcontainer" / "teleocr-vllm.Dockerfile").read_text()
    commit = re.search(r"ARG TELEOCR_COMMIT=(\w+)", dev).group(1)
    teleocr = _stages()["teleocr"]
    assert f"ARG TELEOCR_COMMIT={commit}" in teleocr
    assert "TeleOCR-vllm @ git+https://github.com/caipeng328/TeleOCR" in teleocr
    assert "teleocr_no_repeat_ngram" in teleocr


def test_paddle_installs_gpu_paddlex():
    paddle = _stages()["paddle"]
    assert "PADDLE_PKG=paddlepaddle-gpu" in paddle
    # paddlepaddle-gpu 3.x is only on Paddle's own index (PyPI stops at 2.6.x)
    assert "PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu126/" in paddle
    assert "paddleocr[doc-parser]==3.7.0" in paddle


def test_paddle_install_matches_devcontainer_pipeline_image():
    """Same install as .devcontainer/paddle-pipeline.Dockerfile (run inside the Paddle venv);
    only the wheel default differs."""
    dev = (ROOT / ".devcontainer" / "paddle-pipeline.Dockerfile").read_text().replace("\\\n", " ")
    dev_run = next(ln for ln in dev.splitlines() if "paddleocr[doc-parser]" in ln)
    stage = _stages()["paddle"].replace("\\\n", " ")
    paddle_run = next(ln for ln in stage.splitlines() if "paddleocr[doc-parser]" in ln)

    def install(s: str) -> str:
        s = " ".join(s.split()).replace("python3 ", "python ")
        return s[s.index('pip install --no-cache-dir "${PADDLE_PKG}') :]

    assert install(paddle_run) == install(dev_run)


def test_paddle_installs_into_its_own_venv():
    """paddlepaddle-gpu's nvidia-* wheels would replace the NCCL vLLM's torch needs."""
    stage = _stages()["paddle"]
    assert "uv venv --seed /opt/paddle" in stage
    assert ". /opt/paddle/bin/activate" in stage
    assert "PADDLE_PYTHON=/opt/paddle/bin/python" in stage


def test_paddle_runs_supervisor_and_front_on_8080():
    paddle = _stages()["paddle"]
    for f in ("runpod/paddle_supervisor.py", "runpod/paddle_front.py"):
        assert f in paddle
    assert ".devcontainer/paddle-pipeline-entrypoint.py" in paddle
    assert "PORT=8080" in paddle
    assert "PADDLE_PIPELINES=2" in paddle
    assert "EXPOSE 8080" in paddle
    assert re.search(r'ENTRYPOINT \[.*paddle_supervisor\.py"\]', paddle)


@pytest.mark.live
def test_plain_build_with_fake_vllm_prints_rendered_argv():
    """Needs docker and network (pulls python:3.11-slim); vllm and huggingface_hub are stubs."""
    if shutil.which("docker") is None or not (ROOT / "runpod" / "entrypoint.py").exists():
        pytest.skip("needs docker and runpod/entrypoint.py (task 2.2)")
    fake = ROOT / "tests" / "runpod" / "fake_vllm"
    base = "ocrbench-fake-vllm-base"
    subprocess.run(["docker", "build", "-t", base, str(fake)], check=True)
    img = "ocrbench-runpod-plain-test"
    subprocess.run(
        [
            "docker", "build", "--target", "plain", "-t", img, "-f", str(DOCKERFILE),
            "--build-arg", f"VLLM_IMAGE={base}",
            "--build-arg", "MODEL=typhoon_ocr15",
            "--build-arg", "WEIGHTS_REPO=typhoon-ai/typhoon-ocr1.5-2b",
            "--build-arg", "WEIGHTS_REVISION=" + "a" * 40,
            str(ROOT),
        ],
        check=True,
    )  # fmt: skip
    out = subprocess.run(["docker", "run", "--rm", img], check=True, capture_output=True, text=True)
    assert "--max-log-len 0" in out.stdout


@pytest.mark.parametrize("variant", ["plain", "teleocr", "paddle"])
def test_stage_port_matches_the_template_port(variant):
    from ocr_bench.runpod.render import worker_port

    stage = _stages()[variant]
    if variant == "teleocr":  # inherits the plain stage's ENV
        stage = _stages()["plain"] + stage
    ports = re.findall(r"PORT=(\d+)", stage)
    assert ports and int(ports[-1]) == worker_port(variant)
