"""runpod/paddle_supervisor.py: vLLM, N PaddleX pipelines and the front in one worker (D3)."""

import importlib.util
import shutil
import sys
import time
from pathlib import Path

import pytest

from ocr_bench.runpod.config import load_runpod_config

ROOT = Path(__file__).resolve().parent.parent
REV = "c" * 40

_spec = importlib.util.spec_from_file_location(
    "paddle_supervisor", ROOT / "runpod" / "paddle_supervisor.py"
)
sup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sup)  # type: ignore[union-attr]


@pytest.fixture
def baked(tmp_path):
    app = tmp_path / "app" / "configs"
    (app / "runpod").mkdir(parents=True)
    shutil.copytree(ROOT / "configs" / "models", app / "models")
    text = (ROOT / "configs" / "runpod" / "paddleocr_vl.yaml").read_text()
    (app / "runpod" / "paddleocr_vl.yaml").write_text(
        text.replace("revision: REQUIRED", f"revision: {REV}")
    )
    models = tmp_path / "models"
    cfg = load_runpod_config(app / "runpod" / "paddleocr_vl.yaml")
    (models / cfg.weights.repo).mkdir(parents=True)
    return app, models


def _cmds(baked, **env):
    app, models = baked
    return sup.build_commands(
        {"OCRBENCH_MODEL": "paddleocr_vl", **env}, app_configs=app, models_root=models
    )


def test_vllm_on_loopback_8000(baked):
    name, argv, env = _cmds(baked)[0]
    assert name == "vllm"
    assert argv[:2] == ["vllm", "serve"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--served-model-name") + 1] == "PaddleOCR-VL-1.6-0.9B"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.7"


def test_default_two_pipelines_on_8081_up(baked):
    pipes = [c for c in _cmds(baked) if c[0].startswith("pipeline")]
    assert [env["PORT"] for _, _, env in pipes] == ["8081", "8082"]
    for _, argv, env in pipes:
        assert argv[-1].endswith("paddle-pipeline-entrypoint.py")
        assert env["VL_SERVER_URL"] == "http://127.0.0.1:8000/v1"
        assert env["PIPELINE_DEVICE"] == "gpu:0"


def test_pipelines_run_in_the_paddle_venv(baked):
    """Paddle's own CUDA wheels live in a separate venv so they cannot replace vLLM's NCCL."""
    cmds = _cmds(baked, PADDLE_PYTHON="/opt/paddle/bin/python", PATH="/usr/bin")
    pipes = [c for c in cmds if c[0].startswith("pipeline")]
    for _, argv, env in pipes:
        assert argv[0] == "/opt/paddle/bin/python"
        assert env["PATH"] == "/opt/paddle/bin:/usr/bin"  # entrypoint execs that `paddlex`
    others = [c for c in cmds if not c[0].startswith("pipeline")]
    for _, _, env in others:
        assert env.get("PATH") == "/usr/bin"


def test_paddle_pipelines_env_sets_count(baked):
    pipes = [c for c in _cmds(baked, PADDLE_PIPELINES="3") if c[0].startswith("pipeline")]
    assert [env["PORT"] for _, _, env in pipes] == ["8081", "8082", "8083"]


def test_front_on_public_port_with_pipeline_count(baked):
    name, argv, env = _cmds(baked, PORT="8080", PADDLE_PIPELINES="3")[-1]
    assert name == "front"
    assert argv[-1].endswith("paddle_front.py")
    assert env["PORT"] == "8080"
    assert env["PADDLE_PIPELINES"] == "3"


def test_front_port_defaults_to_8080(baked):
    _, _, env = _cmds(baked)[-1]
    assert env["PORT"] == "8080"


def test_bad_pipeline_count_exits(baked):
    with pytest.raises(SystemExit):
        _cmds(baked, PADDLE_PIPELINES="0")


def test_child_exit_stops_the_rest_and_propagates_code():
    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    fail = [sys.executable, "-c", "import sys; sys.exit(3)"]
    t0 = time.monotonic()
    code = sup.supervise([("a", sleep, {}), ("b", fail, {}), ("c", sleep, {})])
    assert code == 3
    assert time.monotonic() - t0 < 30


def test_clean_child_exit_still_fails_the_worker():
    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    ok = [sys.executable, "-c", "pass"]
    assert sup.supervise([("a", sleep, {}), ("b", ok, {})]) != 0
