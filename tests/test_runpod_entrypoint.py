"""runpod/entrypoint.py: renders `vllm serve` from the baked config (design D2)."""

import importlib.util
import shutil
from pathlib import Path

import pytest
import yaml

from ocr_bench.runpod.config import load_runpod_config, render_vllm_argv

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs" / "runpod"
MODELS = [
    p.stem
    for p in sorted(CONFIGS.glob("*.yaml"))
    if p.stem != "data_centers" and not p.stem.startswith("parity")
]
REV = "c" * 40

_spec = importlib.util.spec_from_file_location(
    "runpod_entrypoint", ROOT / "runpod" / "entrypoint.py"
)
entrypoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entrypoint)  # type: ignore[union-attr]


@pytest.fixture
def baked(tmp_path):
    """An /app/configs + /models tree; revisions pinned (repo configs say REQUIRED until 5.4)."""
    app = tmp_path / "app" / "configs"
    (app / "runpod").mkdir(parents=True)
    shutil.copytree(ROOT / "configs" / "models", app / "models")
    for name in MODELS:
        text = (
            (CONFIGS / f"{name}.yaml").read_text().replace("revision: REQUIRED", f"revision: {REV}")
        )
        (app / "runpod" / f"{name}.yaml").write_text(text)
    return app, tmp_path / "models"


def _with_weights(models: Path, name: str, app: Path) -> None:
    cfg = load_runpod_config(app / "runpod" / f"{name}.yaml")
    (models / cfg.weights.repo).mkdir(parents=True)


def test_models_found():
    assert "typhoon_ocr15" in MODELS and len(MODELS) == 5


@pytest.mark.parametrize("name", MODELS)
def test_argv_for_each_config(baked, name):
    app, models = baked
    _with_weights(models, name, app)
    cfg = load_runpod_config(app / "runpod" / f"{name}.yaml")
    served = yaml.safe_load((app / "models" / f"{name}.yaml").read_text())["served_model_name"]
    argv = entrypoint.build_argv({"OCRBENCH_MODEL": name}, app, models)
    assert argv[:4] == ["vllm", "serve", str(models / cfg.weights.repo), "--served-model-name"]
    assert argv[4] == served
    assert argv[5:] == render_vllm_argv(cfg)


def test_missing_weights_exits_1_naming_path(baked, capsys):
    app, models = baked
    with pytest.raises(SystemExit) as e:
        entrypoint.build_argv({"OCRBENCH_MODEL": "typhoon_ocr15"}, app, models)
    assert e.value.code == 1
    assert str(models / "typhoon-ai" / "typhoon-ocr1.5-2b") in capsys.readouterr().err


def test_unset_model_exits_1(baked, capsys):
    app, models = baked
    with pytest.raises(SystemExit) as e:
        entrypoint.build_argv({}, app, models)
    assert e.value.code == 1
    assert "OCRBENCH_MODEL" in capsys.readouterr().err


def test_other_engine_env_vars_are_ignored(baked):
    app, models = baked
    _with_weights(models, "typhoon_ocr15", app)
    base = entrypoint.build_argv({"OCRBENCH_MODEL": "typhoon_ocr15"}, app, models)
    noisy = {
        "OCRBENCH_MODEL": "typhoon_ocr15",
        "MAX_MODEL_LEN": "4096",
        "DTYPE": "float16",
        "VLLM_EXTRA_ARGS": "--seed 9",
        "TRUST_REMOTE_CODE": "1",
    }
    assert entrypoint.build_argv(noisy, app, models) == base
    assert "4096" not in base


def test_gpu_memory_utilization_override(baked):
    app, models = baked
    _with_weights(models, "typhoon_ocr15", app)
    env = {"OCRBENCH_MODEL": "typhoon_ocr15", "OCRBENCH_GPU_MEMORY_UTILIZATION": "0.5"}
    argv = entrypoint.build_argv(env, app, models)
    assert argv.count("--gpu-memory-utilization") == 1
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.5"


def test_gpu_memory_utilization_added_when_config_lacks_it(baked):
    app, models = baked
    name = "typhoon_ocr15"
    cfg_path = app / "runpod" / f"{name}.yaml"
    lines = cfg_path.read_text().splitlines(keepends=True)
    cfg_path.write_text("".join(ln for ln in lines if "gpu-memory-utilization" not in ln))
    _with_weights(models, name, app)
    env = {"OCRBENCH_MODEL": name, "OCRBENCH_GPU_MEMORY_UTILIZATION": "0.6"}
    argv = entrypoint.build_argv(env, app, models)
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.6"


def test_extra_args_appended_shell_split(baked):
    app, models = baked
    _with_weights(models, "typhoon_ocr15", app)
    env = {
        "OCRBENCH_MODEL": "typhoon_ocr15",
        "OCRBENCH_VLLM_EXTRA_ARGS": "--enforce-eager --foo 'a b'",
    }
    argv = entrypoint.build_argv(env, app, models)
    assert argv[-3:] == ["--enforce-eager", "--foo", "a b"]


def test_invalid_gpu_memory_utilization_exits_1(baked, capsys):
    app, models = baked
    _with_weights(models, "typhoon_ocr15", app)
    env = {"OCRBENCH_MODEL": "typhoon_ocr15", "OCRBENCH_GPU_MEMORY_UTILIZATION": "lots"}
    with pytest.raises(SystemExit) as e:
        entrypoint.build_argv(env, app, models)
    assert e.value.code == 1
    assert "OCRBENCH_GPU_MEMORY_UTILIZATION" in capsys.readouterr().err


def test_main_execs_vllm(baked, monkeypatch):
    app, models = baked
    _with_weights(models, "typhoon_ocr15", app)
    calls = []
    monkeypatch.setattr(entrypoint.os, "execvp", lambda f, a: calls.append((f, a)))
    monkeypatch.setenv("OCRBENCH_MODEL", "typhoon_ocr15")
    entrypoint.main(app_configs=app, models_root=models)
    assert calls[0][0] == "vllm"
    assert calls[0][1][:2] == ["vllm", "serve"]
    assert "--max-log-len" in calls[0][1]
