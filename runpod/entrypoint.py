"""Runpod worker entrypoint (design D2): render `vllm serve` from the baked config and exec it.

Only OCRBENCH_GPU_MEMORY_UTILIZATION and OCRBENCH_VLLM_EXTRA_ARGS change the command; every
other engine env var (MAX_MODEL_LEN, DTYPE, ...) is ignored so a template cannot drift the
engine away from the config the image was built and released with.
"""

import os
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

from ocr_bench.runpod.config import load_runpod_config, render_vllm_argv

APP_CONFIGS = Path("/app/configs")
MODELS_ROOT = Path("/models")
_GPU_FLAG = "--gpu-memory-utilization"


def _die(msg: str) -> "None":
    print(f"entrypoint: {msg}", file=sys.stderr)
    raise SystemExit(1)


def build_argv(
    env: Mapping[str, str],
    app_configs: Path = APP_CONFIGS,
    models_root: Path = MODELS_ROOT,
) -> list[str]:
    model = env.get("OCRBENCH_MODEL", "")
    if not model:
        _die("OCRBENCH_MODEL is not set")
    cfg = load_runpod_config(app_configs / "runpod" / f"{model}.yaml")
    weights = models_root / cfg.weights.repo
    if not weights.is_dir():
        _die(f"weights not found at {weights}")
    served = yaml.safe_load((app_configs / "models" / f"{model}.yaml").read_text())[
        "served_model_name"
    ]
    argv = ["vllm", "serve", str(weights), "--served-model-name", served, *render_vllm_argv(cfg)]

    util = env.get("OCRBENCH_GPU_MEMORY_UTILIZATION")
    if util:
        try:
            float(util)
        except ValueError:
            _die(f"OCRBENCH_GPU_MEMORY_UTILIZATION must be a number, got {util!r}")
        if _GPU_FLAG in argv:
            argv[argv.index(_GPU_FLAG) + 1] = util
        else:
            argv += [_GPU_FLAG, util]

    argv += shlex.split(env.get("OCRBENCH_VLLM_EXTRA_ARGS", ""))
    return argv


def main(app_configs: Path = APP_CONFIGS, models_root: Path = MODELS_ROOT) -> None:
    argv = build_argv(os.environ, app_configs, models_root)
    os.execvp("vllm", argv)


if __name__ == "__main__":
    main()
