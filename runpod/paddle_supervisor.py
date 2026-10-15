"""PaddleOCR-VL worker supervisor (design D3): vLLM, N PaddleX pipelines and the front.

Starts, in one container on one GPU:
- `vllm serve` (argv from the baked config, as runpod/entrypoint.py renders it) on 127.0.0.1:8000;
- `PADDLE_PIPELINES` (default 2) copies of paddle-pipeline-entrypoint.py on 127.0.0.1:8081...,
  each with VL recognition pointed at the loopback vLLM and layout detection on gpu:0;
- runpod/paddle_front.py on `PORT` (default 8080), the only public port.

Every pipeline copy rewrites the same /config/PaddleOCR-VL-1.6.vllm-server.yaml with identical
content in its first second, before `paddlex` (seconds of imports later) reads it; they are
started back to back for that reason, never staggered.

If any process exits, the others are stopped and the supervisor exits non-zero, so Runpod
replaces the worker instead of leaving a half-alive one behind the load balancer.
"""

import importlib.util
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_CONFIGS = Path("/app/configs")
MODELS_ROOT = Path("/models")
VLLM_HOST, VLLM_PORT = "127.0.0.1", 8000
PIPELINE_BASE_PORT = 8081
PIPELINE_SCRIPT = HERE / "paddle-pipeline-entrypoint.py"
_STOP_GRACE_S = 15

Command = tuple[str, list[str], dict[str, str]]


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("runpod_entrypoint", HERE / "entrypoint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _die(msg: str) -> None:
    print(f"paddle_supervisor: {msg}", file=sys.stderr)
    raise SystemExit(1)


def build_commands(
    env: Mapping[str, str],
    app_configs: Path = APP_CONFIGS,
    models_root: Path = MODELS_ROOT,
    pipeline_script: Path = PIPELINE_SCRIPT,
) -> list[Command]:
    try:
        n = int(env.get("PADDLE_PIPELINES", "2"))
    except ValueError:
        n = 0
    if n < 1:
        _die(f"PADDLE_PIPELINES must be a positive integer, got {env.get('PADDLE_PIPELINES')!r}")

    base = dict(env)
    vllm = _load_entrypoint().build_argv(env, app_configs, models_root)
    vllm += ["--host", VLLM_HOST, "--port", str(VLLM_PORT)]
    cmds: list[Command] = [("vllm", vllm, base)]

    for i in range(n):
        port = PIPELINE_BASE_PORT + i
        cmds.append(
            (
                f"pipeline{i}",
                [sys.executable, str(pipeline_script)],
                {
                    **base,
                    "PORT": str(port),
                    "VL_SERVER_URL": f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
                    "PIPELINE_DEVICE": "gpu:0",
                },
            )
        )

    front_env = {
        **base,
        "PORT": env.get("PORT", "8080"),
        "PADDLE_PIPELINES": str(n),
        "VLLM_URL": f"http://{VLLM_HOST}:{VLLM_PORT}",
    }
    cmds.append(("front", [sys.executable, str(HERE / "paddle_front.py")], front_env))
    return cmds


def _stop(procs: dict[str, subprocess.Popen]) -> None:
    for p in procs.values():
        if p.poll() is None:
            p.terminate()
    deadline = time.monotonic() + _STOP_GRACE_S
    for p in procs.values():
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def supervise(cmds: list[Command]) -> int:
    """Run every command; when the first one exits, stop the rest and return a non-zero code."""
    procs: dict[str, subprocess.Popen] = {}
    stopping = False

    def _on_signal(signum, _frame):
        nonlocal stopping
        stopping = True

    old = {s: signal.signal(s, _on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for name, argv, env in cmds:
            print(f"paddle_supervisor: start {name}: {' '.join(argv)}", flush=True)
            procs[name] = subprocess.Popen(argv, env=env or None)
        while not stopping:
            for name, p in procs.items():
                code = p.poll()
                if code is not None:
                    print(f"paddle_supervisor: {name} exited with {code}", flush=True)
                    _stop(procs)
                    return code or 1
            time.sleep(0.2)
        _stop(procs)
        return 0
    finally:
        _stop(procs)
        for s, h in old.items():
            signal.signal(s, h)


def main() -> None:
    sys.exit(supervise(build_commands(os.environ)))


if __name__ == "__main__":
    main()
