"""`ocrbench runpod ...` commands (design D4): render and build."""

import json
import shlex
import subprocess
from pathlib import Path

import typer
import yaml

from ocr_bench.runpod.config import RunpodModelConfig, load_runpod_config, render_vllm_argv
from ocr_bench.runpod.render import endpoint_body, image_repo, template_body

runpod_app = typer.Typer(no_args_is_help=True, help="Build and deploy Runpod serving endpoints.")

DEFAULT_CONFIGS_DIR = Path("configs/runpod")
_PLACEHOLDER_DIGEST = "sha256:" + "0" * 64
_ConfigsDir = typer.Option(DEFAULT_CONFIGS_DIR, "--configs-dir", help="configs/runpod directory.")


def _load(model: str, configs_dir: Path) -> RunpodModelConfig:
    path = configs_dir / f"{model}.yaml"
    try:
        return load_runpod_config(path)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {path}: {e}", err=True)
        raise typer.Exit(2) from e


def _docker(*args: str) -> str:
    try:
        done = subprocess.run(["docker", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as e:
        typer.echo(f"error: docker {args[0]} failed: {getattr(e, 'stderr', '') or e}", err=True)
        raise typer.Exit(1) from e
    return done.stdout.strip()


@runpod_app.command()
def render(
    model: str,
    digest: str = typer.Option(
        _PLACEHOLDER_DIGEST, "--digest", help="Image digest (sha256:...) for the template."
    ),
    profile: str | None = typer.Option(None, "--profile", help="office or off."),
    configs_dir: Path = _ConfigsDir,
) -> None:
    """Print the vllm argv and the template/endpoint JSON a deploy would send."""
    if profile not in (None, "office", "off"):
        typer.echo("error: --profile must be office or off", err=True)
        raise typer.Exit(2)
    cfg = _load(model, configs_dir)
    served = yaml.safe_load((configs_dir.parent / "models" / f"{model}.yaml").read_text())[
        "served_model_name"
    ]
    argv = ["vllm", "serve", f"/models/{cfg.weights.repo}", "--served-model-name", served]
    typer.echo(shlex.join([*argv, *render_vllm_argv(cfg)]))
    try:
        bodies = {
            "template": template_body(cfg, digest),
            "endpoint": endpoint_body(cfg, "<template-id>", profile),  # type: ignore[arg-type]
        }
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    typer.echo(json.dumps(bodies, indent=2, sort_keys=True))


@runpod_app.command()
def build(
    model: str,
    push: bool = typer.Option(False, "--push", help="Push the image and print its digest."),
    tag: str | None = typer.Option(None, "--tag", help="Image tag (default: git short sha)."),
    configs_dir: Path = _ConfigsDir,
) -> None:
    """Build the worker image for MODEL with the pinned base and weights revision."""
    cfg = _load(model, configs_dir)
    if tag is None:
        try:
            tag = subprocess.run(
                ["git", "rev-parse", "--short=12", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            tag = "local"
    image = f"{image_repo(model)}:{tag}"
    _docker(
        "build",
        "--target",
        cfg.image.variant,
        "-f",
        "runpod/Dockerfile",
        "--build-arg",
        f"VLLM_IMAGE={cfg.image.base}",
        "--build-arg",
        f"MODEL={model}",
        "--build-arg",
        f"WEIGHTS_REPO={cfg.weights.repo}",
        "--build-arg",
        f"WEIGHTS_REVISION={cfg.weights.revision}",
        "--secret",
        "id=HF_TOKEN,env=HF_TOKEN",
        "-t",
        image,
        ".",
    )
    typer.echo(f"built {image}")
    if push:
        _docker("push", image)
        typer.echo(_docker("inspect", "--format", "{{index .RepoDigests 0}}", image))
