"""`ocrbench runpod ...` commands (design D4): render, build and deploy."""

import getpass
import json
import shlex
import subprocess
from datetime import date
from pathlib import Path

import typer
import yaml

from ocr_bench.runpod import releases
from ocr_bench.runpod import state as runpod_state
from ocr_bench.runpod.api import RunpodAPI, RunpodAPIError
from ocr_bench.runpod.config import RunpodModelConfig, load_runpod_config, render_vllm_argv
from ocr_bench.runpod.deploy import (
    DeployRefused,
    dry_run_bodies,
    endpoint_name,
    load_allowed_data_centers,
)
from ocr_bench.runpod.deploy import deploy as run_deploy
from ocr_bench.runpod.releases import ReleaseRecordError
from ocr_bench.runpod.render import endpoint_body, image_repo, template_body

runpod_app = typer.Typer(no_args_is_help=True, help="Build and deploy Runpod serving endpoints.")

DEFAULT_CONFIGS_DIR = Path("configs/runpod")
_PLACEHOLDER_DIGEST = "sha256:" + "0" * 64
_ConfigsDir = typer.Option(DEFAULT_CONFIGS_DIR, "--configs-dir", help="configs/runpod directory.")
_StatePath = typer.Option(runpod_state.DEFAULT_PATH, "--state", hidden=True)
_ReleasesPath = typer.Option(releases.DEFAULT_PATH, "--releases", hidden=True)


def _load(model: str, configs_dir: Path) -> RunpodModelConfig:
    path = configs_dir / f"{model}.yaml"
    try:
        return load_runpod_config(path)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {path}: {e}", err=True)
        raise typer.Exit(2) from e


def make_api() -> RunpodAPI:
    """The management API client from `RUNPOD_API_KEY` (tests replace this)."""
    return RunpodAPI.from_env()


def _git(*args: str, default: str) -> str:
    try:
        done = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return default
    return done.stdout.strip() or default


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
        tag = _git("rev-parse", "--short=12", "HEAD", default="local")
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


@runpod_app.command()
def deploy(
    model: str,
    digest: str = typer.Option(
        ..., "--digest", help="Image digest: sha256:... or <repo>@sha256:..."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the template/endpoint bodies; no API calls."
    ),
    configs_dir: Path = _ConfigsDir,
    state_path: Path = _StatePath,
    releases_path: Path = _ReleasesPath,
) -> None:
    """Create or update MODEL's Runpod template and endpoint from its config."""
    cfg = _load(model, configs_dir)
    allowed = load_allowed_data_centers(configs_dir / "data_centers.yaml")
    try:
        if dry_run:
            entry = (runpod_state.load_state(state_path).get("endpoints") or {}).get(
                endpoint_name(cfg), {}
            )
            bodies = dry_run_bodies(cfg, digest, allowed, entry=entry)
            typer.echo(json.dumps(bodies, indent=2, sort_keys=True))
            return
        result = run_deploy(
            cfg,
            digest,
            make_api(),
            allowed_data_centers=allowed,
            state_path=state_path,
            releases_path=releases_path,
            git_sha=_git("rev-parse", "HEAD", default="unknown"),
            deployer=_git("config", "user.name", default=getpass.getuser()),
            today=date.today(),
        )
    except (DeployRefused, ReleaseRecordError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    except RunpodAPIError as e:
        typer.echo(f"error: Runpod API: {e}", err=True)
        raise typer.Exit(1) from e
    if result.action == "up_to_date":
        typer.echo(f"{result.endpoint_name}: up to date")
    elif result.action == "patched":
        typer.echo(f"{result.endpoint_name}: patched {', '.join(result.changed)}")
    else:
        typer.echo(f"{result.endpoint_name}: {result.action}; release appended to {releases_path}")
    typer.echo(f"ids written to {state_path}")
