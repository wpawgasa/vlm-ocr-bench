"""`ocrbench runpod ...` commands (design D4): render, build and deploy."""

import asyncio
import getpass
import json
import os
import shlex
import subprocess
from datetime import date
from pathlib import Path

import httpx
import typer
import yaml

from ocr_bench.config import ConfigError, load_model_config
from ocr_bench.models.registry import build_model
from ocr_bench.models.vllm_client import EndpointUnavailable
from ocr_bench.runpod import manage, releases
from ocr_bench.runpod import review as runpod_review
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
from ocr_bench.runpod.smoke import SmokeReport, run_smoke

runpod_app = typer.Typer(no_args_is_help=True, help="Build and deploy Runpod serving endpoints.")

DEFAULT_CONFIGS_DIR = Path("configs/runpod")
_PLACEHOLDER_DIGEST = "sha256:" + "0" * 64
_ConfigsDir = typer.Option(DEFAULT_CONFIGS_DIR, "--configs-dir", help="configs/runpod directory.")
_StatePath = typer.Option(runpod_state.DEFAULT_PATH, "--state", hidden=True)
_ReleasesPath = typer.Option(releases.DEFAULT_PATH, "--releases", hidden=True)
_ReviewPath = typer.Option(runpod_review.DEFAULT_PATH, "--review", help="Review document.")


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


def _fail(message: str, code: int = 2) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


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
    digest: str | None = typer.Option(
        None, "--digest", help="Image digest: sha256:... or <repo>@sha256:..."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the template/endpoint bodies; no API calls."
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="office (min workers 1, promoted models only) or off (0)."
    ),
    promote: bool = typer.Option(
        False, "--promote", help="Allow client data once parity and the review pass."
    ),
    review: Path = _ReviewPath,
    approver: str | None = typer.Option(None, "--approver", help="Approver (default: git user)."),
    retention: str | None = typer.Option(
        None, "--retention", help="Stated platform retention (needed to promote a queue endpoint)."
    ),
    configs_dir: Path = _ConfigsDir,
    state_path: Path = _StatePath,
    releases_path: Path = _ReleasesPath,
) -> None:
    """Create or update MODEL's Runpod endpoint; with --promote, allow client data."""
    if profile not in (None, "office", "off"):
        raise _fail("--profile must be office or off")
    if digest is None and not promote:
        raise _fail("--digest is required (except with --promote alone)")
    cfg = _load(model, configs_dir)
    allowed = load_allowed_data_centers(configs_dir / "data_centers.yaml")
    try:
        if digest is not None and dry_run:
            entry = (runpod_state.load_state(state_path).get("endpoints") or {}).get(
                endpoint_name(cfg), {}
            )
            bodies = dry_run_bodies(cfg, digest, allowed, profile=profile, entry=entry)  # type: ignore[arg-type]
            typer.echo(json.dumps(bodies, indent=2, sort_keys=True))
            return
        if digest is not None:
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
                profile=profile,  # type: ignore[arg-type]
            )
            if result.action == "up_to_date":
                typer.echo(f"{result.endpoint_name}: up to date")
            elif result.action == "patched":
                typer.echo(f"{result.endpoint_name}: patched {', '.join(result.changed)}")
            else:
                typer.echo(
                    f"{result.endpoint_name}: {result.action}; release appended to {releases_path}"
                )
            typer.echo(f"ids written to {state_path}")
        if promote:
            manage.promote(
                cfg,
                make_api(),
                state_path=state_path,
                releases_path=releases_path,
                review_path=review,
                approver=approver or _git("config", "user.name", default=getpass.getuser()),
                today=date.today(),
                retention=retention,
            )
            typer.echo(f"{endpoint_name(cfg)}: promoted for client data")
    except (DeployRefused, ReleaseRecordError, manage.ManageRefused) as e:
        raise _fail(str(e)) from e
    except RunpodAPIError as e:
        raise _fail(f"Runpod API: {e}", 1) from e


@runpod_app.command()
def rollback(
    model: str,
    configs_dir: Path = _ConfigsDir,
    state_path: Path = _StatePath,
    releases_path: Path = _ReleasesPath,
) -> None:
    """Point MODEL's endpoint back at the previous release's template."""
    cfg = _load(model, configs_dir)
    try:
        result = manage.rollback(
            cfg,
            make_api(),
            state_path=state_path,
            releases_path=releases_path,
            git_sha=_git("rev-parse", "HEAD", default="unknown"),
            deployer=_git("config", "user.name", default=getpass.getuser()),
            today=date.today(),
        )
    except (ReleaseRecordError, manage.ManageRefused) as e:
        raise _fail(str(e)) from e
    except RunpodAPIError as e:
        raise _fail(f"Runpod API: {e}", 1) from e
    typer.echo(
        f"{result.endpoint_name}: rolled back to {result.image_digest}; "
        "promotion suspended until parity passes again"
    )


@runpod_app.command()
def status(
    model: str,
    review: Path = _ReviewPath,
    configs_dir: Path = _ConfigsDir,
    state_path: Path = _StatePath,
    releases_path: Path = _ReleasesPath,
) -> None:
    """Show MODEL's endpoint, release, parity, promotion state and worker counts."""
    cfg = _load(model, configs_dir)
    try:
        st = manage.status(
            cfg,
            make_api(),
            state_path=state_path,
            releases_path=releases_path,
            review_path=review,
            today=date.today(),
        )
    except RunpodAPIError as e:
        raise _fail(f"Runpod API: {e}", 1) from e
    promotion = "not promoted"
    if st.promoted:
        promotion = f"promoted by {st.approver} on {st.approved_on}"
    typer.echo(f"endpoint:    {st.endpoint_name} ({st.masked_id or 'not deployed'})")
    typer.echo(f"release:     {st.release_action or '-'} {st.release_digest or ''}")
    typer.echo(f"released on: {st.release_date or '-'}")
    typer.echo(f"parity:      {st.parity or '-'}")
    typer.echo(f"client data: {promotion}")
    typer.echo(f"workers:     {st.workers if st.workers is not None else 'unknown'}")
    if st.review_problem:
        typer.echo(f"warning: promotion no longer valid: {st.review_problem}")


@runpod_app.command()
def smoke(model: str, configs_dir: Path = _ConfigsDir) -> None:
    """Check MODEL's endpoint: health, served name, one page, auth rejection, latency."""
    cfg = _load(model, configs_dir)
    try:
        mcfg = load_model_config(configs_dir.parent / "models" / f"{model}.yaml", model)
        os.environ["OCRBENCH_ENDPOINT_WAIT_S"] = str(cfg.client.cold_start_wait_s)
        built = build_model(mcfg)
    except (ConfigError, EndpointUnavailable) as e:
        raise _fail(str(e)) from e

    async def go() -> SmokeReport:
        async with httpx.AsyncClient() as http:
            return await run_smoke(
                cfg,
                mcfg,
                built,
                base_url=built.client.base_url,  # type: ignore[attr-defined]
                api_key=getattr(built.client, "api_key", None),
                http=http,
            )

    report = asyncio.run(go())
    typer.echo(f"cold start:   {report.cold_start_s:.1f}s (slo {cfg.slo.cold_start_p95_s:g}s)")
    typer.echo(f"page request: {report.request_s:.1f}s (slo {cfg.slo.page_p95_s:g}s)")
    if report.layout_s is not None:
        typer.echo(f"layout:       {report.layout_s:.1f}s")
    for failure in report.failures:
        typer.echo(f"FAIL: {failure}", err=True)
    if not report.ok:
        raise typer.Exit(1)
    typer.echo(f"{model}: smoke passed (served {mcfg.served_model_name})")
