"""`ocrbench` command-line entry point.

M0 wires the run-directory contract (missing-upstream-artifact checks) and
config validation for every stage. The stage bodies are implemented in later
milestones; until then each command exits 3 with a pointer to its tasks.md
entry.
"""

from pathlib import Path

import typer

from ocr_bench.config import ConfigError, load_config
from ocr_bench.paths import MissingArtifactError, RunPaths, require

app = typer.Typer(
    name="ocrbench",
    no_args_is_help=True,
    help="TeleOCR vs dots.ocr benchmark harness.",
)

DEFAULT_CONFIG_PATH = Path("configs/run.yaml")


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _require_or_exit(path: Path, producer: str) -> None:
    try:
        require(path, producer)
    except MissingArtifactError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _not_implemented(cmd: str, task_ref: str) -> None:
    typer.echo(f"ocrbench {cmd}: not implemented yet (tasks.md {task_ref})", err=True)
    raise typer.Exit(code=3)


@app.command()
def prepare(
    config: Path = typer.Option(  # noqa: B008 — typer's own default-injection idiom
        DEFAULT_CONFIG_PATH, "--config", help="Path to run.yaml."
    ),
    run_id: str = typer.Option(..., "--run-id", help="Run id to prepare."),
) -> None:
    """Build the manifest, degraded images and ground truth for a run."""
    try:
        load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    _not_implemented("prepare", "2.5")


@app.command()
def infer(
    run_id: str = typer.Option(..., "--run-id", help="Run id to infer over."),
    models: str = typer.Option("teleocr,dotsocr", "--models", help="Comma-separated model names."),
) -> None:
    """Run each model over the manifest and record predictions."""
    rp = RunPaths.for_run(run_id)
    _require_or_exit(rp.manifest, "prepare")
    _not_implemented("infer", "3.5")


@app.command()
def probe(
    run_id: str = typer.Option(..., "--run-id", help="Run id to probe."),
    models: str = typer.Option("teleocr,dotsocr", "--models", help="Comma-separated model names."),
) -> None:
    """Probe each model's structural output capabilities."""
    rp = RunPaths.for_run(run_id)
    _require_or_exit(rp.manifest, "prepare")
    _not_implemented("probe", "3.6")


@app.command()
def score(
    run_id: str = typer.Option(..., "--run-id", help="Run id to score."),
) -> None:
    """Score predictions against ground truth and write scores/fields rows."""
    rp = RunPaths.for_run(run_id)
    _require_or_exit(rp.manifest, "prepare")
    _require_or_exit(rp.predictions, "infer")
    _not_implemented("score", "4.7")


@app.command()
def calibrate(
    run_id: str = typer.Option(..., "--run-id", help="Run id to calibrate."),
    target_acc: str = typer.Option(
        "0.99,0.995", "--target-acc", help="Comma-separated target accuracies."
    ),
) -> None:
    """Fit confidence bands and accept/reject thresholds."""
    rp = RunPaths.for_run(run_id)
    _require_or_exit(rp.fields, "score")
    _not_implemented("calibrate", "6.4")


@app.command(name="bench-latency")
def bench_latency(
    run_id: str = typer.Option(..., "--run-id", help="Run id to benchmark."),
    gpu: str = typer.Option(..., "--gpu", help="GPU label for the recorded rows."),
    concurrency: str = typer.Option(
        "1,4,8,16,32", "--concurrency", help="Comma-separated concurrency levels."
    ),
) -> None:
    """Benchmark serving latency and throughput across concurrency levels."""
    rp = RunPaths.for_run(run_id)
    _require_or_exit(rp.manifest, "prepare")
    _not_implemented("bench-latency", "7.3")


@app.command()
def report(
    run_id: str = typer.Option(..., "--run-id", help="Run id to report on."),
) -> None:
    """Render the client-facing scorecard and supporting artifacts."""
    rp = RunPaths.for_run(run_id)
    _require_or_exit(rp.manifest, "prepare")
    _not_implemented("report", "8.4")


if __name__ == "__main__":
    app()
