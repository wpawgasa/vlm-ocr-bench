"""`ocrbench` command-line entry point.

M0 wires the run-directory contract (missing-upstream-artifact checks) and
config validation for every stage. `prepare` is implemented (M1); the other
stage bodies are implemented in later milestones, and until then each exits 3
with a pointer to its tasks.md entry.
"""

import json
from pathlib import Path

import typer

from ocr_bench.config import ConfigError, load_config
from ocr_bench.data.bankstmt import UnknownBankError, coverage_warnings, prepare_statements
from ocr_bench.data.manifest import PreparedSample, materialize
from ocr_bench.data.thaiocrbench import TaskNameError, prepare_thaiocrbench
from ocr_bench.jsonl import write_rows_atomic
from ocr_bench.paths import MissingArtifactError, RunPaths, require
from ocr_bench.schemas import PrepareSummary

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
    datasets: str = typer.Option(
        "", "--datasets", help="Comma-separated subset of run.yaml's datasets (default: all)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an already-prepared run directory."
    ),
) -> None:
    """Build the manifest, degraded images and ground truth for a run."""
    try:
        resolved = load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    rp = RunPaths.for_run(run_id)
    if rp.manifest.exists() and not force:
        typer.echo(f"run {run_id} already prepared; use --force", err=True)
        raise typer.Exit(code=2)

    dataset_names = list(resolved.run.datasets)
    if datasets:
        requested = _csv(datasets)
        unknown = [d for d in requested if d not in dataset_names]
        if unknown:
            typer.echo(
                f"unknown dataset(s) in --datasets: {unknown}; available: {dataset_names}",
                err=True,
            )
            raise typer.Exit(code=2)
        dataset_names = requested

    rp.root.mkdir(parents=True, exist_ok=True)
    rp.config_resolved.write_text(resolved.to_yaml(), encoding="utf-8")

    all_samples: list[PreparedSample] = []
    summary = PrepareSummary(run_id=run_id)

    for name in dataset_names:
        dataset_cfg = resolved.datasets[name]
        try:
            if dataset_cfg.kind == "thaiocrbench":
                samples, info = prepare_thaiocrbench(dataset_cfg)
                summary.thaiocrbench = info["thaiocrbench"]
                summary.thaiocrbench_domains = info["thaiocrbench_domains"]
                summary.domain_slice_available = info["domain_slice_available"]
                summary.warnings.extend(info["warnings"])
            elif dataset_cfg.kind == "bankstmt":
                samples, info = prepare_statements(dataset_cfg)
                summary.statement_pages_by_doc_type = info["statement_pages_by_doc_type"]
                summary.statement_banks = info["statement_banks"]
                summary.statement_files = info["statement_files"]
                summary.statement_multipage_files = info["statement_multipage_files"]
                summary.statement_text_layer_unusable = info["statement_text_layer_unusable"]
                summary.warnings.extend(coverage_warnings(info, dataset_cfg.targets))
            else:  # pragma: no cover — discriminated union covers only these kinds
                continue
        except (FileNotFoundError, TaskNameError, UnknownBankError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc

        all_samples.extend(samples)

    rows = materialize(all_samples, resolved.run.conditions, rp)
    write_rows_atomic(rp.manifest, rows)

    summary.n_samples = len(all_samples)
    summary.n_manifest_rows = len(rows)
    rp.prepare_summary.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for warning in summary.warnings:
        typer.echo(f"warning: {warning}", err=True)
    typer.echo(f"prepared {summary.n_samples} samples, {summary.n_manifest_rows} manifest rows")


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
