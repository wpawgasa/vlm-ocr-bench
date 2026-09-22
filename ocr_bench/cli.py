"""`ocrbench` command-line entry point.

M0 wires the run-directory contract (missing-upstream-artifact checks) and
config validation for every stage. `prepare` is implemented (M1); the other
stage bodies are implemented in later milestones, and until then each exits 3
with a pointer to its tasks.md entry.
"""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import typer
import yaml

from ocr_bench.config import (
    BankOverrides,
    BankStmtConfig,
    ConfigError,
    ModelConfig,
    ResolvedConfig,
    RunConfig,
    load_config,
    load_model_config,
)
from ocr_bench.data.bankstmt import UnknownBankError, coverage_warnings, prepare_statements
from ocr_bench.data.manifest import PreparedSample, materialize
from ocr_bench.data.thaiocrbench import TaskNameError, prepare_thaiocrbench
from ocr_bench.jsonl import latest_predictions, read_rows, write_rows_atomic
from ocr_bench.metrics import tob_official
from ocr_bench.metrics.aggregate import aggregate
from ocr_bench.metrics.dispatch import score_run
from ocr_bench.metrics.duplicates import TARGET_ORIGINALS, build_dupset, eval_dupset
from ocr_bench.metrics.statement_run import file_id_for, map_pages, run_statement_checks
from ocr_bench.models import registry
from ocr_bench.models.base import OcrModel
from ocr_bench.models.probe import build_probe_report, probe_model, select_probe_pages
from ocr_bench.models.runner import infer_model, pending_rows, summarize
from ocr_bench.models.vllm_client import EndpointUnavailable
from ocr_bench.paths import MissingArtifactError, RunPaths, require
from ocr_bench.schemas import (
    GROUND_TRUTH,
    Condition,
    GroundTruth,
    JsonGT,
    ManifestRow,
    NoGT,
    PredictionRow,
    PrepareSummary,
    Task,
)
from ocr_bench.statement_review import (
    AGREEMENT_SAMPLE,
    LabelError,
    load_labels,
    load_manual_pages,
    queue_rows,
    write_manual_gt,
    write_queue,
)

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
    rp.config_resolved.write_text(resolved.to_prepare_yaml(), encoding="utf-8")

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


def _load_models(config: Path, models: str) -> tuple[ResolvedConfig, dict[str, ModelConfig]]:
    """Resolve `--models` (default: run.yaml's list) to model configs; exit 2 on errors.

    A model not listed in run.yaml is loaded from `configs/models/<name>.yaml` directly.
    """
    try:
        resolved = load_config(config)
        names = _csv(models) or list(resolved.run.models)
        cfgs = {
            name: resolved.models.get(name)
            or load_model_config(config.parent / "models" / f"{name}.yaml", name)
            for name in names
        }
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    return resolved, cfgs


def _connect(cfgs: dict[str, ModelConfig]) -> dict[str, OcrModel]:
    """Build each model's adapter and health-check it; exit 2 naming the model and URL."""
    try:
        built = {name: registry.build_model(cfg) for name, cfg in cfgs.items()}
    except EndpointUnavailable as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    async def check_all() -> None:
        for model in built.values():
            await model.health()

    try:
        asyncio.run(check_all())
    except EndpointUnavailable as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    return built


def _write_infer_config(rp: RunPaths, cfgs: dict[str, ModelConfig], concurrency: int) -> None:
    """Record the model settings used; a later run for other models adds to the file."""
    snapshot: dict = {}
    if rp.config_infer.exists():
        snapshot = yaml.safe_load(rp.config_infer.read_text(encoding="utf-8")) or {}
    snapshot["concurrency"] = concurrency
    snapshot.setdefault("models", {})
    for name, cfg in cfgs.items():
        snapshot["models"][name] = cfg.model_dump(mode="json")
    rp.config_infer.write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _progress_printer(name: str, total: int) -> Callable[[PredictionRow], None]:
    done = 0

    def on_row(_: PredictionRow) -> None:
        nonlocal done
        done += 1
        if done % 100 == 0 or done == total:
            typer.echo(f"{name}: {done}/{total}", err=True)

    return on_row


@app.command()
def infer(
    config: Path = typer.Option(  # noqa: B008 — typer's own default-injection idiom
        DEFAULT_CONFIG_PATH, "--config", help="Path to run.yaml."
    ),
    run_id: str = typer.Option(..., "--run-id", help="Run id to infer over."),
    models: str = typer.Option(
        "", "--models", help="Comma-separated model names (default: run.yaml's models)."
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", min=1, help="Pages in flight per model (default: run.yaml's)."
    ),
) -> None:
    """Run each model over the manifest and record predictions."""
    rp = RunPaths.for_run(run_id)
    _require_or_exit(rp.manifest, "prepare")
    resolved, cfgs = _load_models(config, models)
    n_concurrent = concurrency or resolved.run.concurrency
    built = _connect(cfgs)
    _write_infer_config(rp, cfgs, n_concurrent)

    rows = list(read_rows(rp.manifest, ManifestRow))
    # Models run one after another: they may share a GPU, and page latency_ms should
    # not include contention from another model's server.
    for name, model in built.items():
        todo = pending_rows(rows, name, latest_predictions(rp.predictions))
        typer.echo(f"{name}: {len(rows) - len(todo)} of {len(rows)} rows done, {len(todo)} to run")
        on_row = _progress_printer(name, len(todo))
        asyncio.run(infer_model(model, todo, rp.root, rp.predictions, n_concurrent, on_row))

    latest = latest_predictions(rp.predictions)
    mismatched = []
    for name in built:
        summary = summarize(name, rows, latest)
        typer.echo(summary.line())
        if summary.predicted != summary.pages:
            mismatched.append(f"{name} ({summary.predicted} of {summary.pages})")
    if mismatched:
        typer.echo(
            "row count mismatch between manifest and predictions: " + ", ".join(mismatched),
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def probe(
    config: Path = typer.Option(  # noqa: B008 — typer's own default-injection idiom
        DEFAULT_CONFIG_PATH, "--config", help="Path to run.yaml."
    ),
    run_id: str = typer.Option(..., "--run-id", help="Run id to probe."),
    models: str = typer.Option(
        "", "--models", help="Comma-separated model names (default: run.yaml's models)."
    ),
) -> None:
    """Probe each model's structural output capabilities."""
    rp = RunPaths.for_run(run_id)
    _require_or_exit(rp.manifest, "prepare")
    _, cfgs = _load_models(config, models)
    pages = select_probe_pages(list(read_rows(rp.manifest, ManifestRow)))
    if not pages:
        typer.echo(f"no clean bank statement pages in {rp.manifest} to probe", err=True)
        raise typer.Exit(code=2)
    built = _connect(cfgs)

    per_model = {}
    for name, model in built.items():
        typer.echo(f"{name}: probing {len(pages)} statement pages")
        per_model[name] = asyncio.run(probe_model(model, pages, rp.root))
    versions = {name: cfg.plan_for(Task.statement, None).version for name, cfg in cfgs.items()}
    report = build_probe_report(per_model, pages, versions)

    write_rows_atomic(rp.probe_predictions, [p for s, j in per_model.values() for p in (*s, *j)])
    rp.probe.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for capability, cells in report["matrix"].items():
        typer.echo(f"{capability}: " + " ".join(f"{m}={v}" for m, v in cells.items()))


@app.command()
def score(
    run_id: str = typer.Option(..., "--run-id", help="Run id to score."),
) -> None:
    """Score predictions against ground truth and write scores/fields rows.

    Writes scores.jsonl, fields.jsonl, aggregates.jsonl and tob_parity.json.
    """
    rp = RunPaths.for_run(run_id)
    _require_or_exit(rp.manifest, "prepare")
    _require_or_exit(rp.predictions, "infer")

    manifest = list(read_rows(rp.manifest, ManifestRow))
    latest = latest_predictions(rp.predictions)
    known = {(r.sample_id, r.condition.value) for r in manifest}
    stray = sum(1 for (sid, cond, _) in latest if (sid, cond) not in known)
    if stray:
        typer.echo(f"warning: {stray} prediction rows match no manifest row; ignored", err=True)

    gt_cache: dict[str, GroundTruth] = {}
    manual = load_manual_pages(rp)

    def load_gt(row: ManifestRow) -> GroundTruth:
        # A manually labelled page is ground truth for every condition of that page, in
        # preference to its text layer (or to having none at all).
        if row.sample_id in manual:
            return JsonGT(gt_kind="json", statement=manual[row.sample_id])
        if row.gt_path is None:
            return NoGT(gt_kind="none")
        if row.gt_path not in gt_cache:
            text = (rp.root / row.gt_path).read_text(encoding="utf-8")
            gt_cache[row.gt_path] = GROUND_TRUTH.validate_json(text)
        return gt_cache[row.gt_path]

    overrides = _bank_overrides(rp)
    scores, fields = score_run(manifest, latest, load_gt, overrides)
    statement_scores, statements = run_statement_checks(manifest, latest, overrides, manual)
    scores = sorted(
        [*scores, *statement_scores],
        key=lambda s: (s.sample_id, s.condition.value, s.model, s.metric),
    )
    write_rows_atomic(rp.scores, scores)
    write_rows_atomic(rp.fields, fields)
    write_rows_atomic(rp.statements, statements)

    run_cfg = _run_config_for(rp)
    aggregates = aggregate(scores, fields, run_cfg.bootstrap_resamples, run_cfg.seed)
    write_rows_atomic(rp.aggregates, aggregates)

    parity = tob_official.run_parity()
    rp.tob_parity.write_text(
        json.dumps(parity, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    not_comparable = sorted(t for t, info in parity.items() if not info["comparable"])
    missing = sum(1 for s in scores if s.extra.get("missing_prediction"))
    typer.echo(
        f"scored {len(manifest)} manifest rows: {len(scores)} score rows, "
        f"{len(fields)} field rows, {len(aggregates)} aggregate rows, "
        f"{len(statements)} statement files"
    )
    if missing:
        typer.echo(f"warning: {missing} score rows are for missing predictions", err=True)
    if not_comparable:
        typer.echo(f"tob_score not comparable for: {', '.join(not_comparable)}", err=True)


def _bank_overrides(rp: RunPaths) -> dict[str, BankOverrides]:
    """The statement dataset's per-bank parsing overrides from the run's resolved config."""
    if not rp.config_resolved.exists():
        return {}
    data = yaml.safe_load(rp.config_resolved.read_text(encoding="utf-8")) or {}
    for dataset in (data.get("datasets") or {}).values():
        if isinstance(dataset, dict) and dataset.get("kind") == "bankstmt":
            return BankStmtConfig.model_validate(dataset).overrides
    return {}


def _run_config_for(rp: RunPaths) -> RunConfig:
    """The run section of the run's `config.resolved.yaml` (defaults if absent)."""
    if rp.config_resolved.exists():
        data = yaml.safe_load(rp.config_resolved.read_text(encoding="utf-8")) or {}
        if "run" in data:
            return RunConfig.model_validate(data["run"])
    return RunConfig(models=[], datasets=[])


review_app = typer.Typer(
    name="review",
    no_args_is_help=True,
    help="Statement review queue and manual anchor labels.",
)
app.add_typer(review_app, name="review")


@review_app.command("export")
def review_export(
    run_id: str = typer.Option(..., "--run-id", help="Run id to export a review queue for."),
    condition: Condition = typer.Option(  # noqa: B008 — typer's own default-injection idiom
        Condition.clean, "--condition", help="Condition whose pages are reviewed."
    ),
    agreement_sample: float = typer.Option(
        AGREEMENT_SAMPLE,
        "--agreement-sample",
        min=0.0,
        max=1.0,
        help="Share of agreeing cells to sample (disagreements are always exported).",
    ),
) -> None:
    """Write `runs/<id>/review/queue.csv`: every cross-model disagreement plus a seeded
    10% sample of the agreements, with one value column per model and an empty label."""
    rp = RunPaths.for_run(run_id)
    _require_or_exit(rp.manifest, "prepare")
    _require_or_exit(rp.predictions, "infer")

    manifest = [r for r in read_rows(rp.manifest, ManifestRow) if r.task == Task.statement]
    rows = [r for r in manifest if r.condition == condition]
    latest = latest_predictions(rp.predictions)
    models = sorted({model for (_, _, model) in latest})
    if not models:
        typer.echo("no predictions to review", err=True)
        raise typer.Exit(code=2)

    pages = map_pages(rows, latest, _bank_overrides(rp), models)
    run_cfg = _run_config_for(rp)
    queue = queue_rows(
        {(sample_id, model): page for (sample_id, _, model), page in pages.items()},
        models,
        file_ids={r.sample_id: file_id_for(r.sample_id) for r in rows},
        page_numbers={r.sample_id: r.page_no for r in rows},
        seed=run_cfg.seed,
        agreement_sample=agreement_sample,
    )
    n = write_queue(rp.review_queue, queue, models)
    typer.echo(f"wrote {n} review rows for {len(rows)} pages to {rp.review_queue}")


@review_app.command("load")
def review_load(
    run_id: str = typer.Option(..., "--run-id", help="Run id the labels belong to."),
    labels: Path = typer.Option(  # noqa: B008 — typer's own default-injection idiom
        ..., "--labels", help="Filled queue CSV to load."
    ),
) -> None:
    """Validate a filled label CSV and store it as statement ground truth."""
    rp = RunPaths.for_run(run_id)
    _require_or_exit(rp.manifest, "prepare")
    if not labels.exists():
        typer.echo(f"labels file not found: {labels}", err=True)
        raise typer.Exit(code=2)
    try:
        pages = load_labels(labels)
    except LabelError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    written = write_manual_gt(rp, pages)
    rows = sum(len(p.rows) for p in pages.values())
    typer.echo(f"loaded labels for {len(written)} pages ({rows} rows) into {rp.gt_manual}")


dupset_app = typer.Typer(
    name="dupset",
    no_args_is_help=True,
    help="Duplicate-statement test set: build it, then evaluate the detector.",
)
app.add_typer(dupset_app, name="dupset")


@dupset_app.command("build")
def dupset_build(
    run_id: str = typer.Option(..., "--run-id", help="Source run to draw statements from."),
    dup_run_id: str = typer.Option(..., "--dup-run-id", help="Run id to create."),
    seed: int | None = typer.Option(None, "--seed", help="Selection seed (default: run.yaml's)."),
    n_originals: int = typer.Option(
        TARGET_ORIGINALS, "--n-originals", min=1, help="Originals to draw."
    ),
) -> None:
    """Build a separate run of originals, variants and hard negatives for `infer`."""
    src = RunPaths.for_run(run_id)
    _require_or_exit(src.manifest, "prepare")
    dst = RunPaths.for_run(dup_run_id)
    dupset = build_dupset(
        src,
        dst,
        seed=seed if seed is not None else _run_config_for(src).seed,
        n_originals=n_originals,
    )
    for warning in dupset.warnings:
        typer.echo(f"warning: {warning}", err=True)
    typer.echo(
        f"built {dupset.n_originals} originals, {len(dupset.pairs)} pairs "
        f"({dupset.n_negatives} negatives) in {dst.root}; run `ocrbench infer --run-id "
        f"{dup_run_id}` next"
    )


@dupset_app.command("eval")
def dupset_eval(
    dup_run_id: str = typer.Option(..., "--dup-run-id", help="Duplicate run to evaluate."),
) -> None:
    """Evaluate duplicate detection over the built set's predictions."""
    dst = RunPaths.for_run(dup_run_id)
    _require_or_exit(dst.dupset, "dupset build")
    _require_or_exit(dst.predictions, "infer")
    rows, details = eval_dupset(dst, overrides=_bank_overrides(dst))
    for row in rows:
        typer.echo(
            f"{row.model} t={row.threshold}: recall="
            f"{'n/a' if row.recall is None else f'{row.recall:.2f}'} fpr="
            f"{'n/a' if row.fpr is None else f'{row.fpr:.2f}'}"
        )
    typer.echo(f"wrote {len(rows)} sweep rows and {len(details)} pair rows to {dst.root}")


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
