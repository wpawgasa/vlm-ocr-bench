"""Confidence proxy fitting, bands/ECE and threshold search (tasks 6.2-6.4, D10).

`collect_field_records` rebuilds one `FieldRecord` (features + label) per labelled
field of a scored run: ThaiOCRBench kie/kie_map fields (`fields.jsonl` already carries
only ground-truthed rows — see `metrics.dispatch.score_fields`) and statement header
fields on usable digital/manual pages (`metrics.dispatch._metric_values`'s statement
branch, gated on `_has_statement_gt`). `fit_variant` fits the logistic proxy per model
per variant with grouped cross-validation. `run_calibrate` orchestrates the whole
`ocrbench calibrate` command and writes the four run-dir artifacts.
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ocr_bench.confidence.proxy import FieldFeatures, Variant, feature_names, field_features
from ocr_bench.config import BankOverrides, BankStmtConfig
from ocr_bench.jsonl import latest_predictions, read_rows, write_rows_atomic
from ocr_bench.metrics.aggregate import cluster_bootstrap
from ocr_bench.metrics.arithmetic import ArithmeticResult
from ocr_bench.metrics.statement_gt import SCORED_FIELDS
from ocr_bench.metrics.statement_run import StatementRecord, file_id_for, map_pages
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import (
    CalibrationFieldRow,
    CalibrationRow,
    Condition,
    FieldResult,
    ManifestRow,
    NormalizedPage,
    PredictionRow,
    RawPrediction,
    Task,
)

MIN_FIELDS = 50
MIN_DISTINCT_SAMPLES = 5
VARIANTS: tuple[Variant, ...] = ("full", "logprob_only")
TARGET_FMT = "{:g}"  # "0.99" not "0.9900000000000001"

BAND_RANGES: dict[int, tuple[float, float]] = {
    5: (0.98, 1.0),
    4: (0.90, 0.98),
    3: (0.75, 0.90),
    2: (0.50, 0.75),
    1: (0.0, 0.50),
}


def _band_for(conf: float) -> int:
    if conf >= 0.98:
        return 5
    if conf >= 0.90:
        return 4
    if conf >= 0.75:
        return 3
    if conf >= 0.50:
        return 2
    return 1


# --- feature-row collection -----------------------------------------------------------------


@dataclass
class FieldRecord:
    sample_id: str
    condition: str
    model: str
    field: str
    is_critical: bool
    label: int
    features: FieldFeatures


def _bank_overrides(rp: RunPaths) -> dict[str, BankOverrides]:
    """The statement dataset's per-bank parsing overrides, duplicated from
    `cli._bank_overrides` (kept import-direction clean: `confidence` never imports
    `cli`, so this tiny piece of config parsing is repeated rather than shared)."""
    if not rp.config_resolved.exists():
        return {}
    data = yaml.safe_load(rp.config_resolved.read_text(encoding="utf-8")) or {}
    for dataset in (data.get("datasets") or {}).values():
        if isinstance(dataset, dict) and dataset.get("kind") == "bankstmt":
            return BankStmtConfig.model_validate(dataset).overrides
    return {}


def _blank_pred_row(sample_id: str, condition: str, model: str) -> PredictionRow:
    """A stand-in for a field with no recorded prediction (e.g. a full miss): empty
    tokens/logprobs, so `field_features` degrades to `span_found=False`,
    `logprobs_missing=True` rather than raising."""
    return PredictionRow(
        sample_id=sample_id,
        condition=Condition(condition),
        model=model,
        prompt_version="",
        raw=RawPrediction(),
        normalized=NormalizedPage(),
        latency_ms=0.0,
        prompt_tokens=0,
        completion_tokens=0,
    )


def _value_raw(
    row: FieldResult,
    pred_row: PredictionRow | None,
    pages: dict[tuple[str, str, str], object],
) -> str | None:
    """The field's raw (pre-normalization) predicted string, for span location."""
    if pred_row is None:
        return None
    if row.task in (Task.kie, Task.kie_map):
        fv = pred_row.normalized.fields.get(row.field)
        if fv is None:
            return None
        return fv.raw if fv.raw is not None else fv.value
    if row.task == Task.statement and row.field in SCORED_FIELDS:
        page = pages.get((row.sample_id, row.condition.value, row.model))
        if page is None:
            return None
        value = getattr(page, row.field, None)
        if isinstance(value, Decimal):
            return format(value, "f")
        return value
    return None


def _arith_flag(
    row: FieldResult, arithmetic: dict[tuple[str, str, str], ArithmeticResult]
) -> bool | None:
    """Check B's arithmetic-consistency flag for fields that take part in a checked
    equation.

    [IMPLEMENTER DECIDES] Of the statement header fields scored (`SCORED_FIELDS`:
    account_no, account_name, period_start, period_end, opening_balance,
    closing_balance), only `closing_balance` is itself checked by an equation (Check
    B's running-balance chain closes against it); `opening_balance` is the chain's
    basis rather than a value the chain verifies, and the printed summary totals are
    not scored as header fields at all. So `arith` is populated only for
    `closing_balance`, from that file's `ArithmeticResult.closing_mismatch`.
    """
    if row.task != Task.statement or row.field != "closing_balance":
        return None
    result = arithmetic.get((file_id_for(row.sample_id), row.condition.value, row.model))
    if result is None:
        return None
    return not result.closing_mismatch


def collect_field_records(rp: RunPaths) -> list[FieldRecord]:
    """One `FieldRecord` per row of `fields.jsonl` (the labelled fields; see module
    docstring for which rows that already is)."""
    fields = list(read_rows(rp.fields, FieldResult))
    if not fields:
        return []

    manifest = list(read_rows(rp.manifest, ManifestRow)) if rp.manifest.exists() else []
    latest = latest_predictions(rp.predictions)
    overrides = _bank_overrides(rp)

    statement_manifest = [r for r in manifest if r.task == Task.statement]
    models = sorted({f.model for f in fields})
    pages = map_pages(statement_manifest, latest, overrides, models) if statement_manifest else {}

    arithmetic: dict[tuple[str, str, str], ArithmeticResult] = {}
    if rp.statements.exists():
        for rec in read_rows(rp.statements, StatementRecord):
            arithmetic[(rec.file_id, rec.condition.value, rec.model)] = rec.arithmetic

    # (sample_id, condition, field) -> {model: normalized pred value}
    other_values: dict[tuple[str, str, str], dict[str, str | None]] = defaultdict(dict)
    for f in fields:
        other_values[(f.sample_id, f.condition.value, f.field)][f.model] = f.pred

    records: list[FieldRecord] = []
    for f in fields:
        pred_row = latest.get((f.sample_id, f.condition.value, f.model))
        value_raw = _value_raw(f, pred_row, pages)
        arith_flag = _arith_flag(f, arithmetic)
        others = {
            m: v
            for m, v in other_values[(f.sample_id, f.condition.value, f.field)].items()
            if m != f.model
        }
        feats = field_features(
            pred_row
            if pred_row is not None
            else _blank_pred_row(f.sample_id, f.condition.value, f.model),
            f.field,
            value_raw,
            other_value_by_model=others,
            arith_flag=arith_flag,
        )
        records.append(
            FieldRecord(
                sample_id=f.sample_id,
                condition=f.condition.value,
                model=f.model,
                field=f.field,
                is_critical=f.is_critical,
                label=f.field_exact,
                features=feats,
            )
        )
    return records


# --- 6.2: fit ---------------------------------------------------------------------------------


@dataclass
class Weights:
    feature_names: tuple[str, ...]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float

    def predict(self, x: list[float]) -> float:
        """`conf(f) = sigma(w0 + w . scaled(x))`, recomputed from raw features (D10)."""
        z = self.intercept
        for c, xi, m, s in zip(self.coef, x, self.mean, self.scale, strict=True):
            z += c * ((xi - m) / s if s else 0.0)
        return 1.0 / (1.0 + math.exp(-z))

    def to_json(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "scaler": {"mean": list(self.mean), "scale": list(self.scale)},
            "coef": list(self.coef),
            "intercept": self.intercept,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Weights":
        return cls(
            feature_names=tuple(data["feature_names"]),
            mean=list(data["scaler"]["mean"]),
            scale=list(data["scaler"]["scale"]),
            coef=list(data["coef"]),
            intercept=float(data["intercept"]),
        )


@dataclass
class FitResult:
    status: str  # "ok" | "insufficient_data"
    n: int
    classes: int
    oof_conf: list[float | None]  # aligned with the input records list
    weights: Weights | None


def grouped_folds(sample_ids: list[str], n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """5-fold `GroupKFold` splits over `sample_ids`: no sample_id ever spans a train and
    a held-out fold (D10)."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(sample_ids)), groups=np.array(sample_ids)))


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000)),
        ]
    )


def fit_variant(records: list[FieldRecord], variant: Variant, seed: int = 0) -> FitResult:
    """Grouped 5-fold out-of-fold confidences, plus a final fit on all data for
    `thresholds.json`'s weights (D10). `seed` seeds nothing in the fit itself (logistic
    regression here is deterministic); it is accepted for a stable call signature
    alongside the rest of the module's seeded functions."""
    n = len(records)
    labels = [r.label for r in records]
    sample_ids = [r.sample_id for r in records]
    classes = len(set(labels))
    n_groups = len(set(sample_ids))

    if n < MIN_FIELDS or n_groups < MIN_DISTINCT_SAMPLES or classes < 2:
        return FitResult(
            status="insufficient_data", n=n, classes=classes, oof_conf=[None] * n, weights=None
        )

    x = np.array([r.features.vector(variant) for r in records])
    y = np.array(labels)

    oof = np.full(n, np.nan)
    for train_idx, test_idx in grouped_folds(sample_ids, n_splits=5):
        pipe = _make_pipeline()
        pipe.fit(x[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict_proba(x[test_idx])[:, 1]

    final = _make_pipeline()
    final.fit(x, y)
    scaler: StandardScaler = final.named_steps["scaler"]
    clf: LogisticRegression = final.named_steps["clf"]
    weights = Weights(
        feature_names=feature_names(variant),
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coef=clf.coef_[0].tolist(),
        intercept=float(clf.intercept_[0]),
    )
    return FitResult(status="ok", n=n, classes=classes, oof_conf=oof.tolist(), weights=weights)


# --- 6.3: bands and ECE ------------------------------------------------------------------------


def compute_ece(confs: list[float], labels: list[int], n_bins: int = 10) -> float:
    """`sum(n_b/N * |acc_b - mean_conf_b|)` over `n_bins` equal-width bins on [0, 1]."""
    n = len(confs)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j in range(n) if confs[j] >= lo and (confs[j] < hi or i == n_bins - 1)]
        if not idx:
            continue
        acc = sum(labels[j] for j in idx) / len(idx)
        mean_conf = sum(confs[j] for j in idx) / len(idx)
        total += (len(idx) / n) * abs(acc - mean_conf)
    return total


def reliability_bins(confs: list[float], labels: list[int], n_bins: int = 10) -> list[dict]:
    """Per-bin `{lo, hi, n, acc, mean_conf}`, the same 10 equal-width bins as `compute_ece`."""
    n = len(confs)
    bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j in range(n) if confs[j] >= lo and (confs[j] < hi or i == n_bins - 1)]
        if idx:
            acc = sum(labels[j] for j in idx) / len(idx)
            mean_conf = sum(confs[j] for j in idx) / len(idx)
        else:
            acc = None
            mean_conf = None
        bins.append({"lo": lo, "hi": hi, "n": len(idx), "acc": acc, "mean_conf": mean_conf})
    return bins


def band_rows(
    model: str,
    variant: Variant,
    confs: list[float],
    labels: list[int],
    sample_ids: list[str],
    seed: int,
    table: str,
    ece: float,
    n_resamples: int = 1000,
) -> list[CalibrationRow]:
    """The 5-band table for one (model, variant, table); n sums to `len(confs)`. The CI
    resamples whole `sample_id` clusters (D10 / handover 6.3), matching
    `metrics.aggregate.cluster_bootstrap`."""
    rows = []
    for band in sorted(BAND_RANGES, reverse=True):
        lo, hi = BAND_RANGES[band]
        idx = [j for j in range(len(confs)) if _band_for(confs[j]) == band]
        clusters: dict[str, list[float]] = defaultdict(list)
        for j in idx:
            clusters[sample_ids[j]].append(float(labels[j]))
        mean, ci_lo, ci_hi = cluster_bootstrap(clusters, n=n_resamples, seed=seed)
        rows.append(
            CalibrationRow(
                model=model,
                variant=variant,
                band=band,
                lo=lo,
                hi=hi,
                n=len(idx),
                accuracy=mean,
                ci_low=ci_lo,
                ci_high=ci_hi,
                critical_only=(table == "critical"),
                table=table,
                ece=ece,
            )
        )
    return rows


# --- 6.4: threshold search ---------------------------------------------------------------------


def search_threshold(confs: list[float], labels: list[int], target: float) -> dict:
    """`t_accept`/`t_reject` and bucket shares/accuracies for one target accuracy (D10).

    `t_accept` is the smallest candidate threshold whose accepted set {conf >= c}
    reaches `target` accuracy (maximising the auto-accept share); `t_reject` is the
    largest candidate <= t_accept (or <= 1 when unreachable) whose rejected set
    {conf < c} is non-empty with accuracy <= 0.5.
    """
    n = len(confs)
    uniq_asc = sorted(set(confs))

    t_accept = None
    for c in uniq_asc:
        idx = [j for j in range(n) if confs[j] >= c]
        if not idx:
            continue
        acc = sum(labels[j] for j in idx) / len(idx)
        if acc >= target:
            t_accept = c
            break

    upper = t_accept if t_accept is not None else 1.0
    t_reject = None
    for c in sorted((c for c in uniq_asc if c <= upper), reverse=True):
        idx = [j for j in range(n) if confs[j] < c]
        if not idx:
            continue
        acc = sum(labels[j] for j in idx) / len(idx)
        if acc <= 0.5:
            t_reject = c
            break

    accept_idx = {j for j in range(n) if t_accept is not None and confs[j] >= t_accept}
    reject_idx = {j for j in range(n) if t_reject is not None and confs[j] < t_reject}
    review_idx = [j for j in range(n) if j not in accept_idx and j not in reject_idx]

    def stats(idx) -> tuple[float, float | None]:
        share = len(idx) / n if n else 0.0
        acc = sum(labels[j] for j in idx) / len(idx) if idx else None
        return share, acc

    accept_share, accept_acc = stats(accept_idx)
    reject_share, reject_acc = stats(reject_idx)
    review_share, review_acc = stats(review_idx)
    return {
        "t_accept": t_accept,
        "t_reject": t_reject,
        "reachable": t_accept is not None,
        "shares": {"accept": accept_share, "review": review_share, "reject": reject_share},
        "accuracy": {"accept": accept_acc, "review": review_acc, "reject": reject_acc},
    }


# --- orchestration -----------------------------------------------------------------------------


def run_calibrate(
    rp: RunPaths, target_accs: list[float], seed: int = 0, n_resamples: int = 1000
) -> dict:
    """Fit every model/variant, write the four run-dir artifacts, and return a summary
    dict the CLI prints from."""
    records = collect_field_records(rp)
    by_model: dict[str, list[FieldRecord]] = defaultdict(list)
    for r in records:
        by_model[r.model].append(r)

    calibration_rows: list[CalibrationRow] = []
    field_rows: list[CalibrationFieldRow] = []
    reliability: dict[str, dict[str, list[dict]]] = {}
    thresholds: dict[str, dict[str, dict]] = {}
    per_model_summary: dict[str, dict] = {}

    for model in sorted(by_model):
        model_records = by_model[model]
        thresholds[model] = {}
        reliability[model] = {}
        per_model_summary[model] = {}
        labels = [r.label for r in model_records]
        sample_ids = [r.sample_id for r in model_records]

        for variant in VARIANTS:
            fit = fit_variant(model_records, variant, seed=seed)

            for rec, conf in zip(model_records, fit.oof_conf, strict=True):
                field_rows.append(
                    CalibrationFieldRow(
                        sample_id=rec.sample_id,
                        condition=Condition(rec.condition),
                        model=rec.model,
                        variant=variant,
                        field=rec.field,
                        is_critical=rec.is_critical,
                        label=rec.label,
                        conf=conf,
                        mean_logprob=rec.features.mean_logprob,
                        min_logprob=rec.features.min_logprob,
                        span_found=rec.features.span_found,
                        logprobs_missing=rec.features.logprobs_missing,
                        agree=rec.features.agree,
                        arith=rec.features.arith,
                    )
                )

            if fit.status != "ok":
                thresholds[model][variant] = {
                    "status": "insufficient_data",
                    "n": fit.n,
                    "classes": fit.classes,
                }
                continue

            confs = fit.oof_conf
            ece_all = compute_ece(confs, labels)
            calibration_rows.extend(
                band_rows(
                    model, variant, confs, labels, sample_ids, seed, "all", ece_all, n_resamples
                )
            )

            crit_idx = [i for i, r in enumerate(model_records) if r.is_critical]
            if crit_idx:
                c_confs = [confs[i] for i in crit_idx]
                c_labels = [labels[i] for i in crit_idx]
                c_sids = [sample_ids[i] for i in crit_idx]
                ece_crit = compute_ece(c_confs, c_labels)
                calibration_rows.extend(
                    band_rows(
                        model,
                        variant,
                        c_confs,
                        c_labels,
                        c_sids,
                        seed,
                        "critical",
                        ece_crit,
                        n_resamples,
                    )
                )

            reliability[model][variant] = reliability_bins(confs, labels)

            targets_out = {}
            for target in target_accs:
                targets_out[TARGET_FMT.format(target)] = search_threshold(confs, labels, target)
            thresholds[model][variant] = {"weights": fit.weights.to_json(), "targets": targets_out}
            per_model_summary[model][variant] = {"ece": ece_all, "targets": targets_out}

    write_rows_atomic(rp.calibration, calibration_rows)
    write_rows_atomic(rp.calibration_fields, field_rows)
    rp.reliability.write_text(
        json.dumps(reliability, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rp.thresholds.write_text(json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "n_records": len(records),
        "n_calibration_rows": len(calibration_rows),
        "per_model": per_model_summary,
    }
