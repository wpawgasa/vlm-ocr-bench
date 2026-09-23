"""Slice aggregation with seeded bootstrap confidence intervals (task 4.8).

Resampling is over samples, never over individual rows: every value is clustered by its
`sample_id` (so a sample's three conditions, or a page's many fields, move together) and
each resample draws whole clusters with replacement. The point estimate and every
resample mean are micro-averages (sum of values / number of values). With one value per
cluster this is the ordinary percentile bootstrap of the mean.

`n` in an aggregate row is the number of values averaged (score rows, or field rows for
the field tables).
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ocr_bench.schemas import AggregateRow, FieldResult, ScoreRow, Task

PRIORITY_DOMAINS = ("Government", "Finance")

CI = tuple[float | None, float | None, float | None]


def cluster_bootstrap(
    clusters: dict[str, Sequence[float]], n: int = 1000, seed: int = 0, alpha: float = 0.05
) -> CI:
    """(micro mean, ci_low, ci_high) from a percentile bootstrap over clusters."""
    keys = sorted(k for k, v in clusters.items() if len(v))
    if not keys:
        return None, None, None
    sums = np.array([float(np.sum(clusters[k])) for k in keys])
    counts = np.array([len(clusters[k]) for k in keys], dtype=float)
    mean = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Clamp float noise so a constant series gives ci_low == ci_high == mean exactly.
    return mean, float(min(lo, mean)), float(max(hi, mean))


def bootstrap_ci(values: Sequence[float], n: int = 1000, seed: int = 0, alpha: float = 0.05) -> CI:
    """(mean, ci_low, ci_high): percentile bootstrap of the mean, one value per sample."""
    return cluster_bootstrap({f"{i:09d}": [v] for i, v in enumerate(values)}, n, seed, alpha)


KeyFn = Callable[[ScoreRow], dict[str, str | bool | None] | None]


def _cells(
    rows: Iterable[tuple[dict[str, str | bool | None], str, str, float]],
) -> dict[tuple, dict[str, list[float]]]:
    """Group `(keys, metric, cluster, value)` into {(keys, metric): {cluster: [values]}}."""
    cells: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for keys, metric, cluster, value in rows:
        cells[(tuple(sorted(keys.items())), metric)][cluster].append(value)
    return cells


def _emit(
    table: str, cells: dict[tuple, dict[str, list[float]]], n_resamples: int, seed: int
) -> list[AggregateRow]:
    out = []
    for (key_items, metric), clusters in sorted(cells.items(), key=lambda kv: repr(kv[0])):
        mean, lo, hi = cluster_bootstrap(clusters, n_resamples, seed)
        out.append(
            AggregateRow(
                table=table,
                keys=dict(key_items),
                metric=metric,
                mean=mean,
                n=sum(len(v) for v in clusters.values()),
                ci_low=lo,
                ci_high=hi,
            )
        )
    return out


def _score_table(
    table: str, scores: Sequence[ScoreRow], key_fn: KeyFn, n_resamples: int, seed: int
) -> list[AggregateRow]:
    def rows():
        for s in scores:
            if s.value is None:
                continue
            keys = key_fn(s)
            if keys is not None:
                yield keys, s.metric, s.sample_id, s.value

    return _emit(table, _cells(rows()), n_resamples, seed)


def _field_table(fields: Sequence[FieldResult], n_resamples: int, seed: int) -> list[AggregateRow]:
    def rows():
        for f in fields:
            keys = {"is_critical": f.is_critical, "model": f.model}
            yield keys, "field_exact", f.sample_id, float(f.field_exact)
            yield keys, "field_fuzzy", f.sample_id, float(f.field_fuzzy)
            yield keys, "field_lenient", f.sample_id, float(f.field_lenient)
            if f.gt is None:  # false-accept opportunities: no true value
                yield keys, "fa_rate", f.sample_id, float(f.false_accept)
            else:  # false-reject opportunities: a true value exists
                yield keys, "fr_rate", f.sample_id, float(f.false_reject)

    return _emit("critical_model", _cells(rows()), n_resamples, seed)


def aggregate(
    scores: Sequence[ScoreRow],
    fields: Sequence[FieldResult],
    n_resamples: int = 1000,
    seed: int = 0,
) -> list[AggregateRow]:
    """Every slice table, each cell with mean, n and a 95% bootstrap CI."""
    subtasks: dict[Task, set[str | None]] = defaultdict(set)
    for s in scores:
        subtasks[s.task].add(s.subtask)
    multi_subtask = {t for t, subs in subtasks.items() if len(subs) > 1}

    tables: list[tuple[str, KeyFn]] = [
        ("task_model", lambda s: {"task": s.task.value, "model": s.model}),
        (
            "task_subtask_model",
            lambda s: (
                {"task": s.task.value, "subtask": s.subtask, "model": s.model}
                if s.task in multi_subtask
                else None
            ),
        ),
        ("condition_model", lambda s: {"condition": s.condition.value, "model": s.model}),
        # [IMPLEMENTER DECIDES] condition x model is also split by task, since pooling a
        # metric such as cer across tasks mixes very different page types.
        (
            "task_condition_model",
            lambda s: {"task": s.task.value, "condition": s.condition.value, "model": s.model},
        ),
        (
            "bank_doctype_model",
            lambda s: (
                {"bank": s.bank, "doc_type": s.doc_type, "model": s.model}
                if s.task == Task.statement
                else None
            ),
        ),
        # [IMPLEMENTER DECIDES] the domain slice is ThaiOCRBench's Government/Finance
        # categories (statements, also tagged Finance, have their own table), split by task.
        (
            "domain_model",
            lambda s: (
                {"domain": s.domain, "task": s.task.value, "model": s.model}
                if s.domain in PRIORITY_DOMAINS and s.task != Task.statement
                else None
            ),
        ),
    ]
    out: list[AggregateRow] = []
    for name, key_fn in tables:
        out.extend(_score_table(name, scores, key_fn, n_resamples, seed))
    out.extend(_field_table(fields, n_resamples, seed))
    return out
