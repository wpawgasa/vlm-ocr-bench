"""Tests for ocr_bench.metrics.aggregate (task 4.8)."""

import numpy as np
import pytest

from ocr_bench.metrics.aggregate import aggregate, bootstrap_ci, cluster_bootstrap
from ocr_bench.schemas import Condition, FieldResult, ScoreRow, Task


def test_ci_brackets_mean():
    values = np.random.default_rng(1).uniform(0, 1, size=50).tolist()
    mean, lo, hi = bootstrap_ci(values, n=1000, seed=0)
    assert mean == pytest.approx(float(np.mean(values)))
    assert lo < mean < hi


def test_constant_series_has_degenerate_ci():
    assert bootstrap_ci([0.7] * 20) == (pytest.approx(0.7), pytest.approx(0.7), pytest.approx(0.7))


def test_bootstrap_deterministic_and_seed_sensitive():
    values = [0.1, 0.5, 0.9, 0.3, 0.2, 0.8]
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)
    assert bootstrap_ci(values, seed=3) != bootstrap_ci(values, seed=4)


def test_empty_series():
    assert bootstrap_ci([]) == (None, None, None)


def test_cluster_bootstrap_micro_average():
    # cluster a: 2 of 2 correct, cluster b: 0 of 1 -> micro mean 2/3
    mean, lo, hi = cluster_bootstrap({"a": [1, 1], "b": [0]}, n=200, seed=0)
    assert mean == pytest.approx(2 / 3)
    assert lo <= mean <= hi


def _score(sid, cond, model, task, metric, value, **kw):
    return ScoreRow(
        sample_id=sid,
        condition=cond,
        model=model,
        task=task,
        metric=metric,
        value=value,
        **kw,
    )


def _field(sid, model, field, exact, critical, gt="1", fa=False, fr=False):
    return FieldResult(
        sample_id=sid,
        condition=Condition.clean,
        model=model,
        task=Task.kie,
        field=field,
        is_critical=critical,
        pred=None,
        gt=gt,
        field_exact=exact,
        field_fuzzy=exact,
        false_accept=fa,
        false_reject=fr,
    )


def test_aggregate_tables_have_mean_n_ci():
    scores = [
        _score(
            "s1", Condition.clean, "m", Task.ocr_line, "cer", 0.1, subtask="A", domain="Finance"
        ),
        _score("s2", Condition.clean, "m", Task.ocr_line, "cer", 0.3, subtask="B", domain="Arts"),
        _score(
            "s1", Condition.photo, "m", Task.ocr_line, "cer", 0.5, subtask="A", domain="Finance"
        ),
        _score(
            "s3", Condition.clean, "m", Task.statement, "cer", 0.2, bank="kbank", doc_type="digital"
        ),
        _score("s4", Condition.clean, "m", Task.statement, "no_gt", None, bank="scb"),
    ]
    fields = [
        _field("s1", "m", "a", 1, True),
        _field("s1", "m", "b", 0, False, fr=True),
        _field("s2", "m", "c", 0, False, gt=None, fa=True),
    ]
    rows = aggregate(scores, fields, n_resamples=200, seed=0)
    by = {(r.table, tuple(sorted(r.keys.items(), key=str)), r.metric): r for r in rows}

    tm = by[("task_model", (("model", "m"), ("task", "ocr_line")), "cer")]
    assert tm.n == 3 and tm.mean == pytest.approx(0.3)
    assert tm.ci_low <= tm.mean <= tm.ci_high

    sub = by[
        ("task_subtask_model", (("model", "m"), ("subtask", "A"), ("task", "ocr_line")), "cer")
    ]
    assert sub.n == 2

    cm = by[("condition_model", (("condition", "photo"), ("model", "m")), "cer")]
    assert cm.n == 1 and cm.mean == pytest.approx(0.5)

    bank = by[
        ("bank_doctype_model", (("bank", "kbank"), ("doc_type", "digital"), ("model", "m")), "cer")
    ]
    assert bank.n == 1

    dom = [r for r in rows if r.table == "domain_model"]
    assert {r.keys["domain"] for r in dom} == {"Finance"}

    crit = by[("critical_model", (("is_critical", True), ("model", "m")), "field_exact")]
    assert crit.n == 1 and crit.mean == 1.0
    fr = by[("critical_model", (("is_critical", False), ("model", "m")), "fr_rate")]
    assert fr.n == 1 and fr.mean == 1.0
    fa = by[("critical_model", (("is_critical", False), ("model", "m")), "fa_rate")]
    assert fa.n == 1 and fa.mean == 1.0

    # value-less metrics are never aggregated
    assert not any(r.metric == "no_gt" for r in rows)


def test_aggregate_is_deterministic():
    scores = [_score(f"s{i}", Condition.clean, "m", Task.kie, "kie_f1", i / 10) for i in range(10)]
    assert aggregate(scores, [], n_resamples=100, seed=1) == aggregate(
        scores, [], n_resamples=100, seed=1
    )
