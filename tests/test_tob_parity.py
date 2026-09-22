"""Parity of the ported official scorer with ThaiOCRBench's published scores (task 4.4).

Every fixture sample must be within 0.01 of its published score. The threshold is never
loosened: a task that cannot reach it is marked xfail here (with the measured maximum
difference) and reported not comparable by `run_parity()`.
"""

import os

import pytest

from ocr_bench.metrics.tob_official import (
    KEPT_TASKS,
    PARITY_THRESHOLD,
    load_parity_fixture,
    parity_diffs,
    run_parity,
    wordnet_available,
)

# Tasks whose parity could not be reached, with the reason: {task: reason}.
KNOWN_NOT_COMPARABLE: dict[str, str] = {}


@pytest.fixture(scope="module", autouse=True)
def _require_wordnet():
    if wordnet_available():
        return
    msg = "NLTK WordNet missing: run `uv run python -m nltk.downloader wordnet omw-1.4`"
    if os.environ.get("CI"):
        pytest.fail(msg)  # the gate must never be silently skipped in CI
    pytest.skip(msg)


def test_fixture_has_at_least_20_samples_per_task():
    fixture = load_parity_fixture()
    for task in KEPT_TASKS:
        assert sum(item["task"] == task for item in fixture) >= 20, task


@pytest.mark.parametrize("task", KEPT_TASKS)
def test_task_parity_within_threshold(task):
    if task in KNOWN_NOT_COMPARABLE:
        pytest.xfail(KNOWN_NOT_COMPARABLE[task])
    rows = parity_diffs(task)
    off = [
        f"  {sid}: published={pub:.6f} ours={ours:.6f} diff={abs(ours - pub):.6f}"
        for sid, pub, ours in rows
        if abs(ours - pub) > PARITY_THRESHOLD
    ]
    assert not off, f"{task}: {len(off)}/{len(rows)} samples off by > {PARITY_THRESHOLD}:\n" + (
        "\n".join(off)
    )


def test_run_parity_reports_every_task():
    result = run_parity()
    assert set(result) == set(KEPT_TASKS)
    for task, info in result.items():
        assert info["n"] >= 20
        assert info["comparable"] == (task not in KNOWN_NOT_COMPARABLE)
