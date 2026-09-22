"""TDD tests for the duplicate test set and its evaluation (task 5.7).

The source run is built from synthetic page images, never the client corpus.
"""

import json

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from ocr_bench.cli import app
from ocr_bench.data.manifest import PreparedSample, materialize
from ocr_bench.jsonl import read_rows, write_rows_atomic
from ocr_bench.metrics.duplicates import (
    THRESHOLDS,
    DuplicateEvalRow,
    DupPair,
    DupSet,
    build_dupset,
    evaluate,
    jaccard,
    row_set,
)
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import (
    Condition,
    ManifestRow,
    NoGT,
    Source,
    StatementPage,
    StatementRow,
    Task,
)
from tests.test_statement_score import _prediction

runner = CliRunner()
BANKS = ("kbank", "bbl", "ttb")


def _page_image(seed: int):
    rng = np.random.default_rng(seed)
    img = np.full((240, 180, 3), 255, dtype=np.uint8)
    for i in range(12):
        y = 12 + i * 18
        cv2.putText(
            img,
            f"row {i} {rng.integers(100, 999)}",
            (6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 0),
            1,
        )
    return img


def _sample(bank, index, page_no=1, n_pages=1):
    return PreparedSample(
        sample_id=f"BS-{bank}-{index:04d}-p{page_no}",
        source=Source.bankstmt,
        task=Task.statement,
        subtask=None,
        domain="Finance",
        bank=bank,
        doc_type="digital",
        page_no=page_no,
        n_pages=n_pages,
        question="",
        image=_page_image(hash((bank, index, page_no)) % 10_000),
        gt=NoGT(gt_kind="none"),
        critical_fields=[],
    )


@pytest.fixture
def src_run(tmp_path, monkeypatch):
    """A source run with 3 banks x 2 files, one of them two pages."""
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    rp = RunPaths.for_run("src")
    samples = []
    for bank in BANKS:
        samples.append(_sample(bank, 1))
        samples.append(_sample(bank, 2, page_no=1, n_pages=2))
        samples.append(_sample(bank, 2, page_no=2, n_pages=2))
    rows = materialize(samples, [Condition.clean], rp)
    write_rows_atomic(rp.manifest, rows)
    return rp


class TestBuild:
    def test_build_writes_a_runnable_manifest(self, src_run, tmp_path):
        dst = RunPaths.for_run("dup")
        dupset = build_dupset(src_run, dst, seed=42, n_originals=20)
        rows = list(read_rows(dst.manifest, ManifestRow))
        assert rows, "the duplicate run needs a manifest `infer` can process"
        assert {r.condition for r in rows} == {Condition.clean}
        assert all(r.task == Task.statement and r.gt_kind.value == "none" for r in rows)
        assert all((dst.root / r.image_path).exists() for r in rows)
        assert dupset.n_originals == 6  # every file of the source run, page 1 only

    def test_three_variants_per_original(self, src_run):
        dst = RunPaths.for_run("dup")
        dupset = build_dupset(src_run, dst, seed=42)
        variants = [p for p in dupset.pairs if p.kind == "variant"]
        assert len(variants) == 3 * dupset.n_originals
        assert {p.variant for p in variants} == {"scan_low", "rot180_photo", "crop"}

    def test_hard_negatives_are_same_bank_pairs(self, src_run):
        dst = RunPaths.for_run("dup")
        dupset = build_dupset(src_run, dst, seed=42)
        negatives = [p for p in dupset.pairs if p.kind == "negative"]
        assert negatives
        assert {p.negative_kind for p in negatives} == {"same_bank"}
        for pair in negatives:
            assert pair.original != pair.candidate
            assert pair.original.split("-")[1] == pair.candidate.split("-")[1]

    def test_short_corpus_is_recorded_not_fabricated(self, src_run):
        dst = RunPaths.for_run("dup")
        dupset = build_dupset(src_run, dst, seed=42, n_originals=20)
        assert dupset.n_originals < 20
        assert any("20" in w for w in dupset.warnings)
        assert (dst.root / "dupset.json").exists()

    def test_variant_bytes_differ_from_the_original(self, src_run):
        dst = RunPaths.for_run("dup")
        dupset = build_dupset(src_run, dst, seed=42)
        for pair in dupset.pairs:
            if pair.kind == "variant":
                assert dupset.sha256[pair.original] != dupset.sha256[pair.candidate]

    def test_build_is_deterministic(self, src_run, tmp_path):
        first = build_dupset(src_run, RunPaths.for_run("dup1"), seed=42)
        second = build_dupset(src_run, RunPaths.for_run("dup2"), seed=42)
        assert first.sha256 == second.sha256
        assert [p.pair_id for p in first.pairs] == [p.pair_id for p in second.pairs]

    def test_page_two_is_not_used(self, src_run):
        dupset = build_dupset(src_run, RunPaths.for_run("dup"), seed=42)
        assert all("-p2" not in sample_id for sample_id in dupset.sha256)


# --- detection ------------------------------------------------------------------------------


def _statement(dates_amounts):
    return StatementPage(
        rows=[StatementRow(date=d, debit=a) for d, a in dates_amounts],
    )


def _dupset(pairs, sha):
    return DupSet(src_run_id="src", dst_run_id="dup", n_originals=1, pairs=pairs, sha256=sha)


def test_row_set_and_jaccard():
    a = row_set(_statement([("2024-03-01", 10), ("2024-03-02", 20)]))
    b = row_set(_statement([("2024-03-01", 10), ("2024-03-03", 30)]))
    assert jaccard(a, b) == pytest.approx(1 / 3)
    assert jaccard(a, a) == 1.0
    assert jaccard(set(), set()) == 0.0  # no evidence either way


def test_exact_duplicate_is_flagged_whatever_the_extraction_says():
    pairs = [DupPair(pair_id="p1", original="A", candidate="B", kind="variant", variant="crop")]
    dupset = _dupset(pairs, {"A": "same", "B": "same"})
    rows, details = evaluate(dupset, {("m1", "A"): _statement([]), ("m1", "B"): _statement([])}, {})
    assert all(d.sha_match and d.detected for d in details)
    assert {r.recall for r in rows} == {1.0}


def test_threshold_sweep_has_one_row_per_model_and_threshold():
    pairs = [
        DupPair(pair_id="v1", original="A", candidate="B", kind="variant", variant="scan_low"),
        DupPair(
            pair_id="n1", original="A", candidate="C", kind="negative", negative_kind="same_bank"
        ),
    ]
    dupset = _dupset(pairs, {"A": "a", "B": "b", "C": "c"})
    shared = [("2024-03-01", 10), ("2024-03-02", 20), ("2024-03-03", 30), ("2024-03-04", 40)]
    pages = {
        ("m1", "A"): _statement(shared),
        ("m1", "B"): _statement(shared[:3] + [("2024-03-09", 90)]),  # jaccard 3/5 = 0.6
        ("m1", "C"): _statement([("2024-04-01", 1)]),
        ("m2", "A"): _statement(shared),
        ("m2", "B"): _statement(shared),
        ("m2", "C"): _statement(shared),  # a false positive for m2
    }
    rows, details = evaluate(dupset, pages, {"A": 0, "B": 1, "C": 40})
    assert len(rows) == 2 * len(THRESHOLDS)
    assert {(r.model, r.threshold) for r in rows} == {
        (m, t) for m in ("m1", "m2") for t in THRESHOLDS
    }
    by_key = {(r.model, r.threshold): r for r in rows}
    assert by_key[("m1", 0.6)].recall == 1.0
    assert by_key[("m1", 0.7)].recall == 0.0
    assert by_key[("m1", 0.6)].fpr == 0.0
    assert by_key[("m2", 0.9)].recall == 1.0 and by_key[("m2", 0.9)].fpr == 1.0
    assert all(r.n_variants == 1 and r.n_negatives == 1 for r in rows)
    assert {d.phash_distance for d in details} == {1, 40}


# --- CLI -------------------------------------------------------------------------------------


def test_cli_build_then_eval(src_run, tmp_path):
    result = runner.invoke(
        app, ["dupset", "build", "--run-id", "src", "--dup-run-id", "dup", "--seed", "7"]
    )
    assert result.exit_code == 0, result.output
    dst = RunPaths.for_run("dup")
    manifest = list(read_rows(dst.manifest, ManifestRow))

    write_rows_atomic(
        dst.predictions,
        [_prediction(sample_id=row.sample_id, model="m1") for row in manifest],
    )
    result = runner.invoke(app, ["dupset", "eval", "--dup-run-id", "dup"])
    assert result.exit_code == 0, result.output

    rows = list(read_rows(dst.duplicates, DuplicateEvalRow))
    assert len(rows) == len(THRESHOLDS)
    # every page was given the same extraction, so every pair looks like a duplicate
    assert all(r.recall == 1.0 and r.fpr == 1.0 for r in rows)
    assert dst.duplicate_pairs.exists()
    dupset = json.loads(dst.dupset.read_text(encoding="utf-8"))
    assert dupset["src_run_id"] == "src"


def test_cli_eval_without_predictions_exits_2(src_run):
    runner.invoke(app, ["dupset", "build", "--run-id", "src", "--dup-run-id", "dup"])
    result = runner.invoke(app, ["dupset", "eval", "--dup-run-id", "dup"])
    assert result.exit_code == 2
