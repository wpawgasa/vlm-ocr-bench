"""Tests for ocr_bench.confidence.calibrate (tasks 6.2, 6.3, 6.4). Synthetic data only."""

import json
import math

import numpy as np
import pytest
from typer.testing import CliRunner

from ocr_bench.cli import app
from ocr_bench.confidence.calibrate import (
    FieldRecord,
    Weights,
    band_rows,
    collect_field_records,
    compute_ece,
    fit_variant,
    grouped_folds,
    reliability_bins,
    run_calibrate,
    search_threshold,
)
from ocr_bench.confidence.proxy import FieldFeatures
from ocr_bench.jsonl import write_rows_atomic
from ocr_bench.metrics.arithmetic import ArithmeticResult
from ocr_bench.metrics.statement_run import StatementRecord
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import (
    Block,
    BlockType,
    Condition,
    FieldResult,
    ManifestRow,
    NormalizedPage,
    PredictionRow,
    RawPrediction,
    StatementFile,
    Task,
)

runner = CliRunner()


def _rec(
    sample_id, model, label, mean_lp=-0.1, min_lp=-0.3, agree=None, arith=None, critical=False
):
    return FieldRecord(
        sample_id=sample_id,
        condition="clean",
        model=model,
        field="total_amount",
        is_critical=critical,
        label=label,
        features=FieldFeatures(
            mean_logprob=mean_lp,
            min_logprob=min_lp,
            span_found=True,
            logprobs_missing=False,
            agree=agree,
            arith=arith,
        ),
    )


def _synthetic_records(n_samples=20, per_sample=4, seed=0, model="teleocr"):
    """A dataset where higher mean_logprob correlates with a correct label, i.e. a
    proxy a logistic fit can actually learn something from."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_samples):
        for _j in range(per_sample):
            mean_lp = rng.uniform(-3.0, -0.01)
            p_correct = 1 / (1 + math.exp(-(mean_lp + 1.5)))
            label = int(rng.random() < p_correct)
            records.append(
                _rec(f"s{i}", model, label, mean_lp=mean_lp, min_lp=mean_lp - rng.uniform(0, 0.5))
            )
    return records


# --- 6.2: grouped folds -----------------------------------------------------------------------


def test_grouped_folds_never_split_a_sample_id():
    sample_ids = [f"s{i // 3}" for i in range(60)]  # 20 distinct ids, 3 fields each
    for train_idx, test_idx in grouped_folds(sample_ids, n_splits=5):
        train_sids = {sample_ids[i] for i in train_idx}
        test_sids = {sample_ids[i] for i in test_idx}
        assert not (train_sids & test_sids)


def test_fit_variant_two_variants_both_produce_oof_confidences():
    records = _synthetic_records()
    for variant in ("full", "logprob_only"):
        fit = fit_variant(records, variant, seed=0)
        assert fit.status == "ok"
        assert len(fit.oof_conf) == len(records)
        assert all(c is not None and 0.0 <= c <= 1.0 for c in fit.oof_conf)
        assert fit.weights is not None


def test_fit_variant_insufficient_fields():
    records = _synthetic_records(n_samples=6, per_sample=2)  # 12 fields < 50
    fit = fit_variant(records, "full", seed=0)
    assert fit.status == "insufficient_data"
    assert fit.n == 12
    assert fit.weights is None
    assert fit.oof_conf == [None] * 12


def test_fit_variant_insufficient_distinct_samples():
    records = [_rec(f"s{i % 3}", "teleocr", i % 2) for i in range(60)]  # only 3 sample ids
    fit = fit_variant(records, "full", seed=0)
    assert fit.status == "insufficient_data"
    assert fit.n == 60


def test_fit_variant_single_class():
    records = [_rec(f"s{i}", "teleocr", 1) for i in range(60)]  # all label=1
    fit = fit_variant(records, "full", seed=0)
    assert fit.status == "insufficient_data"
    assert fit.classes == 1


# --- 6.3: bands and ECE -----------------------------------------------------------------------


def test_ece_near_zero_for_perfectly_calibrated_confidences():
    rng = np.random.default_rng(7)
    n = 20_000
    confs = rng.uniform(0, 1, size=n)
    labels = (rng.random(n) < confs).astype(int).tolist()
    ece = compute_ece(confs.tolist(), labels, n_bins=10)
    assert ece < 0.02


def test_band_counts_sum_to_n():
    rng = np.random.default_rng(3)
    n = 500
    confs = rng.uniform(0, 1, size=n).tolist()
    labels = (rng.random(n) < np.array(confs)).astype(int).tolist()
    sample_ids = [f"s{i}" for i in range(n)]
    rows = band_rows("teleocr", "full", confs, labels, sample_ids, seed=0, table="all", ece=0.01)
    assert len(rows) == 5
    assert sum(r.n for r in rows) == n
    assert {r.band for r in rows} == {1, 2, 3, 4, 5}


def test_reliability_bins_have_ten_entries_and_conserve_n():
    confs = [0.05, 0.15, 0.55, 0.55, 0.95, 0.99]
    labels = [0, 1, 1, 0, 1, 1]
    bins = reliability_bins(confs, labels, n_bins=10)
    assert len(bins) == 10
    assert sum(b["n"] for b in bins) == len(confs)


# --- 6.4: threshold search --------------------------------------------------------------------


def test_target_not_reachable():
    # The single highest-confidence field is itself wrong, so no accepted set (which
    # must always include it) can ever reach 100% accuracy: target 1.0 is unreachable
    # by construction, regardless of the rest of the (noisy) distribution.
    rng = np.random.default_rng(11)
    confs = rng.uniform(0, 0.9, size=199).tolist() + [0.999]
    labels = rng.integers(0, 2, size=199).tolist() + [0]
    result = search_threshold(confs, labels, target=1.0)
    assert result["reachable"] is False
    assert result["t_accept"] is None
    assert result["shares"]["accept"] == 0.0


def test_target_reachable_with_clean_separation():
    # low confidences are always wrong, high confidences always right: cleanly reachable.
    confs = [0.1] * 50 + [0.99] * 50
    labels = [0] * 50 + [1] * 50
    result = search_threshold(confs, labels, target=0.99)
    assert result["reachable"] is True
    assert result["t_accept"] == pytest.approx(0.99)
    assert result["shares"]["accept"] == pytest.approx(0.5)
    assert result["accuracy"]["accept"] == pytest.approx(1.0)


def test_reject_bucket_has_low_accuracy():
    confs = [0.1] * 50 + [0.99] * 50
    labels = [0] * 50 + [1] * 50
    result = search_threshold(confs, labels, target=0.99)
    assert result["t_reject"] is not None
    assert result["accuracy"]["reject"] <= 0.5


def test_weights_predict_matches_manual_sigmoid():
    weights = Weights(
        feature_names=("mean_logprob", "min_logprob"),
        mean=[0.0, 0.0],
        scale=[1.0, 1.0],
        coef=[2.0, -1.0],
        intercept=0.5,
    )
    x = [0.3, -0.2]
    expected = 1.0 / (1.0 + math.exp(-(0.5 + 2.0 * 0.3 + -1.0 * -0.2)))
    assert weights.predict(x) == pytest.approx(expected)


def test_weights_json_roundtrip():
    weights = Weights(
        feature_names=("mean_logprob", "min_logprob"),
        mean=[0.1, -0.2],
        scale=[1.1, 0.9],
        coef=[2.0, -1.0],
        intercept=0.5,
    )
    restored = Weights.from_json(json.loads(json.dumps(weights.to_json())))
    x = [0.4, -0.6]
    assert restored.predict(x) == pytest.approx(weights.predict(x))


def test_stored_thresholds_reproduce_bucket_shares_from_the_same_oof_confidences():
    # Spec scenario ("Reproducible operating point"): applying thresholds.json's
    # t_accept/t_reject to the *same* out-of-fold confidences reproduces the exact
    # reported shares. [IMPLEMENTER DECIDES, deviation from the handover's paraphrase:
    # the handover suggested recomputing confidences from raw features via the stored
    # (all-data refit) weights; spec.md's literal scenario instead re-applies the stored
    # thresholds to the persisted out-of-fold confidences, which is what is actually
    # guaranteed identical, since D10 mandates thresholds are computed on OOF
    # predictions only and a refit-on-all-data model is not numerically identical to
    # the k fold-specific OOF models.]
    confs = [0.1] * 50 + [0.99] * 50
    labels = [0] * 50 + [1] * 50
    result = search_threshold(confs, labels, target=0.99)

    def bucketize(confs, t_accept, t_reject):
        n = len(confs)
        accept = sum(1 for c in confs if t_accept is not None and c >= t_accept) / n
        reject = sum(1 for c in confs if t_reject is not None and c < t_reject) / n
        return {"accept": accept, "reject": reject, "review": 1 - accept - reject}

    reapplied = bucketize(confs, result["t_accept"], result["t_reject"])
    assert reapplied == pytest.approx(result["shares"])


# --- 6.4: CLI end-to-end on a tiny synthetic run ----------------------------------------------


def _build_kie_run(tmp_path, monkeypatch, n_samples=12, fields_per_sample=5):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    rp = RunPaths.for_run("cal-run")
    rng = np.random.default_rng(42)

    manifest_rows = []
    pred_rows = []
    field_rows = []
    models = ["teleocr", "dots.ocr"]
    from ocr_bench.schemas import FieldResult, FieldValue

    for i in range(n_samples):
        sample_id = f"sample-{i:03d}"
        manifest_rows.append(
            ManifestRow(
                sample_id=sample_id,
                source="thaiocrbench",
                task=Task.kie,
                condition=Condition.clean,
                image_path=f"img/{sample_id}.png",
                gt_kind="json",
                critical_fields=["field_0"],
            )
        )
        for model in models:
            tokens = []
            logprobs = []
            fields = {}
            for f in range(fields_per_sample):
                key = f"field_{f}"
                value = f"{100 + f}.00"
                mean_lp = float(rng.uniform(-3.0, -0.01))
                tokens.extend([f'"{key}": ', f'"{value}"', ", "])
                logprobs.extend([mean_lp, mean_lp, mean_lp])
                fields[key] = FieldValue(value=value, raw=value)
            pred_rows.append(
                PredictionRow(
                    sample_id=sample_id,
                    condition=Condition.clean,
                    model=model,
                    prompt_version="question-v1",
                    raw=RawPrediction(text="".join(tokens), tokens=tokens, token_logprobs=logprobs),
                    normalized=NormalizedPage(text="".join(tokens), fields=fields),
                    latency_ms=1.0,
                    prompt_tokens=1,
                    completion_tokens=len(tokens),
                )
            )
            for f in range(fields_per_sample):
                key = f"field_{f}"
                p_correct = 1 / (1 + math.exp(-(logprobs[f * 3] + 1.5)))
                exact = int(rng.random() < p_correct)
                field_rows.append(
                    FieldResult(
                        sample_id=sample_id,
                        condition=Condition.clean,
                        model=model,
                        task=Task.kie,
                        field=key,
                        is_critical=(f == 0),
                        pred=f"{100 + f}.00" if exact else "0.00",
                        gt=f"{100 + f}.00",
                        field_exact=exact,
                        field_fuzzy=exact,
                    )
                )

    write_rows_atomic(rp.manifest, manifest_rows)
    write_rows_atomic(rp.predictions, pred_rows)
    write_rows_atomic(rp.fields, field_rows)
    return rp, len(field_rows)


def test_cli_calibrate_writes_all_outputs(tmp_path, monkeypatch):
    rp, n_fields = _build_kie_run(tmp_path, monkeypatch, n_samples=12, fields_per_sample=5)
    assert n_fields == 120  # >= 60 fields, 12 samples

    result = runner.invoke(app, ["calibrate", "--run-id", "cal-run", "--target-acc", "0.99,0.995"])
    assert result.exit_code == 0, result.output

    assert rp.calibration.exists()
    assert rp.calibration_fields.exists()
    assert rp.reliability.exists()
    assert rp.thresholds.exists()

    thresholds = json.loads(rp.thresholds.read_text())
    assert set(thresholds) == {"teleocr", "dots.ocr"}
    for model_out in thresholds.values():
        assert set(model_out) == {"full", "logprob_only"}


def test_run_calibrate_function_directly(tmp_path, monkeypatch):
    rp, _ = _build_kie_run(tmp_path, monkeypatch, n_samples=12, fields_per_sample=5)
    summary = run_calibrate(rp, target_accs=[0.99], seed=0)
    assert summary["n_records"] == 120
    assert rp.calibration.exists()


# --- 6.1/6.2: collect_field_records over the statement pathway (Check A/B) --------------------

_STATEMENT_TABLE = [
    ["Date", "Description", "Withdrawal", "Deposit", "Balance"],
    ["01/03/24", "Brought Forward", "", "", "100.00"],
    ["04/03/24", "Fee", "20.00", "", "80.00"],
    ["07/03/24", "Salary", "", "50.00", "130.00"],
]
_HEADER_TEXT = (
    "Account Number 111-2-33333-4\nPeriod 01/03/2024 - 31/03/2024\nClosing Balance 130.00"
)


def _statement_pred_row(model: str) -> PredictionRow:
    tokens = [_HEADER_TEXT]
    logprobs = [-0.1]
    return PredictionRow(
        sample_id="file1-p1",
        condition=Condition.clean,
        model=model,
        prompt_version="statement-v1",
        raw=RawPrediction(text=_HEADER_TEXT, tokens=tokens, token_logprobs=logprobs),
        normalized=NormalizedPage(
            text=_HEADER_TEXT,
            blocks=[Block(type=BlockType.text, text=_HEADER_TEXT)],
            tables=[_STATEMENT_TABLE],
        ),
        latency_ms=1.0,
        prompt_tokens=1,
        completion_tokens=1,
    )


def test_collect_field_records_statement_pathway(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    rp = RunPaths.for_run("stmt-run")

    manifest_rows = [
        ManifestRow(
            sample_id="file1-p1",
            source="bankstmt",
            task=Task.statement,
            condition=Condition.clean,
            bank="kbank",
            image_path="img/file1-p1.png",
            gt_kind="text_layer",
        )
    ]
    write_rows_atomic(rp.manifest, manifest_rows)

    pred_rows = [_statement_pred_row("teleocr"), _statement_pred_row("dots.ocr")]
    write_rows_atomic(rp.predictions, pred_rows)

    field_rows = [
        FieldResult(
            sample_id="file1-p1",
            condition=Condition.clean,
            model=model,
            task=Task.statement,
            field="closing_balance",
            is_critical=True,
            pred="130.00",
            gt="130.00",
            field_exact=1,
            field_fuzzy=1,
        )
        for model in ("teleocr", "dots.ocr")
    ]
    write_rows_atomic(rp.fields, field_rows)

    statement_records = [
        StatementRecord(
            file_id="file1",
            condition=Condition.clean,
            model=model,
            bank="kbank",
            sample_ids=["file1-p1"],
            statement=StatementFile(file_id="file1", bank="kbank", pages=[]),
            arithmetic=ArithmeticResult(file_id="file1", model=model, closing_mismatch=False),
        )
        for model in ("teleocr", "dots.ocr")
    ]
    write_rows_atomic(rp.statements, statement_records)

    records = collect_field_records(rp)
    assert len(records) == 2
    by_model = {r.model: r for r in records}
    for rec in by_model.values():
        assert rec.field == "closing_balance"
        assert rec.label == 1
        # The raw "130.00" is located inside the header block text, so logprob stats
        # come from the located span, not the whole-completion fallback.
        assert rec.features.span_found is True
        # Check B found the closing balance consistent for both models.
        assert rec.features.arith == 1
        # Each model agrees with the other's normalized closing_balance.
        assert rec.features.agree == 1
