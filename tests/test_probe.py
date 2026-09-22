"""Tests for `ocrbench probe` (real adapters over a fake chat client; no network)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ocr_bench.cli import app
from ocr_bench.jsonl import read_rows
from ocr_bench.models import registry
from ocr_bench.models.probe import CAPABILITIES, PROBE_JSON_QUESTION, select_probe_pages
from ocr_bench.models.vllm_client import CallResult
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import Condition, GtKind, ManifestRow, PredictionRow, Source, Task
from tests.fakes import FakeChatClient, make_run

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(REPO_ROOT / "configs" / "run.yaml")
runner = CliRunner()


def stmt(bank: str, idx: int, page: int, condition=Condition.clean) -> ManifestRow:
    sample_id = f"BS-{bank}-{idx:04d}-p{page}"
    suffix = "" if condition == Condition.clean else f"_{condition.value}"
    return ManifestRow(
        sample_id=sample_id,
        source=Source.bankstmt,
        task=Task.statement,
        bank=bank,
        doc_type="digital",
        page_no=page,
        n_pages=2,
        condition=condition,
        image_path=f"img/{sample_id}{suffix}.png",
        gt_kind=GtKind.text_layer,
    )


def tob(i: int) -> ManifestRow:
    return ManifestRow(
        sample_id=f"TOB-kie-{i}",
        source=Source.thaiocrbench,
        task=Task.kie,
        condition=Condition.clean,
        image_path=f"img/TOB-kie-{i}.png",
        gt_kind=GtKind.json,
        question="q",
    )


def manifest() -> list[ManifestRow]:
    return [
        tob(1),
        stmt("scb", 3, 1),
        stmt("bbl", 1, 2),
        stmt("bbl", 1, 1),
        stmt("bbl", 1, 1, Condition.photo),
        stmt("kbank", 1, 1, Condition.scan_low),
        stmt("kbank", 1, 1),
        stmt("bbl", 2, 1),
        stmt("kbank", 1, 2),
    ]


def test_select_probe_pages_spreads_banks_then_tops_up():
    picks = select_probe_pages(manifest())
    assert [p.sample_id for p in picks] == [
        "BS-bbl-0001-p1",
        "BS-kbank-0001-p1",
        "BS-scb-0003-p1",
        "BS-bbl-0001-p2",
        "BS-bbl-0002-p1",
    ]
    assert all(p.condition == Condition.clean for p in picks)


def test_select_probe_pages_caps_at_five_banks():
    rows = [stmt(b, 1, 1) for b in ["ttb", "bbl", "gsb", "kbank", "ktb", "scb"]]
    assert [p.bank for p in select_probe_pages(rows)] == ["bbl", "gsb", "kbank", "ktb", "scb"]


DOTS_LAYOUT = json.dumps(
    [
        {"bbox": [10, 10, 100, 30], "category": "Title", "text": "รายการเดินบัญชี"},
        {
            "bbox": [10, 40, 190, 200],
            "category": "Table",
            "text": "<table><tr><td>01/03</td><td>100.00</td></tr></table>",
        },
    ],
    ensure_ascii=False,
)
GOOD_JSON = '{"account_no": "123", "account_name": "นายก", "closing_balance": "1.00"}'


def dots_responder(image, prompt, sampling):
    if prompt == PROBE_JSON_QUESTION:
        return GOOD_JSON
    return DOTS_LAYOUT


def tele_responder(image, prompt, sampling):
    if prompt == "\nAnalyze the image layout.":
        return "<box:0 0 1000 100><label:text><rotate_up>"
    if prompt == PROBE_JSON_QUESTION:
        return "ขออภัย"
    return "ข้อความ"


def typhoon_responder(image, prompt, sampling):
    if prompt == PROBE_JSON_QUESTION:
        return CallResult(text='{"account_no": "1"}')
    return CallResult(text="ข้อความ\n<page_number>1</page_number>")  # no logprobs


RESPONDERS = {
    "teleocr": tele_responder,
    "dotsocr": dots_responder,
    "typhoon_ocr15": typhoon_responder,
}


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    make_run(tmp_path, "r1", manifest())
    real_build = registry.build_model
    monkeypatch.setattr(
        registry,
        "build_model",
        lambda cfg, **kw: real_build(cfg, client=FakeChatClient(RESPONDERS[cfg.name])),
    )
    return RunPaths.for_run("r1")


def _probe(*extra):
    return runner.invoke(app, ["probe", "--config", CONFIG, "--run-id", "r1", *extra])


def test_probe_writes_matrix_evidence_and_predictions(run_dir):
    result = _probe("--models", "teleocr,dotsocr,typhoon_ocr15")
    assert result.exit_code == 0, result.output
    probe = json.loads(run_dir.probe.read_text(encoding="utf-8"))
    matrix = probe["matrix"]
    assert set(matrix) == set(CAPABILITIES)
    for capability in CAPABILITIES:
        for model in ("teleocr", "dotsocr", "typhoon_ocr15"):
            assert matrix[capability][model] in {"yes", "partial", "no"}

    assert matrix["reading_order_text"] == {
        "teleocr": "yes",
        "dotsocr": "yes",
        "typhoon_ocr15": "yes",
    }
    assert matrix["table_rows_cells"] == {"teleocr": "no", "dotsocr": "yes", "typhoon_ocr15": "no"}
    assert matrix["block_bboxes"] == {"teleocr": "yes", "dotsocr": "yes", "typhoon_ocr15": "no"}
    assert matrix["page_index"] == {"teleocr": "no", "dotsocr": "no", "typhoon_ocr15": "partial"}
    assert matrix["json_fields_on_request"] == {
        "teleocr": "no",
        "dotsocr": "yes",
        "typhoon_ocr15": "no",
    }
    assert matrix["token_logprobs"] == {"teleocr": "yes", "dotsocr": "yes", "typhoon_ocr15": "no"}

    expected_ids = [p.sample_id for p in select_probe_pages(manifest())]
    assert probe["evidence"] == {m: expected_ids for m in ("teleocr", "dotsocr", "typhoon_ocr15")}
    assert probe["prompt_versions"]["dotsocr"] == {
        "statement": "dots-layout-all-v1",
        "json_fields_on_request": "probe-json-v1",
    }
    assert probe["prompt_versions"]["teleocr"]["statement"] == "teleocr-2stage-v1"

    rows = list(read_rows(run_dir.probe_predictions, PredictionRow))
    assert len(rows) == 3 * 2 * 5
    assert sum(1 for r in rows if r.prompt_version == "probe-json-v1") == 15
    assert not run_dir.predictions.exists()  # probe never touches the accuracy predictions


def test_probe_json_partial(run_dir, monkeypatch):
    calls = {"n": 0}

    def flaky_json(image, prompt, sampling):
        if prompt == PROBE_JSON_QUESTION:
            calls["n"] += 1
            return GOOD_JSON if calls["n"] % 2 else "no"
        return DOTS_LAYOUT

    monkeypatch.setattr(
        registry,
        "build_model",
        lambda cfg, **kw: registry.ADAPTERS[cfg.name](cfg, FakeChatClient(flaky_json)),
    )
    result = _probe("--models", "dotsocr")
    assert result.exit_code == 0, result.output
    probe = json.loads(run_dir.probe.read_text(encoding="utf-8"))
    assert probe["matrix"]["json_fields_on_request"]["dotsocr"] == "partial"


def test_probe_health_failure_exits_2(run_dir, monkeypatch):
    def build(cfg, **kw):
        client = FakeChatClient(RESPONDERS[cfg.name], healthy=False)
        return registry.ADAPTERS[cfg.name](cfg, client)

    monkeypatch.setattr(registry, "build_model", build)
    result = _probe("--models", "dotsocr")
    assert result.exit_code == 2
    assert not run_dir.probe.exists()


def test_probe_without_statement_pages_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(tmp_path))
    make_run(tmp_path, "r1", [tob(1)])
    real_build = registry.build_model
    monkeypatch.setattr(
        registry,
        "build_model",
        lambda cfg, **kw: real_build(cfg, client=FakeChatClient(RESPONDERS[cfg.name])),
    )
    result = _probe("--models", "dotsocr")
    assert result.exit_code == 2
    assert "statement" in result.output
