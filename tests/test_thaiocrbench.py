"""TDD tests for ocr_bench.data.thaiocrbench, using fake in-memory rows (no network/HF hub)."""

import numpy as np
import pytest
from PIL import Image

from ocr_bench.config import TaskSpec, ThaiOCRBenchConfig
from ocr_bench.data.thaiocrbench import (
    TaskNameError,
    prepare_thaiocrbench,
    select,
    to_sample,
)
from ocr_bench.schemas import HtmlGT, JsonGT, Task, TextGT


def _img():
    return Image.new("RGB", (4, 4), color=(10, 20, 30))


def _row(task, id_, answer="hello", category="Government", question="Q?"):
    return {
        "Id": id_,
        "Task": task,
        "category": category,
        "question": question,
        "answer": answer,
        "image": _img(),
    }


def _cfg(tasks: dict[str, TaskSpec], domain_field: str | None = "category") -> ThaiOCRBenchConfig:
    return ThaiOCRBenchConfig(
        kind="thaiocrbench",
        hf_repo="typhoon-ai/ThaiOCRBench",
        seed=42,
        tasks=tasks,
        domain_field=domain_field,
    )


class TestSelectCaps:
    def test_cap_reduces_to_exactly_cap(self):
        rows = [_row("Full-page OCR", f"{i:08x}") for i in range(30)]
        cfg = _cfg({"Full-page OCR": TaskSpec(task=Task.ocr_fullpage, code="fullpage", cap=10)})
        selected, info = select(rows, cfg)
        assert len(selected) == 10
        assert info["thaiocrbench"]["Full-page OCR"] == 10

    def test_no_cap_keeps_all(self):
        rows = [_row("Table parsing", f"{i:08x}") for i in range(5)]
        cfg = _cfg({"Table parsing": TaskSpec(task=Task.table, code="table", cap=None)})
        selected, info = select(rows, cfg)
        assert len(selected) == 5
        assert info["thaiocrbench"]["Table parsing"] == 5

    def test_cap_greater_than_n_keeps_all(self):
        rows = [_row("Table parsing", f"{i:08x}") for i in range(5)]
        cfg = _cfg({"Table parsing": TaskSpec(task=Task.table, code="table", cap=100)})
        selected, info = select(rows, cfg)
        assert len(selected) == 5


class TestSelectStratification:
    def test_two_categories_split_20_10_cap_9(self):
        rows = [_row("Full-page OCR", f"a{i:07x}", category="Government") for i in range(20)]
        rows += [_row("Full-page OCR", f"b{i:07x}", category="Finance") for i in range(10)]
        cfg = _cfg({"Full-page OCR": TaskSpec(task=Task.ocr_fullpage, code="fullpage", cap=9)})
        selected, info = select(rows, cfg)
        assert len(selected) == 9
        cats = [rows[i]["category"] for i in selected]
        assert cats.count("Government") == 6
        assert cats.count("Finance") == 3


class TestSelectDeterminism:
    def test_two_calls_same_ids(self):
        rows = [
            _row("Full-page OCR", f"{i:08x}", category=("Government" if i % 2 else "Finance"))
            for i in range(30)
        ]
        cfg = _cfg({"Full-page OCR": TaskSpec(task=Task.ocr_fullpage, code="fullpage", cap=10)})
        selected1, _ = select(rows, cfg)
        selected2, _ = select(rows, cfg)
        ids1 = sorted(rows[i]["Id"] for i in selected1)
        ids2 = sorted(rows[i]["Id"] for i in selected2)
        assert ids1 == ids2


class TestSelectTaskNameError:
    def test_missing_task_raises_with_present_names(self):
        rows = [_row("Full-page OCR", "1")]
        cfg = _cfg(
            {
                "Full-page OCR": TaskSpec(task=Task.ocr_fullpage, code="fullpage", cap=None),
                "Ghost Task": TaskSpec(task=Task.classify, code="ghost", cap=None),
            }
        )
        with pytest.raises(TaskNameError) as exc_info:
            select(rows, cfg)
        msg = str(exc_info.value)
        assert "Ghost Task" in msg
        assert "Full-page OCR" in msg


class TestSelectDomain:
    def test_missing_category_gives_domain_none_and_warning(self):
        rows = [
            {
                "Id": "1",
                "Task": "Full-page OCR",
                "question": "Q?",
                "answer": "hello",
                "image": _img(),
            }
        ]
        cfg = _cfg({"Full-page OCR": TaskSpec(task=Task.ocr_fullpage, code="fullpage", cap=None)})
        selected, info = select(rows, cfg)
        assert info["domain_slice_available"] is False
        assert any("domain" in w.lower() for w in info["warnings"])

        samples, _ = prepare_thaiocrbench(cfg, rows=rows)
        assert samples[0].domain is None


class TestToSample:
    def test_kie_answer_with_fence_and_trailing_comma(self):
        answer = '```json\n{"ราคารวม": "1,000.00 บาท",}\n```'
        row = _row("Key information extraction", "abc123", answer=answer)
        spec = TaskSpec(task=Task.kie, code="kie", cap=None)
        sample = to_sample(row, spec, "Key information extraction", "category")
        assert isinstance(sample.gt, JsonGT)
        assert sample.gt.fields["ราคารวม"] == "1,000.00 บาท"
        assert "ราคารวม" in sample.critical_fields
        assert sample.gt.raw == answer

    def test_markdown_table_becomes_html_with_th(self):
        answer = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        row = _row("Table parsing", "t1", answer=answer)
        spec = TaskSpec(task=Task.table, code="table", cap=None)
        sample = to_sample(row, spec, "Table parsing", "category")
        assert isinstance(sample.gt, HtmlGT)
        assert "<th>a</th>" in sample.gt.html

    def test_bare_tr_wrapped_in_table(self):
        answer = "<tr><td>x</td></tr>"
        row = _row("Table parsing", "t2", answer=answer)
        spec = TaskSpec(task=Task.table, code="table", cap=None)
        sample = to_sample(row, spec, "Table parsing", "category")
        assert sample.gt.html.startswith("<table>")
        assert sample.gt.html.endswith("</table>")

    def test_classify_label_stripped(self):
        row = _row("Document classification", "c1", answer="  Invoice  \n")
        spec = TaskSpec(task=Task.classify, code="classify", cap=None)
        sample = to_sample(row, spec, "Document classification", "category")
        assert isinstance(sample.gt, JsonGT)
        assert sample.gt.label == "Invoice"

    def test_ocr_fullpage_is_text_gt(self):
        row = _row("Full-page OCR", "f1", answer="some text\nline2")
        spec = TaskSpec(task=Task.ocr_fullpage, code="fullpage", cap=None)
        sample = to_sample(row, spec, "Full-page OCR", "category")
        assert isinstance(sample.gt, TextGT)
        assert sample.gt.text == "some text\nline2"
        assert sample.critical_fields == []

    def test_sample_id_and_fields(self):
        row = _row("Full-page OCR", "deadbeef", question="Read the page")
        spec = TaskSpec(task=Task.ocr_fullpage, code="fullpage", cap=None)
        sample = to_sample(row, spec, "Full-page OCR", "category")
        assert sample.sample_id == "TOB-fullpage-deadbeef"
        assert sample.question == "Read the page"
        assert sample.page_no == 1
        assert sample.n_pages == 1
        assert sample.bank is None
        assert sample.doc_type is None
        assert sample.subtask == "Full-page OCR"
        assert isinstance(sample.image, np.ndarray)
        assert sample.image.shape == (4, 4, 3)

    def test_unparseable_kie_answer_raises_naming_sample(self):
        row = _row("Key information extraction", "bad1", answer="not json {{{")
        spec = TaskSpec(task=Task.kie, code="kie", cap=None)
        with pytest.raises(ValueError) as exc_info:
            to_sample(row, spec, "Key information extraction", "category")
        assert "bad1" in str(exc_info.value)


class TestPrepareThaiOCRBench:
    def test_end_to_end_with_fake_rows(self):
        rows = [_row("Full-page OCR", f"{i:08x}") for i in range(3)]
        rows += [_row("Table parsing", f"t{i:07x}", answer="<tr><td>1</td></tr>") for i in range(2)]
        cfg = _cfg(
            {
                "Full-page OCR": TaskSpec(task=Task.ocr_fullpage, code="fullpage", cap=None),
                "Table parsing": TaskSpec(task=Task.table, code="table", cap=None),
            }
        )
        samples, info = prepare_thaiocrbench(cfg, rows=rows)
        assert len(samples) == 5
        assert info["thaiocrbench"]["Full-page OCR"] == 3
        assert info["thaiocrbench"]["Table parsing"] == 2
