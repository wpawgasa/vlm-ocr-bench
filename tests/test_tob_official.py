"""Unit tests for the ported official ThaiOCRBench scorer (task 4.4)."""

import pytest

from ocr_bench.metrics.tob_official import (
    KEPT_TASKS,
    TEDS,
    cn_vqa_evaluation,
    compute_f1_score,
    convert_str_to_dict,
    doc_parsing_evaluation,
    generate_combinations,
    tob_score,
    wordnet_available,
    wrap_html_table,
)

needs_wordnet = pytest.mark.skipif(
    not wordnet_available(),
    reason="NLTK WordNet missing: run `uv run python -m nltk.downloader wordnet omw-1.4`",
)

TABLE = (
    "<table><tr><td>ก</td><td colspan='2'>ข</td></tr>"
    "<tr><td>1</td><td>2</td><td>3</td></tr></table>"
)


def test_kept_tasks_are_the_nine_thaiocrbench_names():
    assert len(KEPT_TASKS) == 9
    assert "Full-page OCR" in KEPT_TASKS and "Key information mapping" in KEPT_TASKS


def test_wrap_html_table_adds_missing_tags():
    assert wrap_html_table("<tr><td>a</td></tr>") == (
        "<html><body><table><tr><td>a</td></tr></table></body></html>"
    )
    assert wrap_html_table("<table><tr><td>a</td></tr>") == (
        "<html><body><table><tr><td>a</td></tr></table></body></html>"
    )


def test_teds_identical_is_one_and_differs_below_one():
    teds = TEDS()
    assert teds.evaluate(wrap_html_table(TABLE), wrap_html_table(TABLE)) == 1.0
    other = TABLE.replace("<td>3</td>", "<td>4</td>")
    assert 0 < teds.evaluate(wrap_html_table(other), wrap_html_table(TABLE)) < 1


def test_cn_vqa_containment_and_anls_threshold():
    assert cn_vqa_evaluation("คำตอบคือ ใบเสร็จ", ["ใบเสร็จ"]) == 1
    # far from the answer -> below 0.5 similarity -> 0
    assert cn_vqa_evaluation("xyz", ["ใบเสร็จรับเงิน"]) == 0
    # close but not contained -> ANLS value
    assert cn_vqa_evaluation("abcdefghiX", ["abcdefghij"]) == pytest.approx(0.9)


def test_cn_vqa_list_literal_answers():
    assert cn_vqa_evaluation("B", ["['A', 'B']"]) == 1


def test_convert_str_to_dict_fenced_json():
    s = '```json\n{"a": " 1 ", "b": 2}\n```'
    assert convert_str_to_dict(s) == {"a": "1", "b": "2"}
    assert convert_str_to_dict("not a dict at all") == {}


def test_generate_combinations_cartesian():
    combos = generate_combinations({"a": [1, 2], "b": ["x"]})
    assert combos == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_compute_f1_ignores_case_and_spaces():
    assert compute_f1_score({"k": "A B"}, {"k": "ab"}) == 1.0
    assert compute_f1_score({"k": "a", "x": "1"}, {"k": "a"}) == 0.5
    assert compute_f1_score({}, {}) == 0


def test_doc_parsing_identical_is_one():
    md = "# หัวข้อ\nบรรทัดหนึ่ง\n## ย่อย\nบรรทัดสอง"
    assert doc_parsing_evaluation(md, md) == 1.0
    assert doc_parsing_evaluation(None, md) == 0


def test_tob_score_none_answer_is_zero():
    for task in KEPT_TASKS:
        assert tob_score(task, None, "x", "") == 0.0


def test_tob_score_table_without_table_tag_is_zero():
    assert tob_score("Table parsing", "no table here", TABLE, "") == 0.0
    assert tob_score("Table parsing", "noise\n" + TABLE, TABLE, "") == 1.0


def test_tob_score_kie_best_combination():
    gt = "{'ชื่อ': ['สมชาย', 'สมชาย ใจดี'], 'ยอด': '100'}"
    assert (
        tob_score("Key information extraction", '{"ชื่อ": "สมชาย ใจดี", "ยอด": "100"}', gt, "") == 1.0
    )


def test_tob_score_kie_json_null_answer_accepted():
    # our stored raw answers are JSON; `null` is not a Python literal
    gt = '{"a": "1", "b": null}'
    assert tob_score("Key information mapping", '{"a": "1"}', gt, "") == 1.0


def test_tob_score_unknown_task_raises():
    with pytest.raises(ValueError):
        tob_score("Chart parsing", "x", "y", "")


@needs_wordnet
def test_bmfl_identical_text_is_one():
    text = "สวัสดีครับ วันนี้อากาศดี มาก และ ท้องฟ้า สดใส"
    assert tob_score("Full-page OCR", text, text, "") == pytest.approx(1.0, abs=1e-3)


@needs_wordnet
def test_bmfl_empty_prediction_is_zero():
    assert tob_score("Fine-grained text recognition", "", "ข้อความ", "") == 0.0
    assert tob_score("Handwritten content extraction", "", "ข้อความ", "") == 0.0
