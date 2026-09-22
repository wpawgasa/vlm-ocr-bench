"""The official ThaiOCRBench per-sample score (`tob_score`), ported (task 4.4).

Attribution
-----------
Ported from OCRBench v2's evaluator, https://github.com/Yuliang-Liu/MultimodalOCR,
directory `OCRBench_v2/eval_scripts/`, files `eval.py` (per-task dispatch),
`page_ocr_metric.py` (`cal_per_metrics`), `vqa_metric.py` (`levenshtein_distance`,
`cn_vqa_evaluation`) and `TEDS_metric.py` (`TEDS`, `wrap_html_table`,
`convert_str_to_dict`, `generate_combinations`, `compute_f1_score`, `pre_clean`,
`get_tree`, `STEDS`, `doc_parsing_evaluation`).

    MIT License

    Copyright (c) 2023 Yuliang Liu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

The `TEDS`/`TableTree`/`CustomConfig` code in OCRBench v2's `TEDS_metric.py` is itself
PubTabNet's TEDS: Copyright 2020 IBM, author peter.zhong@au1.ibm.com, distributed under
the Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0). Changes here: the
`distance`/`editdistance`/`Levenshtein` packages are replaced by `rapidfuzz` (same
unit-cost Levenshtein), `node.getchildren()` by `list(node)`, parallel batch evaluation
and debugging code are dropped, and the code is restyled.

ThaiOCRBench (https://github.com/scb-10x/ThaiOCRBench) adapts that evaluator with a few
Thai-specific changes. Its scripts carry no license, so none of their text is used here;
the behaviour of each change was reimplemented independently and is marked
"ThaiOCRBench behaviour" below:

1. `cal_per_metrics` tokenizes with PyThaiNLP `newmm` when either side has Thai.
2. `cn_vqa_evaluation` accepts a list-literal answer, and a single answer string is
   always checked by containment and then ANLS (no comma-count rule).
3. `compute_f1_score` compares dict values as sorted JSON and treats only `None` (not
   every falsy value) as missing.
4. Task names ("Full-page OCR", ...) and the per-task dispatch in `tob_score`.
"""

import ast
import json
import re
import warnings
from collections import deque
from functools import cache, lru_cache
from importlib import resources
from itertools import product
from typing import Any

import nltk
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score
from pythainlp.tokenize import word_tokenize
from rapidfuzz.distance import Levenshtein
from zss import Node, simple_distance

PARITY_THRESHOLD = 0.01

BMFL_TASKS = ("Full-page OCR", "Fine-grained text recognition", "Handwritten content extraction")
VQA_TASKS = ("Text recognition", "Document classification")
KIE_TASKS = ("Key information extraction", "Key information mapping")
KEPT_TASKS = (*BMFL_TASKS, *VQA_TASKS, "Table parsing", "Document parsing", *KIE_TASKS)


def wordnet_available() -> bool:
    """True if the NLTK WordNet corpus (needed by METEOR) is installed."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        try:
            nltk.data.find("corpora/wordnet.zip")
        except LookupError:
            return False
    return True


# ---------------------------------------------------------------------------
# page_ocr_metric.py: cal_per_metrics
# ---------------------------------------------------------------------------

_THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fa5]")


def contain_chinese_string(text: str) -> bool:
    return bool(_CHINESE_RE.search(text))


def _jieba_cut(text: str) -> list[str]:
    import logging

    import jieba

    jieba.setLogLevel(logging.WARNING)
    return jieba.lcut(text)


def _tokenize_pair(pred: str, gt: str) -> tuple[list[str], list[str]]:
    # ThaiOCRBench behaviour: Thai on either side -> PyThaiNLP newmm (default arguments,
    # so whitespace tokens are kept); otherwise OCRBench v2's jieba / whitespace split.
    if _THAI_RE.search(gt) or _THAI_RE.search(pred):
        return word_tokenize(gt, engine="newmm"), word_tokenize(pred, engine="newmm")
    if contain_chinese_string(gt) or contain_chinese_string(pred):
        return _jieba_cut(gt), _jieba_cut(pred)
    return gt.split(), pred.split()


def cal_per_metrics(pred: str, gt: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    reference, hypothesis = _tokenize_pair(pred, gt)

    with warnings.catch_warnings():
        # NLTK warns when an n-gram order has no overlap; the value is still returned.
        warnings.simplefilter("ignore")
        metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference_set = set(reference)
    hypothesis_set = set(hypothesis)
    metrics["f_measure"] = f_measure(reference_set, hypothesis_set)
    metrics["precision"] = precision(reference_set, hypothesis_set)
    metrics["recall"] = recall(reference_set, hypothesis_set)
    # nltk.edit_distance with default arguments is unit-cost Levenshtein.
    metrics["edit_dist"] = Levenshtein.distance(pred, gt) / max(len(pred), len(gt))
    return metrics


def _zero_if_none(value: float | None) -> float:
    return 0.0 if value is None else value


def bmfl(pred: str, gt: str) -> float:
    """(BLEU + METEOR + F-measure + (1 - edit distance)) / 4."""
    m = cal_per_metrics(pred, gt)
    return (
        _zero_if_none(m["bleu"])
        + _zero_if_none(m["meteor"])
        + _zero_if_none(m["f_measure"])
        + (1 - _zero_if_none(m["edit_dist"]))
    ) / 4


# ---------------------------------------------------------------------------
# vqa_metric.py: levenshtein_distance, cn_vqa_evaluation
# ---------------------------------------------------------------------------


def levenshtein_distance(s1: str, s2: str) -> int:
    return Levenshtein.distance(s1, s2)


def _cn_norm(s: Any) -> str:
    if isinstance(s, (int, float)):
        s = str(s)
    return s.lower().strip().replace("\n", " ").replace(" ", "")


def _anls_value(predict: str, answer: str) -> float:
    dist = levenshtein_distance(predict, answer)
    length = max(len(predict), len(answer))
    value = 0.0 if length == 0 else float(dist) / float(length)
    return 1 - value


def cn_vqa_evaluation(predict: Any, answers: list[Any]) -> float:
    score: float = 0
    first = answers[0]
    if isinstance(first, str) and first.startswith("[") and first.endswith("]"):
        # ThaiOCRBench behaviour: a list-literal answer holds the accepted alternatives,
        # scored with OCRBench v2's multi-answer rules.
        for candidate in ast.literal_eval(first):
            answer = _cn_norm(candidate)
            pred = _cn_norm(predict)
            if len(answer.split(",")) < 4:
                if answer in pred:
                    score = 1
            else:
                value = _anls_value(pred, answer)
                if value >= 0.5 and value > score:
                    score = value
        return score

    # ThaiOCRBench behaviour: one answer string, containment first, then ANLS >= 0.5.
    answer = _cn_norm(first)
    pred = _cn_norm(predict)
    if answer in pred:
        return 1
    value = _anls_value(pred, answer)
    if value >= 0.5 and value > score:
        score = value
    return score


# ---------------------------------------------------------------------------
# TEDS_metric.py: TEDS (PubTabNet, Apache 2.0) and helpers
# ---------------------------------------------------------------------------


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == "td":
            result = (
                f'"tag": {self.tag}, "colspan": {self.colspan:d}, '
                f'"rowspan": {self.rowspan:d}, "text": {self.content}'
            )
        else:
            result = f'"tag": {self.tag}'
        for child in self.children:
            result += child.bracket()
        return f"{{{result}}}"


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1"""
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """Tree Edit Distance based Similarity"""

    def __init__(self, structure_only=False, ignore_nodes=None):
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        """Tokenizes table cells"""
        self.__tokens__.append(f"<{node.tag}>")
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in list(node):
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append(f"</{node.tag}>")
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Converts HTML tree to the format required by apted"""
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in list(node):
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        """Computes TEDS score between the prediction and the ground truth of a
        given sample
        """
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath("body/table") and true.xpath("body/table"):
            pred = pred.xpath("body/table")[0]
            true = true.xpath("body/table")[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0


def wrap_html_table(html_table: str) -> str:
    """
    The TEDS computation from PubTabNet code requires that the input html table should
    have <html>, <body>, and <table> tags. Add them if they are missing.
    """
    html_table = html_table.replace("\n", "")
    # add missing <table> tag if missing
    if "<table" in html_table and "</table>" not in html_table:
        html_table = html_table + "</table>"
    elif "<table" not in html_table and "</table>" in html_table:
        html_table = "<table>" + html_table
    elif "<table" not in html_table and "</table>" not in html_table:
        html_table = "<table>" + html_table + "</table>"
    # add <body> and <html> tags if missing
    if "<body>" not in html_table:
        html_table = "<body>" + html_table + "</body>"
    if "<html>" not in html_table:
        html_table = "<html>" + html_table + "</html>"
    return html_table


def convert_str_to_dict(predict_str: str) -> dict[str, str]:
    """
    Parses the 'predict' string and returns a dictionary.
    Missing or unparseable content is handled gracefully.
    """
    # Remove code fences like ```python\n...\n```
    code_fence_pattern = r"```(?:python|json)?\n(.*?)\n```"
    match = re.search(code_fence_pattern, predict_str, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
    else:
        content = predict_str.strip()

    data: Any = {}
    success = False

    # try parsing with JSON
    try:
        data = json.loads(content)
        success = True
    except json.JSONDecodeError:
        pass

    # try parsing with ast.literal_eval
    if not success:
        try:
            data = ast.literal_eval(content)
            if isinstance(data, dict):
                success = True
        except (ValueError, SyntaxError):
            pass

    # try parsing with regex
    if not success:
        key_value_pattern = r'["\']?([\w\s]+)["\']?\s*[:=]\s*["\']?([^\n,"\'{}]+)["\']?'
        matches = re.findall(key_value_pattern, content)
        try:
            for key, value in matches:
                data[key.strip()] = value.strip()
        except Exception:
            return {}

    if not data:
        return {}

    try:
        result = {k.strip(): str(v).strip() for k, v in data.items()}
    except Exception:
        return {}
    return result


def generate_combinations(input_dict: Any) -> Any:
    """
    Function to generate all possible combinations of values from a dictionary.
    """
    kie_answer = input_dict
    if not isinstance(kie_answer, dict):
        kie_answer = kie_answer.strip('"')
        try:
            kie_answer = json.loads(kie_answer)
        except json.JSONDecodeError:
            try:
                kie_answer = ast.literal_eval(kie_answer)
                if not isinstance(kie_answer, dict):
                    kie_answer = ast.literal_eval(kie_answer)
            except (ValueError, SyntaxError):
                return {}

        # Ensure the parsed result is a dictionary.
        if not isinstance(kie_answer, dict):
            raise ValueError("Input could not be parsed into a dictionary.")

        keys = list(kie_answer.keys())

        value_lists = []
        for single_key in keys:
            single_value = kie_answer[single_key]
            if not isinstance(single_value, list):
                single_value = [single_value]
            value_lists.append(single_value)

        # Compute the Cartesian product of the value lists.
        combinations = list(product(*value_lists))

        # Create a dictionary for each combination of values.
        return [dict(zip(keys, values, strict=True)) for values in combinations]

    keys = list(input_dict.keys())
    value_lists = [input_dict[key] for key in keys]

    # Compute the Cartesian product of the value lists.
    combinations = list(product(*value_lists))

    # Create a dictionary for each combination of values.
    return [dict(zip(keys, values, strict=True)) for values in combinations]


def _kie_value(value: Any, *, parse_str: bool) -> str | None:
    # ThaiOCRBench behaviour: dict values (or, on the prediction side, strings holding a
    # dict literal) are compared as key-sorted JSON; anything else by its str(); then
    # lower-cased, trimmed, newlines to spaces and all spaces removed. Only None counts
    # as missing (an empty string is a value).
    if value is None:
        return None
    if isinstance(value, dict):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    elif parse_str and isinstance(value, str):
        # The reference evaluates the string as Python; a literal parse is used here so
        # model output is never executed. Only a dict result is used either way.
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            value = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    return str(value).lower().strip().replace("\n", " ").replace(" ", "")


def compute_f1_score(preds: dict, gts: dict, ignores: tuple[str, ...] = ()) -> float:
    """Mean per-key F1 (1 on an exact normalized match, else 0) over the union of keys;
    keys missing on both sides are skipped; 0 when no key remains."""
    keys = set(preds.keys()).union(set(gts.keys())) - set(ignores)
    f1_scores: dict[Any, float] = {}

    for key in keys:
        pred_value = _kie_value(preds.get(key, None), parse_str=True)
        gt_value = _kie_value(gts.get(key, None), parse_str=False)

        if pred_value is None and gt_value is None:
            continue
        elif pred_value is None or gt_value is None:
            precision_, recall_ = 0.0, 0.0
        elif pred_value == gt_value:
            precision_, recall_ = 1.0, 1.0
        else:
            precision_, recall_ = 0.0, 0.0

        denominator = precision_ + recall_
        f1_scores[key] = 2 * precision_ * recall_ / denominator if denominator > 0 else 0.0

    if len(f1_scores) == 0:
        return 0
    return sum(f1_scores.values()) / len(f1_scores)


def pre_clean(text: str) -> str:
    text = re.sub(r"<bos>|<eos>|<pad>|<unk>", "", text)
    text = re.sub(r"\s##(\S)", r"\1", text)
    text = re.sub(r"\\\s", r"\\", text)
    text = re.sub(r"\s\*\s\*\s", r"**", text)
    text = re.sub(r"{\s", r"{", text)
    text = re.sub(r"\s}", r"}", text)
    text = re.sub(r"\s}", r"}", text)
    text = re.sub(r"\\begin\s", r"\\begin", text)
    text = re.sub(r"\\end\s", r"\\end", text)
    text = re.sub(r"\\end{table}", r"\\end{table} \n\n", text)
    text = text.replace("\n", " ")
    text = text.replace("*", " ")
    text = text.replace("_", " ")
    return text


def get_tree(input_str: str) -> Node:
    tree = Node("ROOT").addkid(Node("TITLE"))

    lines = input_str.split("\n")
    lines = [pre_clean(line) for line in lines]
    last_title = ""
    for line in lines:
        if line.startswith("#"):
            child = tree.get("ROOT")
            line = line.replace("#", "")
            child.addkid(Node(line))
            last_title = line
        else:
            if last_title == "":
                child = tree.get("TITLE")
                child.addkid(Node(line))
            else:
                child = tree.get(last_title)
                child.addkid(Node(line))
    return tree


def steds(pred_tree: Node, ref_tree: Node) -> float:
    def my_distance(pred: str, ref: str) -> int:
        if len(pred.split()) == 0 or len(ref.split()) == 0:
            return 1
        else:
            return 0

    total_distance = simple_distance(pred_tree, ref_tree, label_dist=my_distance)
    num_of_nodes = max(len(list(pred_tree.iter())), len(list(ref_tree.iter())))
    return 1 - total_distance / num_of_nodes


def doc_parsing_evaluation(pred: Any, gt: str) -> float:
    if not isinstance(pred, str):
        return 0
    return steds(get_tree(pred), get_tree(gt))


# ---------------------------------------------------------------------------
# eval.py: per-task dispatch (ThaiOCRBench behaviour for the task names)
# ---------------------------------------------------------------------------

_TEDS = TEDS(structure_only=False)


def _table_score(predict: str, gt_answer: str) -> float:
    predict_table = predict.replace("\n", "")
    if "<body" in predict_table:
        predict_table = re.findall("<body.*", predict_table)[0]
    elif "<table" in predict_table:
        predict_table = re.findall("<table.*", predict_table)[0]
    else:
        return 0.0
    try:
        return _TEDS.evaluate(wrap_html_table(predict_table), wrap_html_table(gt_answer))
    except Exception:
        return 0.0


def _kie_answer(gt_answer: str) -> Any:
    try:
        return ast.literal_eval(gt_answer)
    except (ValueError, SyntaxError):
        # [IMPLEMENTER DECIDES] the reference evaluator only literal-evals the answer (and
        # would crash on JSON `null`/`true`); every published answer parses that way, and
        # for our stored answers JSON is accepted as a fallback.
        from ocr_bench.data.gt_utils import parse_answer_json

        return parse_answer_json(gt_answer)


def _kie_score(predict: str, gt_answer: str) -> float:
    try:
        answers = _kie_answer(gt_answer)
    except ValueError:
        return 0.0
    if not isinstance(answers, dict):
        return 0.0
    answers = {k: v if isinstance(v, list) else [v] for k, v in answers.items()}
    combos = generate_combinations(answers)
    pred_kie_dict = convert_str_to_dict(predict)
    if len(combos) == 1:
        return float(compute_f1_score(pred_kie_dict, combos[0]))
    max_score = 0.0
    for answer in combos:
        max_score = max(max_score, compute_f1_score(pred_kie_dict, answer))
    return float(max_score)


def tob_score(task_name: str, answer_text: str | None, gt_answer: str, question: str) -> float:
    """The official ThaiOCRBench score of one sample.

    `task_name` is the ThaiOCRBench task name, `answer_text` the model's answer in its
    final documented form (None for a failed request, which scores 0), `gt_answer` the
    original benchmark answer string. `question` is accepted for parity with the
    reference signature; none of the kept tasks' scores depends on it.
    """
    del question
    if task_name not in KEPT_TASKS:
        raise ValueError(f"no official score for ThaiOCRBench task {task_name!r}")
    if answer_text is None:
        return 0.0
    if task_name in BMFL_TASKS:
        # [IMPLEMENTER DECIDES] the reference scores an empty Full-page / Fine-grained
        # answer 0 and would crash on a whitespace-only one (a None F-measure); all three
        # BMFL tasks score an empty answer 0 here and treat a None component as 0.
        if len(answer_text) == 0:
            return 0.0
        return float(bmfl(answer_text, gt_answer))
    if task_name in VQA_TASKS:
        return float(cn_vqa_evaluation(answer_text, [gt_answer]))
    if task_name == "Table parsing":
        return float(_table_score(answer_text, gt_answer))
    if task_name == "Document parsing":
        return float(doc_parsing_evaluation(answer_text, gt_answer))
    return _kie_score(answer_text, gt_answer)


# ---------------------------------------------------------------------------
# Parity gate
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_parity_fixture() -> tuple[dict, ...]:
    """The published-score fixture (package data; see tob_parity/README.md)."""
    ref = resources.files("ocr_bench.metrics").joinpath("tob_parity/fixture.json")
    return tuple(json.loads(ref.read_text(encoding="utf-8")))


@cache
def parity_diffs(task: str) -> tuple[tuple[str, float, float], ...]:
    """`(id, published_score, our_score)` for every fixture sample of `task` (cached)."""
    return tuple(
        (
            item["id"],
            float(item["published_score"]),
            tob_score(task, item["predict"], item["answers"][0], item["question"]),
        )
        for item in load_parity_fixture()
        if item["task"] == task
    )


def run_parity() -> dict[str, dict[str, Any]]:
    """Per kept task: fixture size, max |ours - published| and whether it is within
    `PARITY_THRESHOLD` (the gate the report uses to call `tob_score` comparable).

    A task whose scorer cannot run (e.g. METEOR without WordNet) is not comparable and
    carries a `reason`.
    """
    result: dict[str, dict[str, Any]] = {}
    for task in KEPT_TASKS:
        try:
            rows = parity_diffs(task)
        except LookupError as exc:
            reason = "NLTK WordNet missing" if "wordnet" in str(exc) else str(exc)
            result[task] = {"n": 0, "max_abs_diff": None, "comparable": False, "reason": reason}
            continue
        max_diff = max((abs(ours - pub) for _, pub, ours in rows), default=float("inf"))
        result[task] = {
            "n": len(rows),
            "max_abs_diff": round(max_diff, 6) if rows else None,
            "comparable": bool(rows) and max_diff <= PARITY_THRESHOLD,
        }
    return result
