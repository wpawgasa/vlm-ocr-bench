"""Scorer dispatch on `(task, gt_kind)` (task 4.7).

| task and ground truth                          | metrics                                    |
| ---------------------------------------------- | ------------------------------------------ |
| ocr_fullpage, handwriting, ocr_line + text     | cer, wer, bmfl, tob_score                  |
| table + html                                   | ted, cer, tob_score                        |
| docparse + text                                | ted, cer, tob_score                        |
| kie, kie_map + json                            | kie_precision, kie_recall, kie_f1,         |
|                                                | tob_score, plus one FieldResult per field  |
| classify + json                                | anls, tob_score                            |
| statement + text_layer                         | cer, row_precision/recall/f1, header fields|
| statement + json (manual labels)               | row_precision/recall/f1, header fields     |
| anything + none                                | no_gt (value None)                         |
| any other pair                                 | not_applicable (value None)                |

`tob_score` is only written for ThaiOCRBench samples whose subtask is a kept task.

A full miss (a failed request, or a missing prediction) gets worst-case values: cer/wer 1,
every other metric 0, every field `field_exact=0`, `field_fuzzy=0` and a false reject
where the ground truth has a value. A partial TeleOCR page ("k of n blocks failed" with
some output) is scored as it is.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ocr_bench.config import BankOverrides
from ocr_bench.metrics import tob_official
from ocr_bench.metrics.anls import anls
from ocr_bench.metrics.cer_wer import cer, wer
from ocr_bench.metrics.kie_f1 import FieldScore, micro_prf, score_fields, worst_case_fields
from ocr_bench.metrics.statement_gt import parse_text_layer, score_statement, worst_case_statement
from ocr_bench.metrics.ted import first_table, ted, ted_docparse
from ocr_bench.normalize.statement import map_statement_page
from ocr_bench.schemas import (
    FieldResult,
    GroundTruth,
    HtmlGT,
    JsonGT,
    ManifestRow,
    NoGT,
    PredictionRow,
    ScoreRow,
    Source,
    StatementPage,
    Task,
    TextGT,
    TextLayerGT,
)

_PARTIAL_RE = re.compile(r"^\d+ of \d+ blocks failed$")
_TEXT_TASKS = {Task.ocr_fullpage, Task.handwriting, Task.ocr_line}
_KIE_TASKS = {Task.kie, Task.kie_map}
_WORST = {"cer": 1.0, "wer": 1.0}  # every other metric's worst value is 0


@dataclass
class SampleScores:
    scores: list[ScoreRow] = field(default_factory=list)
    fields: list[FieldResult] = field(default_factory=list)


def is_partial(pred: PredictionRow) -> bool:
    return pred.raw.error is not None and bool(_PARTIAL_RE.match(pred.raw.error))


def _has_output(pred: PredictionRow) -> bool:
    return bool(pred.raw.text.strip() or pred.normalized.text.strip())


def is_full_miss(pred: PredictionRow | None) -> bool:
    """No prediction, or an error other than a partial page, or an error with no output."""
    if pred is None:
        return True
    if pred.raw.error is None:
        return False
    return not (is_partial(pred) and _has_output(pred))


def answer_text(pred: PredictionRow | None) -> str | None:
    """The model's answer in its final documented form, as the official scorer gets it:
    the reply itself for benchmark-question requests, else the normalized page text
    (the model's own post-processing: OTSL->HTML, layout JSON -> markdown). None for a
    full miss."""
    if pred is None or is_full_miss(pred):
        return None
    if pred.prompt_version.startswith("question-"):
        return pred.raw.text
    return pred.normalized.text


def gt_answer(gt: GroundTruth) -> str:
    """The original ThaiOCRBench answer string for the official scorer."""
    if isinstance(gt, TextGT):
        return gt.text
    if isinstance(gt, HtmlGT):
        # [IMPLEMENTER DECIDES] the verbatim answer is kept in `raw` from this milestone on;
        # ground truth prepared earlier only has the HTML, where the ~10% of table
        # answers that were markdown were converted (the official scorer would see the
        # markdown, which it cannot parse as a table).
        return gt.raw if gt.raw is not None else gt.html
    if isinstance(gt, JsonGT):
        if gt.raw is not None:
            return gt.raw
        if gt.label is not None:
            return gt.label
        return json.dumps(gt.fields, ensure_ascii=False)
    if isinstance(gt, TextLayerGT):
        return gt.gt_text
    return ""


def _pred_table_html(pred: PredictionRow) -> str:
    """The prediction text if it holds a `<table>`; else its first parsed table."""
    if first_table(pred.normalized.text) is not None or not pred.normalized.tables:
        return pred.normalized.text
    rows = pred.normalized.tables[0]
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table>{body}</table>"


def _pred_fields(pred: PredictionRow) -> dict[str, str | None]:
    return {
        k: (fv.raw if fv.raw is not None else fv.value) for k, fv in pred.normalized.fields.items()
    }


Overrides = dict[str, BankOverrides]


def overrides_for(row: ManifestRow, overrides: Overrides | None) -> BankOverrides | None:
    """The bank's parsing overrides, if the dataset config has any for it."""
    if not overrides or row.bank is None:
        return None
    return overrides.get(row.bank)


def gt_statement(
    row: ManifestRow, gt: GroundTruth, overrides: Overrides | None
) -> tuple[StatementPage | None, bool]:
    """The ground-truth statement page and whether its column header was found.

    A text layer is parsed on the fly unless `prepare` already stored a parsed
    `statement`; a manual label (`JsonGT.statement`) is ground truth as given.
    """
    if isinstance(gt, JsonGT):
        return gt.statement, gt.statement is not None
    if isinstance(gt, TextLayerGT):
        if gt.statement is not None:
            return gt.statement, True
        parsed = parse_text_layer(
            gt, bank=row.bank, page_no=row.page_no, overrides=overrides_for(row, overrides)
        )
        return parsed.page, parsed.has_header
    return None, False


def pred_statement(
    row: ManifestRow, pred: PredictionRow, overrides: Overrides | None
) -> StatementPage:
    """The model's statement page, mapped by the shared rule-based mapper."""
    return map_statement_page(
        pred.normalized,
        bank=row.bank,
        page_no=row.page_no,
        overrides=overrides_for(row, overrides),
    )


def _tob_task(row: ManifestRow) -> str | None:
    if row.source == Source.thaiocrbench and row.subtask in tob_official.KEPT_TASKS:
        return row.subtask
    return None


def _metric_values(
    row: ManifestRow, gt: GroundTruth, pred: PredictionRow, overrides: Overrides | None = None
) -> tuple[dict[str, float | None], list[FieldScore], dict[str, dict]]:
    """Metric values for a scoreable prediction (not a full miss).

    The third element carries per-metric `extra` (only statements use it so far, to say
    why a row metric is not applicable)."""
    task = row.task
    text = pred.normalized.text
    answer = answer_text(pred)
    tob_task = _tob_task(row)
    values: dict[str, float | None] = {}
    fields: list[FieldScore] = []
    metric_extra: dict[str, dict] = {}

    if isinstance(gt, NoGT):
        return {"no_gt": None}, [], {}
    if task in _TEXT_TASKS and isinstance(gt, TextGT):
        values["cer"] = cer(text, gt.text)
        values["wer"] = wer(text, gt.text)
        bmfl = 0.0 if not answer else tob_official.bmfl(answer, gt.text)
        values["bmfl"] = bmfl
        if tob_task in tob_official.BMFL_TASKS:
            values["tob_score"] = bmfl  # identical by definition; computed once
    elif task == Task.table and isinstance(gt, HtmlGT):
        values["ted"] = ted(_pred_table_html(pred), gt.html)
        values["cer"] = cer(text, gt.html)
    elif task == Task.docparse and isinstance(gt, TextGT):
        values["ted"] = ted_docparse(answer or "", gt.text)
        values["cer"] = cer(text, gt.text)
    elif task in _KIE_TASKS and isinstance(gt, JsonGT):
        fields = score_fields(_pred_fields(pred), gt.fields)
        p, r, f1 = micro_prf(fields)
        values.update(kie_precision=p, kie_recall=r, kie_f1=f1)
    elif task == Task.classify and isinstance(gt, JsonGT):
        label = pred.normalized.label if pred.normalized.label is not None else pred.raw.text
        values["anls"] = anls(label.strip(), gt.label or "")
    elif task == Task.statement and _has_statement_gt(gt):
        if isinstance(gt, TextLayerGT):
            values["cer"] = cer(text, gt.gt_text)
        gt_page, has_header = gt_statement(row, gt, overrides)
        statement_values, fields = score_statement(
            pred_statement(row, pred, overrides), gt_page, has_header
        )
        values.update(statement_values)
        if not has_header:
            metric_extra = {name: {"reason": "no_header"} for name in statement_values}
    else:
        return {"not_applicable": None}, [], {}

    if tob_task is not None and "tob_score" not in values:
        values["tob_score"] = tob_official.tob_score(tob_task, answer, gt_answer(gt), row.question)
    return values, fields, metric_extra


def _has_statement_gt(gt: GroundTruth) -> bool:
    """True for ground truth that carries statement rows/fields: a text layer, or a
    manual label whose `statement` was filled in by `ocrbench review load`."""
    if isinstance(gt, TextLayerGT):
        return True
    return isinstance(gt, JsonGT) and gt.statement is not None


def _worst_values(
    row: ManifestRow, gt: GroundTruth, overrides: Overrides | None = None
) -> tuple[dict[str, float | None], list, dict[str, dict]]:
    """The metric names a scoreable prediction would get, at their worst values."""
    task = row.task
    if isinstance(gt, NoGT):
        return {"no_gt": None}, [], {}
    if task == Task.statement and _has_statement_gt(gt):
        gt_page, has_header = gt_statement(row, gt, overrides)
        values, fields = worst_case_statement(gt_page, has_header)
        if isinstance(gt, TextLayerGT):
            values["cer"] = _WORST["cer"]
        extra = {} if has_header else {n: {"reason": "no_header"} for n in values if n != "cer"}
        return values, fields, extra
    if task in _TEXT_TASKS and isinstance(gt, TextGT):
        names = ["cer", "wer", "bmfl"]
    elif task == Task.table and isinstance(gt, HtmlGT):
        names = ["ted", "cer"]
    elif task == Task.docparse and isinstance(gt, TextGT):
        names = ["ted", "cer"]
    elif task in _KIE_TASKS and isinstance(gt, JsonGT):
        names = ["kie_precision", "kie_recall", "kie_f1"]
    elif task == Task.classify and isinstance(gt, JsonGT):
        names = ["anls"]
    else:
        return {"not_applicable": None}, [], {}
    if _tob_task(row) is not None:
        names.append("tob_score")
    fields = worst_case_fields(gt.fields) if isinstance(gt, JsonGT) and task in _KIE_TASKS else []
    return {n: _WORST.get(n, 0.0) for n in names}, fields, {}


def score_sample(
    row: ManifestRow,
    gt: GroundTruth,
    pred: PredictionRow | None,
    model: str,
    overrides: Overrides | None = None,
) -> SampleScores:
    """Score rows (and field rows) for one manifest row and one model's prediction."""
    extra: dict = {}
    if pred is None:
        extra = {"full_miss": True, "missing_prediction": True}
        values, fields, metric_extra = _worst_values(row, gt, overrides)
    elif is_full_miss(pred):
        extra = {"full_miss": True, "error": pred.raw.error}
        values, fields, metric_extra = _worst_values(row, gt, overrides)
    else:
        if pred.raw.error is not None:
            extra = {"partial": True, "error": pred.raw.error}
        values, fields, metric_extra = _metric_values(row, gt, pred, overrides)

    slice_keys = dict(
        sample_id=row.sample_id,
        condition=row.condition,
        model=model,
        task=row.task,
        subtask=row.subtask,
        domain=row.domain,
        bank=row.bank,
        doc_type=row.doc_type,
    )
    out = SampleScores()
    for metric in sorted(values):
        value = values[metric]
        out.scores.append(
            ScoreRow(
                **slice_keys,
                metric=metric,
                value=None if value is None else float(value),
                is_critical=None,
                extra={**extra, **metric_extra.get(metric, {})},
            )
        )
    critical = set(row.critical_fields)
    for f in fields:
        out.fields.append(
            FieldResult(
                **slice_keys,
                field=f.field,
                is_critical=f.field in critical,
                pred=f.pred,
                gt=f.gt,
                field_exact=f.field_exact,
                field_fuzzy=f.field_fuzzy,
                false_accept=f.false_accept,
                false_reject=f.false_reject,
            )
        )
    return out


def score_run(
    manifest: list[ManifestRow],
    latest: dict[tuple[str, str, str], PredictionRow],
    load_gt: Callable[[ManifestRow], GroundTruth],
    overrides: Overrides | None = None,
) -> tuple[list[ScoreRow], list[FieldResult]]:
    """Score every manifest row for every model that has predictions in the run.

    [IMPLEMENTER DECIDES] a manifest row a model has no prediction for is scored as a
    full miss (flagged `missing_prediction`), so no page silently drops out of a slice.
    Output is sorted deterministically.
    """
    models = sorted({model for (_, _, model) in latest})
    scores: list[ScoreRow] = []
    fields: list[FieldResult] = []
    for row in manifest:
        gt = load_gt(row)
        for model in models:
            pred = latest.get((row.sample_id, row.condition.value, model))
            result = score_sample(row, gt, pred, model, overrides)
            scores.extend(result.scores)
            fields.extend(result.fields)
    scores.sort(key=lambda s: (s.sample_id, s.condition.value, s.model, s.metric))
    fields.sort(key=lambda f: (f.sample_id, f.condition.value, f.model, f.field))
    return scores, fields
