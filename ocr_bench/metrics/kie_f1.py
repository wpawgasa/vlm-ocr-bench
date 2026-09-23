"""Per-field KIE scoring with false accept / false reject, and micro P/R/F1 (task 4.6).

Both sides are flat `{dotted.key: value | None}` maps (as produced by
`gt_utils.flatten_fields`) and each value is normalized by `normalize_field` before
comparison.

`field_exact` is the headline metric. `field_lenient` is a separately reported, more
forgiving rule: the true value also counts as found when it appears as a whole token inside
a short predicted value ("ราคาตั๋ว / Ticket Price: 5 บาท / Baht" for "5").
"""

import re
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from ocr_bench.normalize.fields import normalize_field
from ocr_bench.normalize.text import canonical_text

FUZZY_CER = 0.1


@dataclass(frozen=True)
class FieldScore:
    field: str
    pred: str | None  # normalized prediction value (None = null or missing)
    gt: str | None  # normalized ground-truth value (None = null, or an extra pred field)
    field_exact: int
    field_fuzzy: int
    false_accept: bool
    false_reject: bool
    in_gt: bool  # False for a predicted key the ground truth does not have
    field_lenient: int = 0


#: A lenient match may carry at most this much extra text around the true value.
LENIENT_MAX_EXTRA = 40


def _lenient(pred_raw: str, gt_raw: str) -> bool:
    """The true value appears as a whole token in a predicted value that is not much longer
    (a label or unit around it), so "5" is found in "Price: 5 บาท" but not in "25"."""
    p, g = canonical_text(pred_raw).lower(), canonical_text(gt_raw).lower()
    if not g or len(p) > max(3 * len(g), len(g) + LENIENT_MAX_EXTRA):
        return False
    return re.search(rf"(?<![\w.,]){re.escape(g)}(?!\w|[.,]\d)", p) is not None


def _fuzzy(pred_raw: str, gt_raw: str) -> bool:
    p, g = canonical_text(pred_raw), canonical_text(gt_raw)
    return Levenshtein.distance(p, g) / max(1, len(g)) <= FUZZY_CER


def _null_if_blank(value: str | None) -> str | None:
    # [IMPLEMENTER DECIDES] a blank string is a null: ThaiOCRBench marks an absent value
    # with "" (it has no JSON nulls), and models answer "" for fields they cannot see.
    if value is None or canonical_text(value) == "":
        return None
    return value


def score_fields(pred: dict[str, str | None], gt: dict[str, str | None]) -> list[FieldScore]:
    """One `FieldScore` per key of `gt`, then per predicted key not in `gt` (sorted)."""
    rows: list[FieldScore] = []
    extras = sorted(k for k in pred if k not in gt)
    for key in [*gt.keys(), *extras]:
        in_gt = key in gt
        gt_raw = _null_if_blank(gt.get(key))
        pred_raw = _null_if_blank(pred.get(key))
        gt_val = normalize_field(key, gt_raw).value
        pred_val = normalize_field(key, pred_raw).value
        if gt_raw is None and pred_raw is None:
            exact = fuzzy = lenient = True
        elif gt_raw is None or pred_raw is None:
            exact = fuzzy = lenient = False
        else:
            exact = pred_val == gt_val
            fuzzy = exact or _fuzzy(pred_raw, gt_raw)
            lenient = exact or _lenient(pred_raw, gt_raw)
        rows.append(
            FieldScore(
                field=key,
                pred=pred_val,
                gt=gt_val,
                field_exact=int(exact),
                field_fuzzy=int(fuzzy),
                field_lenient=int(lenient),
                # [IMPLEMENTER DECIDES] an extra predicted field (key not in the ground
                # truth) has no true value, so a non-null prediction there is a false
                # accept, and it is a false positive in micro P/R.
                false_accept=gt_raw is None and pred_raw is not None,
                false_reject=gt_raw is not None and pred_raw is None,
                in_gt=in_gt,
            )
        )
    return rows


def worst_case_fields(gt: dict[str, str | None]) -> list[FieldScore]:
    """Rows for a failed request: no credit anywhere (even where the ground truth is
    null), a false reject wherever the ground truth has a value."""
    return [
        FieldScore(
            field=key,
            pred=None,
            gt=normalize_field(key, _null_if_blank(value)).value,
            field_exact=0,
            field_fuzzy=0,
            false_accept=False,
            false_reject=_null_if_blank(value) is not None,
            in_gt=True,
        )
        for key, value in gt.items()
    ]


def micro_prf(rows: list[FieldScore]) -> tuple[float, float, float]:
    """Micro precision, recall and F1 over non-null (field, value) pairs.

    A true positive is a non-null pair with `field_exact`; predicted pairs include
    extra fields. [IMPLEMENTER DECIDES] with no predicted pairs precision is 1 if the
    ground truth also has none, else 0 (and symmetrically for recall), so an all-null
    page answered all-null scores 1.
    """
    n_pred = sum(r.pred is not None for r in rows)
    n_gt = sum(r.gt is not None and r.in_gt for r in rows)
    tp = sum(r.field_exact == 1 and r.pred is not None and r.in_gt for r in rows)
    if n_pred:
        precision = tp / n_pred
    else:
        precision = 1.0 if n_gt == 0 else 0.0
    if n_gt:
        recall = tp / n_gt
    else:
        recall = 1.0 if n_pred == 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1
