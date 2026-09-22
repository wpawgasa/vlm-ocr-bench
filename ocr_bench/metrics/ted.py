"""Tree-edit-distance similarity for tables and documents (task 4.5).

`ted` is our table TEDS (1 = identical trees), built on the TEDS class ported in
`tob_official`. It differs from the official `tob_score` for Table parsing in the inputs
it is given: both sides are canonicalized (`canonical_text`), and
[IMPLEMENTER DECIDES] `<th>` cells are scored as `<td>` (PubTabNet's TEDS ignores the
text of any non-`td` node, which would leave header text unscored) and `thead`/`tbody`/
`tfoot` wrappers are ignored, so a model is not penalized for how it groups rows.

For docparse, `ted_docparse` *is* the official document-parsing score
(`doc_parsing_evaluation`, a structure-aware tree edit similarity over the markdown
heading tree), so the `ted` and `tob_score` rows of a docparse sample coincide.
"""

import re

from ocr_bench.metrics.tob_official import TEDS, doc_parsing_evaluation, wrap_html_table
from ocr_bench.normalize.text import canonical_text

_TABLE_RE = re.compile(r"<table\b.*?</table\s*>", re.IGNORECASE | re.DOTALL)
_TH_OPEN_RE = re.compile(r"<th\b", re.IGNORECASE)
_TH_CLOSE_RE = re.compile(r"</th\s*>", re.IGNORECASE)

_TEDS = TEDS(structure_only=False, ignore_nodes=["thead", "tbody", "tfoot"])


def first_table(text: str) -> str | None:
    """The first `<table>…</table>` in `text`, or None."""
    m = _TABLE_RE.search(text)
    return m.group(0) if m else None


def _prep(table_html: str) -> str:
    s = _TH_CLOSE_RE.sub("</td>", _TH_OPEN_RE.sub("<td", table_html))
    return wrap_html_table(canonical_text(s))


def ted(pred_html: str, gt_html: str) -> float:
    """TEDS between the first table in the prediction and the ground-truth table;
    0 when the prediction has no `<table>`."""
    pred_table = first_table(pred_html)
    if pred_table is None:
        return 0.0
    gt_table = first_table(gt_html) or gt_html
    try:
        return float(_TEDS.evaluate(_prep(pred_table), _prep(gt_table)))
    except Exception:
        return 0.0


def ted_docparse(pred_text: str, gt_text: str) -> float:
    """The official document-parsing score (equal to docparse `tob_score`)."""
    return float(doc_parsing_evaluation(pred_text, gt_text))
