"""Check C: cross-model agreement on statement pages (task 5.4).

Two models' pages are compared field by field and row by row, with both sides passed
through the same normalizer as the accuracy scorers. A value present in one model and
absent in the other is `None` — not counted as agreement or disagreement — per the
spec's "missing in one model" scenario. Rows are aligned on (date, amount), and the
balance is compared on the matched pairs; a row only one model saw is a disagreement,
since that is exactly the kind of miss the review queue exists to catch.

`field_agreement` is also M5's per-field agreement feature for the confidence proxy.
"""

from itertools import combinations

from ocr_bench.metrics.statement_gt import SCORED_FIELDS, field_map, row_key
from ocr_bench.normalize.fields import normalize_field
from ocr_bench.schemas import StatementPage, StatementRow

Agreement = dict[str, bool | None]


def _normalized(field: str, value: str | None) -> str | None:
    if value is None:
        return None
    return normalize_field(field, value).value


def _row_groups(rows: list[StatementRow]) -> dict[str, list[StatementRow]]:
    groups: dict[str, list[StatementRow]] = {}
    for row in rows:
        date, amount = row_key(row)
        groups.setdefault(f"row:{date or ''}|{amount or ''}", []).append(row)
    return groups


def field_agreement(a: StatementPage | None, b: StatementPage | None) -> Agreement:
    """`{key: True | False | None}` over header fields and rows of one page pair."""
    map_a, map_b = field_map(a), field_map(b)
    result: Agreement = {}
    for name in SCORED_FIELDS:
        left, right = _normalized(name, map_a[name]), _normalized(name, map_b[name])
        result[name] = None if left is None or right is None else left == right

    groups_a = _row_groups(a.rows if a else [])
    groups_b = _row_groups(b.rows if b else [])
    if a is None or b is None:
        # A page the model did not produce at all: nothing to compare, not a disagreement.
        for key in sorted(set(groups_a) | set(groups_b)):
            for index in range(max(len(groups_a.get(key, [])), len(groups_b.get(key, [])))):
                result[f"{key}#{index}:present"] = None
        return result

    for key in sorted(set(groups_a) | set(groups_b)):
        rows_a, rows_b = groups_a.get(key, []), groups_b.get(key, [])
        for index in range(max(len(rows_a), len(rows_b))):
            if index >= len(rows_a) or index >= len(rows_b):
                result[f"{key}#{index}:present"] = False
                continue
            left, right = rows_a[index].balance, rows_b[index].balance
            result[f"{key}#{index}:balance"] = (
                None if left is None or right is None else left == right
            )
    return result


def agreement_rate(agreement: Agreement) -> float | None:
    """The share of comparable keys that agree, or None when nothing is comparable."""
    values = [v for v in agreement.values() if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def pair_agreements(pages: dict[str, StatementPage | None]) -> dict[tuple[str, str], Agreement]:
    """`{(model_a, model_b): agreement}` for every model pair, model names sorted."""
    return {(a, b): field_agreement(pages[a], pages[b]) for a, b in combinations(sorted(pages), 2)}
