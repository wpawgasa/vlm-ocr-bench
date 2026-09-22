"""Per-field confidence features for the calibration proxy (task 6.1, D10).

`locate_span` finds where a field's raw value sits in the concatenated completion
tokens, first by an exact (last-occurrence) match, then by a digit-only fallback for
amounts whose formatting drifted (thousands separators, spacing). `field_features`
turns that into the feature set the logistic proxy is fit on: logprob statistics over
the located span (falling back to the whole completion), cross-model agreement and the
arithmetic-consistency flag, both null-indicator encoded per D10.
"""

import bisect
from dataclasses import dataclass

from ocr_bench.normalize.fields import normalize_field
from ocr_bench.schemas import PredictionRow

#: Feature order for the `full` design matrix (D10): logprob stats plus null-indicator
#: encoded agreement and arithmetic flags.
FULL_FEATURES: tuple[str, ...] = (
    "mean_logprob",
    "min_logprob",
    "agree_1",
    "agree_null",
    "arith_1",
    "arith_null",
)
#: Feature order for the `logprob_only` variant (single-model production use).
LOGPROB_FEATURES: tuple[str, ...] = ("mean_logprob", "min_logprob")

Variant = str  # "full" | "logprob_only"

# [IMPLEMENTER DECIDES] with 3 models running, `agree` is computed against one "other"
# model, not averaged across all others (the spec gives no aggregation rule). The
# dots.ocr/teleocr pair is primary, since both are native-OCR references the report
# leans on most; a model outside that pair (typhoon) falls back to the alphabetically
# first other model that has a value. Handover: "With 3 models, use the dots-vs-teleocr
# pair primarily."
PRIMARY_PAIR = ("dots.ocr", "teleocr")


def locate_span(tokens: list[str], value_raw: str | None) -> tuple[int, int] | None:
    """The `[start, end)` token index range covering `value_raw` in `tokens`' completion.

    Exact match wins, taking the *last* occurrence (JSON keys can repeat a value's
    digits before the value itself appears). Failing that, a digit-only match locates
    amounts whose separators the model reformatted. Returns None if neither matches.
    """
    if not tokens or not value_raw:
        return None

    starts: list[int] = []
    pos = 0
    for tok in tokens:
        starts.append(pos)
        pos += len(tok)
    completion = "".join(tokens)

    def char_to_token(char_pos: int) -> int:
        return bisect.bisect_right(starts, char_pos) - 1

    idx = completion.rfind(value_raw)
    if idx != -1:
        start_tok = char_to_token(idx)
        end_tok = char_to_token(idx + len(value_raw) - 1)
        return start_tok, end_tok + 1

    def digits_with_positions(s: str) -> tuple[str, list[int]]:
        digits = []
        positions = []
        for i, ch in enumerate(s):
            if ch.isdigit():
                digits.append(ch)
                positions.append(i)
        return "".join(digits), positions

    comp_digits, comp_positions = digits_with_positions(completion)
    val_digits, _ = digits_with_positions(value_raw)
    if not val_digits:
        return None
    didx = comp_digits.rfind(val_digits)
    if didx == -1:
        return None
    start_char = comp_positions[didx]
    end_char = comp_positions[didx + len(val_digits) - 1]
    start_tok = char_to_token(start_char)
    end_tok = char_to_token(end_char)
    return start_tok, end_tok + 1


@dataclass(frozen=True)
class FieldFeatures:
    mean_logprob: float
    min_logprob: float
    span_found: bool
    logprobs_missing: bool
    agree: int | None
    arith: int | None

    def vector(self, variant: Variant) -> list[float]:
        """The design-matrix row for `variant` (D10's null-indicator encoding)."""
        if variant == "logprob_only":
            return [self.mean_logprob, self.min_logprob]
        return [
            self.mean_logprob,
            self.min_logprob,
            1.0 if self.agree == 1 else 0.0,
            1.0 if self.agree is None else 0.0,
            1.0 if self.arith == 1 else 0.0,
            1.0 if self.arith is None else 0.0,
        ]


def feature_names(variant: Variant) -> tuple[str, ...]:
    return LOGPROB_FEATURES if variant == "logprob_only" else FULL_FEATURES


def _agree(
    self_model: str,
    field_key: str,
    field_value: str | None,
    other_value_by_model: dict[str, str | None],
) -> int | None:
    if not other_value_by_model:
        return None
    partner = None
    if self_model in PRIMARY_PAIR:
        candidate = PRIMARY_PAIR[1] if self_model == PRIMARY_PAIR[0] else PRIMARY_PAIR[0]
        if candidate in other_value_by_model:
            partner = candidate
    if partner is None:
        partner = sorted(other_value_by_model)[0]
    other_raw = other_value_by_model[partner]
    if other_raw is None:
        return None
    self_norm = normalize_field(field_key, field_value).value if field_value is not None else None
    other_norm = normalize_field(field_key, other_raw).value
    # [IMPLEMENTER DECIDES] both sides null (neither model saw the field) counts as
    # agreement, since it is not the "other model has no value" null case the spec
    # names (that case is already excluded above by `other_raw is None`).
    return int(self_norm == other_norm)


def field_features(
    pred_row: PredictionRow,
    field_key: str,
    field_value: str | None,
    *,
    other_value_by_model: dict[str, str | None],
    arith_flag: bool | None,
) -> FieldFeatures:
    """One field's confidence features (D10).

    `field_value` is the field's raw (pre-normalization) predicted string, used both to
    locate the span in the completion and, normalized, to compare against
    `other_value_by_model` (each entry already the other model's *normalized* value, or
    None when that model has no value for the field).
    """
    tokens = pred_row.raw.tokens
    logprobs = pred_row.raw.token_logprobs
    span = locate_span(tokens, field_value) if field_value else None
    span_found = span is not None

    if not logprobs:
        mean_lp = 0.0
        min_lp = 0.0
        logprobs_missing = True
    else:
        logprobs_missing = False
        lp_slice = logprobs[span[0] : span[1]] if span is not None else []
        if not lp_slice:
            lp_slice = logprobs
            span_found = False
        mean_lp = sum(lp_slice) / len(lp_slice)
        min_lp = min(lp_slice)

    agree = _agree(pred_row.model, field_key, field_value, other_value_by_model)
    arith = None if arith_flag is None else int(bool(arith_flag))

    return FieldFeatures(
        mean_logprob=mean_lp,
        min_logprob=min_lp,
        span_found=span_found,
        logprobs_missing=logprobs_missing,
        agree=agree,
        arith=arith,
    )
