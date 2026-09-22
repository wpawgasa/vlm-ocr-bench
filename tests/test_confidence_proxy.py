"""Tests for ocr_bench.confidence.proxy (task 6.1)."""

from ocr_bench.confidence.proxy import FULL_FEATURES, LOGPROB_FEATURES, field_features, locate_span
from ocr_bench.schemas import Condition, NormalizedPage, PredictionRow, RawPrediction


def _pred_row(tokens: list[str], logprobs: list[float], model: str = "teleocr") -> PredictionRow:
    text = "".join(tokens)
    return PredictionRow(
        sample_id="s1",
        condition=Condition.clean,
        model=model,
        prompt_version="question-v1",
        raw=RawPrediction(text=text, tokens=tokens, token_logprobs=logprobs),
        normalized=NormalizedPage(text=text),
        latency_ms=0.0,
        prompt_tokens=0,
        completion_tokens=len(tokens),
    )


def test_locate_span_exact_match():
    tokens = ['{"total": ', '"1234"', "}"]
    span = locate_span(tokens, "1234")
    assert span == (1, 2)


def test_locate_span_not_found_returns_none():
    tokens = ["hello", " world"]
    assert locate_span(tokens, "9999") is None


def test_locate_span_digit_only_fallback():
    # completion has no comma ("1234.50"); the raw value carries one ("1,234.50"). The
    # exact search fails, and the digit-only fallback matches on stripped digits.
    tokens = ["Total: ", "1234", ".50", " THB"]
    span = locate_span(tokens, "1,234.50")
    assert span == (1, 3)


def test_locate_span_picks_last_occurrence():
    # The field's JSON key text repeats the value's digits before the value itself
    # appears (a key like "amount_1234"), so the last occurrence must win.
    tokens = ["amount_1234", " is ", "1234"]
    span = locate_span(tokens, "1234")
    assert span == (2, 3)


def test_locate_span_empty_value_is_none():
    assert locate_span(["a", "b"], "") is None
    assert locate_span(["a", "b"], None) is None


def test_locate_span_empty_tokens_is_none():
    assert locate_span([], "1234") is None


def test_field_features_span_not_found_uses_whole_completion_stats():
    pred = _pred_row(["hello", " world"], [-0.1, -0.2])
    feats = field_features(pred, "field", "9999", other_value_by_model={}, arith_flag=None)
    assert feats.span_found is False
    assert feats.mean_logprob == (-0.1 + -0.2) / 2
    assert feats.min_logprob == -0.2
    assert feats.logprobs_missing is False


def test_field_features_span_found_uses_span_stats():
    tokens = ['{"total": ', '"1234"', "}"]
    logprobs = [-0.05, -0.9, -0.01]
    pred = _pred_row(tokens, logprobs)
    feats = field_features(pred, "total", "1234", other_value_by_model={}, arith_flag=None)
    assert feats.span_found is True
    assert feats.mean_logprob == -0.9
    assert feats.min_logprob == -0.9


def test_field_features_no_logprobs_at_all():
    pred = _pred_row(["hello"], [])
    feats = field_features(pred, "field", "hello", other_value_by_model={}, arith_flag=None)
    assert feats.mean_logprob == 0.0
    assert feats.min_logprob == 0.0
    assert feats.logprobs_missing is True


def test_field_features_agree_null_when_no_other_model():
    pred = _pred_row(["1234"], [-0.1])
    feats = field_features(pred, "total_amount", "1234", other_value_by_model={}, arith_flag=None)
    assert feats.agree is None


def test_field_features_agree_null_when_other_has_no_value():
    pred = _pred_row(["1234"], [-0.1], model="teleocr")
    feats = field_features(
        pred, "total_amount", "1234", other_value_by_model={"dots.ocr": None}, arith_flag=None
    )
    assert feats.agree is None


def test_field_features_agree_matches():
    pred = _pred_row(["1,234.00"], [-0.1], model="teleocr")
    feats = field_features(
        pred,
        "total_amount",
        "1,234.00",
        other_value_by_model={"dots.ocr": "1234.00"},
        arith_flag=None,
    )
    assert feats.agree == 1


def test_field_features_agree_differs():
    pred = _pred_row(["1,234.00"], [-0.1], model="teleocr")
    feats = field_features(
        pred,
        "total_amount",
        "1,234.00",
        other_value_by_model={"dots.ocr": "9999.00"},
        arith_flag=None,
    )
    assert feats.agree == 0


def test_field_features_agree_uses_dots_teleocr_pair_over_a_third_model():
    # A third model's value is present too, but the dots.ocr/teleocr pair is primary.
    pred = _pred_row(["1,234.00"], [-0.1], model="teleocr")
    feats = field_features(
        pred,
        "total_amount",
        "1,234.00",
        other_value_by_model={"dots.ocr": "9999.00", "typhoon": "1234.00"},
        arith_flag=None,
    )
    assert feats.agree == 0  # matches typhoon, but dots.ocr disagrees and wins


def test_field_features_arith_indicator_passthrough():
    pred = _pred_row(["1234"], [-0.1])
    assert field_features(pred, "f", "1234", other_value_by_model={}, arith_flag=True).arith == 1
    assert field_features(pred, "f", "1234", other_value_by_model={}, arith_flag=False).arith == 0
    assert field_features(pred, "f", "1234", other_value_by_model={}, arith_flag=None).arith is None


def test_design_vectors_use_null_indicators():
    pred = _pred_row(["1234"], [-0.1, -0.2])
    feats = field_features(pred, "f", "1234", other_value_by_model={}, arith_flag=None)
    assert feats.agree is None and feats.arith is None
    full = feats.vector("full")
    assert full == [feats.mean_logprob, feats.min_logprob, 0.0, 1.0, 0.0, 1.0]
    assert feats.vector("logprob_only") == [feats.mean_logprob, feats.min_logprob]


def test_feature_name_lists():
    assert FULL_FEATURES == (
        "mean_logprob",
        "min_logprob",
        "agree_1",
        "agree_null",
        "arith_1",
        "arith_null",
    )
    assert LOGPROB_FEATURES == ("mean_logprob", "min_logprob")
