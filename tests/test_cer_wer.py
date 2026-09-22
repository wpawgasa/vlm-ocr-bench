"""Tests for ocr_bench.metrics.cer_wer (task 4.3)."""

import pytest

from ocr_bench.metrics.cer_wer import cer, wer


def test_identical_text_is_zero():
    assert cer("สวัสดีครับ", "สวัสดีครับ") == 0.0
    assert wer("สวัสดีครับ", "สวัสดีครับ") == 0.0


def test_identical_after_normalization_is_zero():
    assert cer("<p>ยอด ๑๐๐</p>", "ยอด  100") == 0.0


def test_hand_computed_cer():
    # one substitution in 5 characters
    assert cer("abxde", "abcde") == pytest.approx(0.2)
    # one deletion in 4 characters
    assert cer("abc", "abcd") == pytest.approx(0.25)


def test_hand_computed_wer_whitespace_tokens_ignored():
    # English tokens via newmm: "the", "cat", "sat" (whitespace tokens dropped)
    assert wer("the dog sat", "the cat sat") == pytest.approx(1 / 3)


def test_cer_capped_at_one():
    assert cer("x" * 100, "ab") == 1.0
    assert wer("a b c d e f", "z") == 1.0


def test_empty_gt_uses_denominator_one():
    assert cer("", "") == 0.0
    assert cer("ab", "") == 1.0


def test_empty_prediction_is_one():
    assert cer("", "abc") == 1.0
