"""Character and word error rates over the markup-free canonical view (task 4.3).

Both sides go through `plain_text`. Both rates are capped at 1.0: a hallucination loop
(a prediction much longer than the ground truth) would otherwise score arbitrarily worse
than an empty answer, and one runaway sample would dominate a slice mean.
"""

from functools import lru_cache

from pythainlp.tokenize import word_tokenize
from rapidfuzz.distance import Levenshtein

from ocr_bench.normalize.text import plain_text


def _rate(dist: int, gt_len: int) -> float:
    return min(1.0, dist / max(1, gt_len))


def cer(pred: str, gt: str) -> float:
    """min(1, Levenshtein(pred, gt) / max(1, len(gt))) over `plain_text` of each side."""
    p, g = plain_text(pred), plain_text(gt)
    return _rate(Levenshtein.distance(p, g), len(g))


@lru_cache(maxsize=4096)
def _tokens(text: str) -> tuple[str, ...]:
    return tuple(t for t in word_tokenize(text, engine="newmm") if t.strip())


def wer(pred: str, gt: str) -> float:
    """CER's formula over PyThaiNLP `newmm` word tokens (whitespace tokens dropped)."""
    p, g = _tokens(plain_text(pred)), _tokens(plain_text(gt))
    return _rate(Levenshtein.distance(p, g), len(g))
