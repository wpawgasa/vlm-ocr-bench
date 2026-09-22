"""Average Normalized Levenshtein Similarity for one answer (task 4.5).

Standard ANLS (Biten et al., 2019) on canonical, lower-cased text: similarity
1 - Levenshtein / max(len) if that is >= `threshold`, else 0.
"""

from rapidfuzz.distance import Levenshtein

from ocr_bench.normalize.text import canonical_text


def anls(pred: str | None, gt: str, threshold: float = 0.5) -> float:
    p = canonical_text(pred or "").lower()
    g = canonical_text(gt).lower()
    length = max(len(p), len(g))
    if length == 0:
        return 1.0
    similarity = 1.0 - Levenshtein.distance(p, g) / length
    return similarity if similarity >= threshold else 0.0
