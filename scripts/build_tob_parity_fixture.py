"""Build `ocr_bench/metrics/tob_parity/fixture.json` from ThaiOCRBench's published results.

Usage:
    uv run python scripts/build_tob_parity_fixture.py <res_folder>/<qwen2.5-vl-72b>.json

The input is a ThaiOCRBench `res_folder` result file: a list of
`{type, id, question, answers, predict, score}`. For each kept task, samples are sorted by
id and split into score bins {0, (0, 1), 1}; `PER_TASK` samples are drawn across the bins
in proportion to their sizes (largest remainder, at least one from every non-empty bin)
with `random.Random(42)`, then written sorted by (task, id).
"""

import json
import math
import random
import sys
from pathlib import Path

from ocr_bench.metrics.tob_official import KEPT_TASKS

PER_TASK = 24
SEED = 42
OUT = Path(__file__).resolve().parents[1] / "ocr_bench/metrics/tob_parity/fixture.json"


def _bin(score: float) -> str:
    return "zero" if score == 0 else "one" if score == 1 else "mid"


def _quotas(sizes: dict[str, int], n: int) -> dict[str, int]:
    total = sum(sizes.values())
    n = min(n, total)
    exact = {b: n * s / total for b, s in sizes.items()}
    quotas = {b: max(1 if sizes[b] else 0, math.floor(e)) for b, e in exact.items()}
    by_fraction = sorted(sizes, key=lambda b: (-(exact[b] - math.floor(exact[b])), b))
    i = 0
    while sum(quotas.values()) < n:
        b = by_fraction[i % len(by_fraction)]
        if quotas[b] < sizes[b]:
            quotas[b] += 1
        i += 1
    while sum(quotas.values()) > n:
        b = max(quotas, key=lambda k: (quotas[k], k))
        quotas[b] -= 1
    return quotas


def select(items: list[dict], task: str) -> list[dict]:
    rows = sorted((x for x in items if x["type"] == task), key=lambda x: x["id"])
    bins: dict[str, list[dict]] = {"zero": [], "mid": [], "one": []}
    for x in rows:
        bins[_bin(x["score"])].append(x)
    quotas = _quotas({b: len(v) for b, v in bins.items()}, PER_TASK)
    rng = random.Random(SEED)
    chosen: list[dict] = []
    for b in ("zero", "mid", "one"):
        chosen.extend(rng.sample(bins[b], quotas[b]))
    return sorted(chosen, key=lambda x: x["id"])


def main(src: str) -> None:
    items = json.loads(Path(src).read_text(encoding="utf-8"))
    fixture = []
    for task in KEPT_TASKS:
        for x in select(items, task):
            fixture.append(
                {
                    "task": task,
                    "id": x["id"],
                    "question": x["question"],
                    "answers": x["answers"],
                    "predict": x["predict"],
                    "published_score": x["score"],
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(fixture)} samples to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
