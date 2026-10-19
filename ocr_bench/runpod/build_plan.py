"""Which Runpod worker images to build, and with what build args (used by runpod-images.yml).

    python -m ocr_bench.runpod.build_plan --changed runpod/Dockerfile configs/runpod/dotsocr.yaml
    python -m ocr_bench.runpod.build_plan --model dotsocr

Prints a JSON list for a GitHub Actions matrix: one entry per model with the stage
(`variant`) and the values the Dockerfile takes as build args.
"""

import argparse
import json
import sys
from pathlib import Path

from ocr_bench.runpod.config import load_runpod_config

_NOT_MODELS = {"data_centers", "parity_samples.v1", "releases"}  # releases: the deploy log


def _all_models(configs_dir: Path) -> list[str]:
    return sorted(p.stem for p in configs_dir.glob("*.yaml") if p.stem not in _NOT_MODELS)


def affected_models(changed: list[str], configs_dir: Path) -> list[str]:
    """Models whose image changes: all of them for `runpod/**`, else the changed configs."""
    models = _all_models(configs_dir)
    if any(p.startswith("runpod/") for p in changed):
        return models
    prefix = "configs/runpod/"
    touched = {Path(p).stem for p in changed if p.startswith(prefix) and p.endswith(".yaml")}
    return [m for m in models if m in touched]


def build_matrix(models: list[str], configs_dir: Path) -> list[dict[str, str]]:
    matrix = []
    for model in models:
        cfg = load_runpod_config(configs_dir / f"{model}.yaml")
        matrix.append(
            {
                "model": model,
                "variant": cfg.image.variant,
                "base": cfg.image.base,
                "repo": cfg.weights.repo,
                "revision": cfg.weights.revision,
            }
        )
    return matrix


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--changed", nargs="*", help="changed file paths (repo-relative)")
    group.add_argument("--model", help="build exactly this model")
    ap.add_argument("--configs-dir", type=Path, default=Path("configs/runpod"))
    args = ap.parse_args(argv)
    if args.model:
        if args.model not in _all_models(args.configs_dir):
            print(f"unknown model {args.model!r}", file=sys.stderr)
            return 2
        models = [args.model]
    else:
        models = affected_models(args.changed, args.configs_dir)
    print(json.dumps(build_matrix(models, args.configs_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
