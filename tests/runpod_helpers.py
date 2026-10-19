"""Shared by the runpod render/CLI tests: a configs tree with pinned weights revisions."""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REV = "b" * 40
MODELS = ["dotsocr", "ovisocr2", "paddleocr_vl", "teleocr", "typhoon_ocr15"]


def pinned_configs(dest: Path) -> Path:
    """Copy configs/ to `dest` with each `weights.revision` set to the fixed test hash REV."""
    shutil.copytree(ROOT / "configs" / "models", dest / "models")
    (dest / "runpod").mkdir()
    for model in MODELS:
        text = (ROOT / "configs" / "runpod" / f"{model}.yaml").read_text()
        (dest / "runpod" / f"{model}.yaml").write_text(
            re.sub(r"revision: \S+", f"revision: {REV}", text, count=1)
        )
    return dest / "runpod"
