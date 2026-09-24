"""Bank-statement ingestion: discovery, bank routing and digital/scanned/photo
classification (tasks 2.2 and 2.3).
"""

import fnmatch
import re
import subprocess
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pdfplumber
from PIL import Image, ImageSequence

from ocr_bench.config import BankStmtConfig, CorpusTargets
from ocr_bench.data.gt_utils import STATEMENT_CRITICAL_FIELDS
from ocr_bench.data.manifest import PreparedSample
from ocr_bench.data.pdf_repair import repair_tounicode
from ocr_bench.schemas import NoGT, Source, Task, TextLayerGT, Word

_DOCUMENT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class UnknownBankError(ValueError):
    pass


def discover(input_dir: Path) -> list[Path]:
    """Collect statement files recursively, sorted by POSIX relative path."""
    files = [
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in _DOCUMENT_SUFFIXES
    ]
    files.sort(key=lambda p: p.relative_to(input_dir).as_posix())
    return files


def bank_for(rel: str, cfg: BankStmtConfig) -> str:
    """Return the bank for `rel` per `cfg.bank_map`'s first matching pattern."""
    rel_lower = rel.lower()
    for pattern, bank in cfg.bank_map.items():
        if fnmatch.fnmatch(rel_lower, pattern):
            if bank not in cfg.banks:
                raise UnknownBankError(
                    f"file {rel!r} matched pattern {pattern!r} to bank {bank!r}, "
                    f"which is not in the configured banks {cfg.banks}"
                )
            return bank
    raise UnknownBankError(f"cannot determine bank for file {rel!r}")


def route(pdf: Path) -> Literal["digital", "scanned"]:
    """Classify `pdf` as `digital` (has embedded fonts) or `scanned` (no fonts)."""
    result = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    lines = result.stdout.splitlines()
    content_lines = lines[2:]
    return "digital" if any(line.strip() for line in content_lines) else "scanned"


def rasterize_page(pdf: Path, page_no: int, dpi: int) -> np.ndarray:
    """Rasterize one page of `pdf` at `dpi` to a BGR uint8 array."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_stem = Path(tmpdir) / "page"
        result = subprocess.run(
            [
                "pdftoppm",
                "-r",
                str(dpi),
                "-f",
                str(page_no),
                "-l",
                str(page_no),
                "-png",
                "-singlefile",
                str(pdf),
                str(out_stem),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return cv2.imread(str(out_stem) + ".png")


# Every real statement page carries at least one of these; a text layer with none is
# scrambled glyph codes (custom font encoding / wrong ToUnicode map, seen on KTB and
# Krungsri PDFs) and must not be used as ground truth.
_STATEMENT_KEYWORD_RE = re.compile(
    r"ยอด|บัญชี|วันที่|รายการ|คงเหลือ|ถอน|ฝาก|"
    r"balance|account|date|statement|deposit|withdraw|page",
    re.IGNORECASE,
)


def text_layer_usable(gt_text: str) -> bool:
    """True if an extracted statement text layer is readable text, not glyph codes."""
    return bool(_STATEMENT_KEYWORD_RE.search(unicodedata.normalize("NFC", gt_text)))


def text_layer(pdf: Path, page_no: int, dpi: int) -> TextLayerGT:
    """Extract the layout-preserving text and per-word boxes of one digital page."""
    result = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page_no), "-l", str(page_no), str(pdf), "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    gt_text = result.stdout

    scale = dpi / 72
    with pdfplumber.open(pdf) as doc:
        page = doc.pages[page_no - 1]
        words = page.extract_words()

    gt_words = [
        Word(
            text=w["text"],
            bbox=(w["x0"] * scale, w["top"] * scale, w["x1"] * scale, w["bottom"] * scale),
        )
        for w in words
    ]
    return TextLayerGT(gt_kind="text_layer", gt_text=gt_text, gt_words=gt_words)


def _page_count(pdf: Path) -> int:
    with pdfplumber.open(pdf) as doc:
        return len(doc.pages)


def _load_image_frames(path: Path) -> list[np.ndarray]:
    """Load an image file as a list of BGR uint8 frames (>1 only for multi-frame TIFF)."""
    frames = []
    with Image.open(path) as img:
        for frame in ImageSequence.Iterator(img):
            rgb = frame.convert("RGB")
            frames.append(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
    return frames


def coverage_warnings(info: dict, targets: CorpusTargets) -> list[str]:
    """Warn about any statement-corpus target-mix requirement that is not met."""
    warnings: list[str] = []
    pages = info.get("statement_pages_by_doc_type", {})
    digital = pages.get("digital", 0) - info.get("statement_text_layer_unusable", 0)
    scanned_photo = pages.get("scanned", 0) + pages.get("photo", 0)
    banks = info.get("statement_banks", [])
    multipage = info.get("statement_multipage_files", 0)

    if digital < targets.min_digital_pages:
        warnings.append(
            f"only {digital} digital pages with a usable text layer "
            f"(< {targets.min_digital_pages}); Check A is weakened"
        )
    if scanned_photo < targets.min_scanned_photo_pages:
        warnings.append(
            f"only {scanned_photo} scanned/photo pages (< {targets.min_scanned_photo_pages})"
        )
    if len(banks) < targets.min_banks:
        warnings.append(f"only {len(banks)} banks (< {targets.min_banks})")
    if multipage < targets.min_multipage_files:
        warnings.append(f"only {multipage} multi-page statements (< {targets.min_multipage_files})")
    return warnings


def prepare_statements(cfg: BankStmtConfig) -> tuple[list[PreparedSample], dict]:
    """Ingest every statement file under `cfg.input_dir` into per-page samples."""
    input_dir = Path(cfg.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"bank statement input_dir does not exist: {input_dir}")

    files = discover(input_dir)
    bank_counters: dict[str, int] = defaultdict(int)
    pages_by_doc_type: dict[str, int] = defaultdict(int)
    banks_seen: set[str] = set()
    multipage_files = 0
    text_layer_unusable = 0
    text_layer_repaired = 0
    samples: list[PreparedSample] = []

    # Resolve every file's bank before any rasterization so a mapping gap fails
    # in seconds, naming all offending files, rather than minutes into the run.
    resolved: list[tuple[Path, str]] = []
    unknown: list[str] = []
    for path in files:
        rel = path.relative_to(input_dir).as_posix()
        try:
            resolved.append((path, bank_for(rel, cfg)))
        except UnknownBankError as exc:
            unknown.append(str(exc))
    if unknown:
        raise UnknownBankError(
            f"{len(unknown)} file(s) with no configured bank "
            f"(add a pattern to bank_map and the bank to banks):\n  " + "\n  ".join(unknown)
        )

    # Removed after the loop; its finalizer also removes it if ingestion raises.
    repair_dir = tempfile.TemporaryDirectory()
    for path, bank in resolved:
        bank_counters[bank] += 1
        idx = bank_counters[bank]
        banks_seen.add(bank)

        if path.suffix.lower() == ".pdf":
            doc_type = route(path)
            n_pages = _page_count(path)
            if n_pages > 1:
                multipage_files += 1
            # Text comes from a copy with broken ToUnicode maps rebuilt (KTB); the image
            # is always rasterized from the original.
            text_source = path
            if doc_type == "digital":
                repaired_pdf = Path(repair_dir.name) / f"{bank}-{idx:04d}.pdf"
                if repair_tounicode(path, repaired_pdf):
                    text_source = repaired_pdf
                    text_layer_repaired += n_pages
            for page_no in range(1, n_pages + 1):
                image = rasterize_page(path, page_no, cfg.dpi)
                gt: TextLayerGT | NoGT = NoGT(gt_kind="none")
                if doc_type == "digital":
                    layer = text_layer(text_source, page_no, cfg.dpi)
                    if text_layer_usable(layer.gt_text):
                        gt = layer
                    else:
                        text_layer_unusable += 1
                pages_by_doc_type[doc_type] += 1
                samples.append(
                    PreparedSample(
                        sample_id=f"BS-{bank}-{idx:04d}-p{page_no}",
                        source=Source.bankstmt,
                        task=Task.statement,
                        subtask=None,
                        domain="Finance",
                        bank=bank,
                        doc_type=doc_type,
                        page_no=page_no,
                        n_pages=n_pages,
                        question="",
                        image=image,
                        gt=gt,
                        critical_fields=list(STATEMENT_CRITICAL_FIELDS),
                    )
                )
        else:
            doc_type = "photo"
            frames = _load_image_frames(path)
            n_pages = len(frames)
            if n_pages > 1:
                multipage_files += 1
            for page_no, frame in enumerate(frames, start=1):
                pages_by_doc_type[doc_type] += 1
                samples.append(
                    PreparedSample(
                        sample_id=f"BS-{bank}-{idx:04d}-p{page_no}",
                        source=Source.bankstmt,
                        task=Task.statement,
                        subtask=None,
                        domain="Finance",
                        bank=bank,
                        doc_type=doc_type,
                        page_no=page_no,
                        n_pages=n_pages,
                        question="",
                        image=frame,
                        gt=NoGT(gt_kind="none"),
                        critical_fields=list(STATEMENT_CRITICAL_FIELDS),
                    )
                )

    repair_dir.cleanup()

    info = {
        "statement_pages_by_doc_type": dict(pages_by_doc_type),
        "statement_banks": sorted(banks_seen),
        "statement_files": len(files),
        "statement_multipage_files": multipage_files,
        "statement_text_layer_unusable": text_layer_unusable,
        "statement_text_layer_repaired": text_layer_repaired,
    }
    return samples, info
