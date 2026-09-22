"""Duplicate-statement detection: the test set and its evaluation (task 5.7).

`build_dupset` turns a scored run's statement pages into a *separate* run that
`ocrbench infer` can process unchanged: up to 20 originals (page 1 only, chosen
deterministically across banks), three variants each (`scan_low`, 180° rotation then
`photo`, and a 5% margin crop) and one same-bank hard negative per original.

`evaluate` then combines three signals per pair — the exact SHA-256 of the image bytes,
the Jaccard similarity of the extracted (date, amount) row sets, and a perceptual hash
of the page — and reports recall on the variants and the false-positive rate on the
negatives for each (model, Jaccard threshold).
"""

import hashlib
import random
from collections import defaultdict
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ocr_bench.data.degrade import encode_condition
from ocr_bench.jsonl import read_rows, write_rows_atomic
from ocr_bench.metrics.statement_gt import row_key
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import Condition, GtKind, ManifestRow, Source, StatementPage, Task

THRESHOLDS: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
TARGET_ORIGINALS = 20
CROP_MARGIN = 0.05
VARIANTS = ("scan_low", "rot180_photo", "crop")

VariantName = Literal["scan_low", "rot180_photo", "crop"]


class DupPair(BaseModel):
    """One evaluated pair: a variant of an original, or a hard negative."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    original: str
    candidate: str
    kind: Literal["variant", "negative"]
    variant: str | None = None
    negative_kind: Literal["same_account", "same_bank"] | None = None

    @property
    def is_duplicate(self) -> bool:
        return self.kind == "variant"


class DupSet(BaseModel):
    """`dupset.json`: what was built, and what each pair is."""

    model_config = ConfigDict(extra="forbid")

    src_run_id: str
    dst_run_id: str
    seed: int = 42
    n_originals: int = 0
    n_negatives: int = 0
    pairs: list[DupPair] = Field(default_factory=list)
    #: SHA-256 of every stored image, keyed by sample id (the exact-duplicate signal).
    sha256: dict[str, str] = Field(default_factory=dict)
    banks: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PairSignals(BaseModel):
    """One pair's signals under one model and one threshold."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    model: str
    threshold: float
    kind: str
    variant: str | None = None
    negative_kind: str | None = None
    is_duplicate: bool
    sha_match: bool
    jaccard: float
    phash_distance: int | None = None
    detected: bool


class DuplicateEvalRow(BaseModel):
    """One `(model, threshold)` row of `duplicates.jsonl`."""

    model_config = ConfigDict(extra="forbid")

    model: str
    threshold: float
    n_variants: int
    n_negatives: int
    recall: float | None
    fpr: float | None
    sha_recall: float | None = None
    mean_jaccard_variants: float | None = None
    mean_jaccard_negatives: float | None = None
    mean_phash_variants: float | None = None
    mean_phash_negatives: float | None = None


# --- signals -------------------------------------------------------------------------------


def row_set(page: StatementPage | None) -> set[tuple[str | None, str | None]]:
    """A page's (date, amount) row tuples — the fingerprint the Jaccard runs over."""
    return {row_key(row) for row in (page.rows if page else [])}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity; two empty sets score 0 (no evidence, not a match)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def phash_distance(a: Any, b: Any) -> int | None:
    """Hamming distance between two perceptual hashes (`imagehash.ImageHash` subtraction)."""
    if a is None or b is None:
        return None
    return int(abs(a - b))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate(
    dupset: DupSet,
    pages: dict[tuple[str, str], StatementPage | None],
    phashes: dict[str, Any],
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> tuple[list[DuplicateEvalRow], list[PairSignals]]:
    """Sweep the Jaccard threshold for every model over every pair.

    [IMPLEMENTER DECIDES] the detector fires on an exact SHA-256 match or on
    `jaccard >= threshold`. The perceptual hash is recorded next to them but does not
    vote: no labelled corpus exists here to calibrate a distance cut-off, and a 180°
    rotation moves it far while the pair is still a duplicate. M6 can read the recorded
    distances to choose a cut-off later.
    """
    models = sorted({model for model, _ in pages})
    rows: list[DuplicateEvalRow] = []
    details: list[PairSignals] = []
    for model in models:
        for threshold in thresholds:
            signals = []
            for pair in dupset.pairs:
                sha_a = dupset.sha256.get(pair.original)
                sha_b = dupset.sha256.get(pair.candidate)
                sha_match = sha_a is not None and sha_a == sha_b
                similarity = jaccard(
                    row_set(pages.get((model, pair.original))),
                    row_set(pages.get((model, pair.candidate))),
                )
                signal = PairSignals(
                    pair_id=pair.pair_id,
                    model=model,
                    threshold=threshold,
                    kind=pair.kind,
                    variant=pair.variant,
                    negative_kind=pair.negative_kind,
                    is_duplicate=pair.is_duplicate,
                    sha_match=sha_match,
                    jaccard=similarity,
                    phash_distance=phash_distance(
                        phashes.get(pair.original), phashes.get(pair.candidate)
                    ),
                    detected=sha_match or similarity >= threshold,
                )
                signals.append(signal)
            details.extend(signals)
            variants = [s for s in signals if s.is_duplicate]
            negatives = [s for s in signals if not s.is_duplicate]
            rows.append(
                DuplicateEvalRow(
                    model=model,
                    threshold=threshold,
                    n_variants=len(variants),
                    n_negatives=len(negatives),
                    recall=_mean([float(s.detected) for s in variants]),
                    fpr=_mean([float(s.detected) for s in negatives]),
                    sha_recall=_mean([float(s.sha_match) for s in variants]),
                    mean_jaccard_variants=_mean([s.jaccard for s in variants]),
                    mean_jaccard_negatives=_mean([s.jaccard for s in negatives]),
                    mean_phash_variants=_mean(
                        [float(s.phash_distance) for s in variants if s.phash_distance is not None]
                    ),
                    mean_phash_negatives=_mean(
                        [float(s.phash_distance) for s in negatives if s.phash_distance is not None]
                    ),
                )
            )
    return rows, details


# --- building the set ------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _crop(image: np.ndarray, margin: float = CROP_MARGIN) -> np.ndarray:
    h, w = image.shape[:2]
    dy, dx = int(round(h * margin)), int(round(w * margin))
    return image[dy : h - dy, dx : w - dx]


def variant_bytes(image: np.ndarray, sample_id: str, variant: str) -> tuple[str, bytes]:
    """`(extension, bytes)` of one variant of a page image."""
    if variant == "scan_low":
        ext, data, _ = encode_condition(image, sample_id, Condition.scan_low)
        return ext, data
    if variant == "rot180_photo":
        ext, data, _ = encode_condition(
            cv2.rotate(image, cv2.ROTATE_180), sample_id, Condition.photo
        )
        return ext, data
    if variant == "crop":
        ext, data, _ = encode_condition(_crop(image), sample_id, Condition.clean)
        return ext, data
    raise ValueError(f"unknown variant: {variant!r}")


def select_originals(rows: list[ManifestRow], seed: int, n_originals: int) -> list[ManifestRow]:
    """Page-1 statement rows, drawn round-robin across banks so every bank is represented.

    Each bank's files are shuffled with the run seed, then taken one per bank in turn,
    which keeps the selection deterministic and bank-balanced.
    """
    by_bank: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        by_bank[row.bank or ""].append(row)
    rng = random.Random(seed)
    for bank in sorted(by_bank):
        by_bank[bank].sort(key=lambda r: r.sample_id)
        rng.shuffle(by_bank[bank])

    chosen: list[ManifestRow] = []
    banks = sorted(by_bank)
    index = 0
    while len(chosen) < n_originals and any(index < len(by_bank[b]) for b in banks):
        for bank in banks:
            if index < len(by_bank[bank]) and len(chosen) < n_originals:
                chosen.append(by_bank[bank][index])
        index += 1
    return chosen


def build_dupset(
    src: RunPaths,
    dst: RunPaths,
    *,
    seed: int = 42,
    n_originals: int = TARGET_ORIGINALS,
) -> DupSet:
    """Build the duplicate run `dst` from the statement pages of run `src`."""
    from ocr_bench.metrics.statement_run import file_id_for

    manifest = [
        row
        for row in read_rows(src.manifest, ManifestRow)
        if row.task == Task.statement and row.condition == Condition.clean and row.page_no == 1
    ]
    originals = select_originals(manifest, seed, n_originals)
    dupset = DupSet(
        src_run_id=src.root.name,
        dst_run_id=dst.root.name,
        seed=seed,
        n_originals=len(originals),
    )
    dst.images.mkdir(parents=True, exist_ok=True)
    out_rows: list[ManifestRow] = []

    def _store(sample_id: str, ext: str, data: bytes, row: ManifestRow) -> None:
        rel = f"img/{sample_id}{ext}"
        (dst.root / rel).write_bytes(data)
        dupset.sha256[sample_id] = _sha256(data)
        dupset.banks[sample_id] = row.bank or ""
        out_rows.append(
            ManifestRow(
                sample_id=sample_id,
                source=Source.bankstmt,
                task=Task.statement,
                domain=row.domain,
                bank=row.bank,
                doc_type=row.doc_type,
                page_no=1,
                n_pages=1,
                condition=Condition.clean,
                image_path=rel,
                gt_kind=GtKind.none,
                gt_path=None,
            )
        )

    ids: dict[str, str] = {}
    for row in originals:
        source_path = src.root / row.image_path
        data = source_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        base = f"DUP-{file_id_for(row.sample_id)}"
        ids[row.sample_id] = base
        _store(base, source_path.suffix, data, row)
        for variant in VARIANTS:
            ext, variant_data = variant_bytes(image, f"{base}-{variant}", variant)
            candidate = f"{base}-{variant}"
            _store(candidate, ext, variant_data, row)
            dupset.pairs.append(
                DupPair(
                    pair_id=f"{base}:{variant}",
                    original=base,
                    candidate=candidate,
                    kind="variant",
                    variant=variant,
                )
            )

    # Hard negatives: a different file of the same bank. [IMPLEMENTER DECIDES] the corpus
    # gives no account number before extraction (file names carry none), so negatives are
    # same-bank pairs and `negative_kind` records that; a same-account pair would be
    # recorded as `same_account` if one could be identified.
    by_bank: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in originals:
        by_bank[row.bank or ""].append(row)
    for bank in sorted(by_bank):
        rows = by_bank[bank]
        if len(rows) < 2:
            dupset.warnings.append(f"bank {bank!r} has only {len(rows)} file(s): no hard negative")
            continue
        for index, row in enumerate(rows):
            other = rows[(index + 1) % len(rows)]
            dupset.pairs.append(
                DupPair(
                    pair_id=f"{ids[row.sample_id]}:neg:{ids[other.sample_id]}",
                    original=ids[row.sample_id],
                    candidate=ids[other.sample_id],
                    kind="negative",
                    negative_kind="same_bank",
                )
            )
    dupset.n_negatives = sum(p.kind == "negative" for p in dupset.pairs)

    if dupset.n_originals < TARGET_ORIGINALS:
        dupset.warnings.append(
            f"only {dupset.n_originals} original statements available (target {TARGET_ORIGINALS})"
        )
    if dupset.n_negatives < TARGET_ORIGINALS:
        dupset.warnings.append(
            f"only {dupset.n_negatives} hard negatives available (target {TARGET_ORIGINALS})"
        )

    out_rows.sort(key=lambda r: r.sample_id)
    write_rows_atomic(dst.manifest, out_rows)
    dst.dupset.write_text(
        dupset.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return dupset


# --- evaluating the set ----------------------------------------------------------------------


def page_phash(path) -> Any:
    """The perceptual hash of a stored page image (`imagehash.phash`)."""
    import imagehash
    from PIL import Image

    with Image.open(path) as img:
        return imagehash.phash(img.convert("RGB"))


def eval_dupset(
    dst: RunPaths,
    thresholds: tuple[float, ...] = THRESHOLDS,
    overrides: dict | None = None,
) -> tuple[list[DuplicateEvalRow], list[PairSignals]]:
    """Score the duplicate run's predictions and write `duplicates.jsonl`."""
    from ocr_bench.jsonl import latest_predictions
    from ocr_bench.normalize.statement import map_statement_page

    dupset = DupSet.model_validate_json(dst.dupset.read_text(encoding="utf-8"))
    manifest = {row.sample_id: row for row in read_rows(dst.manifest, ManifestRow)}
    pages: dict[tuple[str, str], StatementPage | None] = {}
    for (sample_id, _, model), pred in latest_predictions(dst.predictions).items():
        row = manifest.get(sample_id)
        if row is None:
            continue
        pages[(model, sample_id)] = map_statement_page(
            pred.normalized,
            bank=row.bank,
            page_no=1,
            overrides=(overrides or {}).get(row.bank or ""),
        )
    phashes = {
        sample_id: page_phash(dst.root / manifest[sample_id].image_path)
        for sample_id in sorted(dupset.sha256)
        if sample_id in manifest
    }
    rows, details = evaluate(dupset, pages, phashes, thresholds)
    write_rows_atomic(dst.duplicates, rows)
    write_rows_atomic(dst.duplicate_pairs, details)
    return rows, details
