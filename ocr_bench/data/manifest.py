"""The shared prepared-sample type and manifest materialization (`ocrbench prepare`)."""

from dataclasses import dataclass

import numpy as np

from ocr_bench.data.degrade import encode_condition, transform_bbox
from ocr_bench.paths import RunPaths
from ocr_bench.schemas import (
    Condition,
    GroundTruth,
    GtKind,
    ManifestRow,
    Source,
    Task,
    TextLayerGT,
    Word,
)


@dataclass
class PreparedSample:
    sample_id: str
    source: Source
    task: Task
    subtask: str | None
    domain: str | None
    bank: str | None
    doc_type: str | None
    page_no: int
    n_pages: int
    question: str
    image: np.ndarray  # BGR uint8, clean
    gt: GroundTruth  # the model instance
    critical_fields: list[str]


_CONDITION_ORDER = [Condition.clean, Condition.scan_low, Condition.photo]
_CONDITION_RANK = {c: i for i, c in enumerate(_CONDITION_ORDER)}


def materialize(
    samples: list[PreparedSample], conditions: list[Condition], paths: RunPaths
) -> list[ManifestRow]:
    """Write every sample's ground truth and degraded images, returning manifest rows."""
    paths.images.mkdir(parents=True, exist_ok=True)
    paths.gt.mkdir(parents=True, exist_ok=True)

    ordered_conditions = [c for c in _CONDITION_ORDER if c in conditions]

    rows: list[ManifestRow] = []
    for sample in samples:
        gt_kind = GtKind(sample.gt.gt_kind)
        gt_rel_path: str | None = None
        if gt_kind != GtKind.none:
            gt_rel_path = f"gt/{sample.sample_id}.json"
            (paths.root / gt_rel_path).write_text(sample.gt.model_dump_json(), encoding="utf-8")

        h_img, w_img = sample.image.shape[:2]

        for condition in ordered_conditions:
            ext, data, hmat = encode_condition(sample.image, sample.sample_id, condition)
            suffix = "" if condition == Condition.clean else f"_{condition.value}"
            img_rel = f"img/{sample.sample_id}{suffix}{ext}"
            (paths.root / img_rel).write_bytes(data)

            row_gt_path = gt_rel_path
            if (
                condition == Condition.photo
                and hmat is not None
                and isinstance(sample.gt, TextLayerGT)
            ):
                transformed_words = [
                    Word(
                        text=word.text,
                        bbox=transform_bbox(word.bbox, hmat, w_img, h_img),
                    )
                    for word in sample.gt.gt_words
                ]
                sidecar = sample.gt.model_copy(
                    update={"gt_words": transformed_words, "homography": hmat.tolist()}
                )
                sidecar_rel = f"gt/{sample.sample_id}_photo.json"
                (paths.root / sidecar_rel).write_text(sidecar.model_dump_json(), encoding="utf-8")
                row_gt_path = sidecar_rel

            rows.append(
                ManifestRow(
                    sample_id=sample.sample_id,
                    source=sample.source,
                    task=sample.task,
                    subtask=sample.subtask,
                    domain=sample.domain,
                    bank=sample.bank,
                    doc_type=sample.doc_type,
                    page_no=sample.page_no,
                    n_pages=sample.n_pages,
                    condition=condition,
                    image_path=img_rel,
                    gt_kind=gt_kind,
                    gt_path=row_gt_path,
                    critical_fields=sample.critical_fields,
                    question=sample.question,
                )
            )

    rows.sort(key=lambda r: (r.source.value, r.sample_id, _CONDITION_RANK[r.condition]))
    return rows
