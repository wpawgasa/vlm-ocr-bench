"""Deterministic image-quality degradation for the `clean`/`scan_low`/`photo` conditions.

Every random parameter is derived from a seed keyed on `(sample_id, condition)`,
so re-running `prepare` reproduces pixel-identical degraded images.
"""

import hashlib

import cv2
import numpy as np

from ocr_bench.schemas import BBox, Condition


def seed_for(sample_id: str, condition: str) -> int:
    digest = hashlib.sha256(f"{sample_id}|{condition}".encode()).hexdigest()
    return int(digest[:8], 16)


_SCAN_LOW_JPEG_QUALITY = 40
_PHOTO_JPEG_QUALITY = 55


def _encode(img: np.ndarray, ext: str, params: list[int]) -> bytes:
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"{ext} encode failed")
    return buf.tobytes()


def _scan_low_pre_jpeg(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    small_w = max(1, round(w * 0.75))
    small_h = max(1, round(h * 0.75))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(back, (0, 0), sigmaX=0.6)


def _photo_pre_jpeg(img: np.ndarray, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed_for(sample_id, "photo"))
    angle = rng.uniform(-3, 3)
    jit = rng.uniform(-0.02, 0.02, size=(4, 2))
    bright = 1 + rng.uniform(-0.15, 0.15)

    rot2x3 = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rmat = np.vstack([rot2x3, [0.0, 0.0, 1.0]])

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = (src + jit * np.array([w, h])).astype(np.float32)
    pmat = cv2.getPerspectiveTransform(src, dst)
    hmat = pmat @ rmat

    out = cv2.warpPerspective(
        img,
        hmat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    out = np.clip(out.astype(np.float32) * bright, 0, 255).astype(np.uint8)
    return out, hmat


def encode_condition(
    img: np.ndarray, sample_id: str, condition: Condition
) -> tuple[str, bytes, np.ndarray | None]:
    """Render `img` (BGR uint8) under `condition` and return `(ext, bytes, homography)`.

    `clean` is lossless PNG; `scan_low`/`photo` end in a JPEG step, so their stored
    bytes *are* that JPEG — no lossless re-encode of already-lossy pixels. The
    homography is the 3x3 matrix applied for `photo`, else None.
    """
    if condition == Condition.clean:
        return ".png", _encode(img, ".png", []), None
    if condition == Condition.scan_low:
        pre = _scan_low_pre_jpeg(img)
        return (
            ".jpg",
            _encode(pre, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, _SCAN_LOW_JPEG_QUALITY]),
            None,
        )
    if condition == Condition.photo:
        pre, hmat = _photo_pre_jpeg(img, sample_id)
        return ".jpg", _encode(pre, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, _PHOTO_JPEG_QUALITY]), hmat
    raise ValueError(f"unknown condition: {condition!r}")


def degrade(
    img: np.ndarray, sample_id: str, condition: Condition
) -> tuple[np.ndarray, np.ndarray | None]:
    """Degrade `img` (BGR uint8) under `condition`, returning the decoded pixels and,
    for `photo` only, the 3x3 homography that was applied (else None)."""
    if condition == Condition.clean:
        return img.copy(), None
    _, data, hmat = encode_condition(img, sample_id, condition)
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), hmat


def transform_bbox(bbox: BBox, hmat: np.ndarray, w: int, h: int) -> BBox:
    """Map `bbox`'s 4 corners through `hmat`, returning the clipped axis-aligned box."""
    x0, y0, x1, y1 = bbox
    corners = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(corners, hmat).reshape(4, 2)
    nx0 = float(mapped[:, 0].min())
    ny0 = float(mapped[:, 1].min())
    nx1 = float(mapped[:, 0].max())
    ny1 = float(mapped[:, 1].max())
    nx0 = min(max(nx0, 0.0), w)
    ny0 = min(max(ny0, 0.0), h)
    nx1 = min(max(nx1, 0.0), w)
    ny1 = min(max(ny1, 0.0), h)
    return (nx0, ny0, nx1, ny1)
