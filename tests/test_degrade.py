"""Tests for ocr_bench.data.degrade: deterministic image degradation."""

import cv2
import numpy as np
import pytest

from ocr_bench.data.degrade import degrade, encode_condition, seed_for, transform_bbox
from ocr_bench.schemas import Condition


def _checkerboard(h: int = 64, w: int = 96) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if ((x // 8) + (y // 8)) % 2 == 0:
                img[y : y + 8, x : x + 8] = (30, 90, 200)
            else:
                img[y : y + 8, x : x + 8] = (210, 180, 40)
    return img


def test_seed_for_is_deterministic():
    a = seed_for("s1", "photo")
    b = seed_for("s1", "photo")
    assert a == b
    assert isinstance(a, int)


def test_seed_for_varies_by_sample_and_condition():
    assert seed_for("s1", "photo") != seed_for("s2", "photo")
    assert seed_for("s1", "photo") != seed_for("s1", "scan_low")


def test_clean_equals_input():
    img = _checkerboard()
    out, h = degrade(img, "s1", Condition.clean)
    assert np.array_equal(out, img)
    assert h is None
    # degrade must not mutate or alias the input
    out[0, 0] = (1, 2, 3)
    assert not np.array_equal(out, img) or img[0, 0].tolist() != [1, 2, 3]


def test_clean_is_deterministic():
    img = _checkerboard()
    out1, _ = degrade(img, "s1", Condition.clean)
    out2, _ = degrade(img, "s1", Condition.clean)
    assert np.array_equal(out1, out2)


def test_scan_low_differs_and_keeps_shape():
    img = _checkerboard()
    out, h = degrade(img, "s1", Condition.scan_low)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)
    assert h is None


def test_scan_low_is_deterministic():
    img = _checkerboard()
    out1, _ = degrade(img, "s1", Condition.scan_low)
    out2, _ = degrade(img, "s1", Condition.scan_low)
    assert np.array_equal(out1, out2)


def test_photo_differs_and_keeps_shape_and_returns_homography():
    img = _checkerboard()
    out, hmat = degrade(img, "s1", Condition.photo)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)
    assert hmat is not None
    assert hmat.shape == (3, 3)


def test_photo_is_deterministic():
    img = _checkerboard()
    out1, h1 = degrade(img, "s1", Condition.photo)
    out2, h2 = degrade(img, "s1", Condition.photo)
    assert np.array_equal(out1, out2)
    assert np.array_equal(h1, h2)


def test_photo_differs_by_sample_id():
    img = _checkerboard()
    out1, h1 = degrade(img, "s1", Condition.photo)
    out2, h2 = degrade(img, "s2", Condition.photo)
    assert not np.array_equal(out1, out2)
    assert not np.array_equal(h1, h2)


def test_transform_bbox_identity():
    identity = np.eye(3, dtype=np.float64)
    bbox = (10.0, 20.0, 30.0, 40.0)
    result = transform_bbox(bbox, identity, w=100, h=100)
    assert result == bbox


def test_transform_bbox_translation():
    translate = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    bbox = (10.0, 20.0, 30.0, 40.0)
    result = transform_bbox(bbox, translate, w=1000, h=1000)
    assert result == (15.0, 27.0, 35.0, 47.0)


def test_transform_bbox_clips_to_image():
    translate = np.array([[1.0, 0.0, -5.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    bbox = (0.0, 0.0, 10.0, 10.0)
    result = transform_bbox(bbox, translate, w=100, h=100)
    assert result == (0.0, 0.0, 5.0, 5.0)


def test_transform_bbox_matches_manual_homography_math():
    img = _checkerboard()
    _, hmat = degrade(img, "s1", Condition.photo)
    h, w = img.shape[:2]
    bbox = (10.0, 5.0, 40.0, 30.0)
    x0, y0, x1, y1 = bbox
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    ones = np.ones((4, 1), dtype=np.float64)
    homog = np.hstack([corners, ones])
    mapped = homog @ hmat.T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    exp_x0 = max(0.0, float(mapped[:, 0].min()))
    exp_y0 = max(0.0, float(mapped[:, 1].min()))
    exp_x1 = min(float(w), float(mapped[:, 0].max()))
    exp_y1 = min(float(h), float(mapped[:, 1].max()))

    result = transform_bbox(bbox, hmat, w=w, h=h)
    assert result == pytest.approx((exp_x0, exp_y0, exp_x1, exp_y1))


@pytest.mark.parametrize(
    ("condition", "ext"),
    [(Condition.clean, ".png"), (Condition.scan_low, ".jpg"), (Condition.photo, ".jpg")],
)
def test_encode_condition_decodes_to_degrade_output(condition, ext):
    # The stored bytes are exactly what the model will see: decoding them must give
    # the same pixels as degrade(), so storing JPEG loses nothing.
    img = _checkerboard(120, 160)
    got_ext, data, hmat = encode_condition(img, "s1", condition)
    assert got_ext == ext
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    out, h2 = degrade(img, "s1", condition)
    assert np.array_equal(decoded, out)
    assert (hmat is None) == (h2 is None)
    if hmat is not None:
        assert np.array_equal(hmat, h2)


def test_encode_condition_is_byte_deterministic():
    img = _checkerboard(120, 160)
    assert (
        encode_condition(img, "s1", Condition.photo)[1]
        == encode_condition(img, "s1", Condition.photo)[1]
    )
