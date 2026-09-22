"""Tests for ocr_bench.models.imaging (crop / rotate / resize helpers) on synthetic images."""

import numpy as np
from PIL import Image

from ocr_bench.models.imaging import (
    crop_block,
    crop_region,
    downscale_if_huge,
    region_to_pixels,
    resize_by_need,
    resize_max_side,
    rotate_crop,
    smart_resize,
)


def test_smart_resize_matches_qwen2vl():
    assert smart_resize(1000, 2000, max_pixels=4_000_000) == (1008, 1988)
    h, w = smart_resize(4000, 3000, max_pixels=4_000_000)
    assert h % 28 == 0 and w % 28 == 0 and h * w <= 4_000_000
    assert smart_resize(10, 10, max_pixels=4_000_000) == (56, 56)  # min_pixels 3136


def test_resize_max_side_never_upscales():
    assert resize_max_side(Image.new("RGB", (3600, 1800)), 1800).size == (1800, 900)
    assert resize_max_side(Image.new("RGB", (100, 50)), 1800).size == (100, 50)


def test_region_to_pixels_and_crop():
    assert region_to_pixels((274, 385, 573, 501), (2000, 1000)) == (548, 385, 1146, 501)
    crop = crop_region(Image.new("RGB", (2000, 1000)), (274, 385, 573, 501))
    assert crop.size == (598, 116)


def test_region_to_pixels_never_empty():
    x1, y1, x2, y2 = region_to_pixels((500, 500, 500, 500), (100, 100))
    assert x2 > x1 and y2 > y1


def test_crop_block_rectangle():
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    crop = crop_block(img, [(100, 100), (600, 500)])
    assert crop.size == (100, 40)


def test_crop_block_polygon_masks_outside_pixels():
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    # triangle (0,0) (1000,0) (0,1000): the bottom-right corner is outside it
    crop = crop_block(img, [(0, 0), (1000, 0), (0, 1000)])
    arr = np.array(crop)
    assert crop.size[0] >= 99 and crop.size[1] >= 99
    assert tuple(arr[2, 2]) == (255, 255, 255)
    assert tuple(arr[-2, -2]) == (0, 0, 0)


def test_rotate_crop_expands():
    img = Image.new("RGB", (40, 10))
    assert rotate_crop(img, 90).size == (10, 40)
    assert rotate_crop(img, 180).size == (40, 10)
    assert rotate_crop(img, None).size == (40, 10)
    assert rotate_crop(img, 0).size == (40, 10)


def test_resize_by_need_pads_extreme_ratio_with_white():
    img = Image.new("RGB", (2000, 10), (0, 0, 0))
    out = resize_by_need(img, min_edge=28, max_ratio=50)
    assert out.size == (2000, 40)
    arr = np.array(out)
    assert tuple(arr[0, 0]) == (255, 255, 255)  # padding
    assert tuple(arr[20, 1000]) == (0, 0, 0)  # original content, centred


def test_resize_by_need_upscales_small_edge():
    out = resize_by_need(Image.new("RGB", (10, 20)), min_edge=28, max_ratio=50)
    assert out.size == (28, 56)


def test_resize_by_need_noop_for_normal_image():
    assert resize_by_need(Image.new("RGB", (300, 200))).size == (300, 200)


def test_downscale_if_huge():
    assert downscale_if_huge(Image.new("RGB", (400, 100)), max_pixels=10_000).size == (200, 50)
    assert downscale_if_huge(Image.new("RGB", (40, 10)), max_pixels=10_000).size == (40, 10)
