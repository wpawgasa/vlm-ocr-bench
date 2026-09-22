"""Image pre-processing for request plans: model pre-resize, region crops and the TeleOCR
block pipeline (polygon crop, rotation, `resize_by_need`), following each model's
official reference code."""

import math

import cv2
import numpy as np
from PIL import Image

Region = tuple[int, int, int, int]


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 11_289_600,
) -> tuple[int, int]:
    """Qwen2-VL `smart_resize`: (height, width) rounded to multiples of `factor` and kept
    within [min_pixels, max_pixels]. Returns (height, width)."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """Shrink so the longest side is <= `max_side`, keeping aspect; never upscale."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)


def region_to_pixels(region: Region, size: tuple[int, int]) -> Region:
    """Map a 0-1000 region to integer pixels of an image of `size` (w, h), non-empty."""
    w, h = size
    x1, y1, x2, y2 = region
    px1 = min(max(round(min(x1, x2) * w / 1000), 0), w - 1)
    py1 = min(max(round(min(y1, y2) * h / 1000), 0), h - 1)
    px2 = min(max(round(max(x1, x2) * w / 1000), px1 + 1), w)
    py2 = min(max(round(max(y1, y2) * h / 1000), py1 + 1), h)
    return px1, py1, px2, py2


def crop_region(image: Image.Image, region: Region) -> Image.Image:
    return image.crop(region_to_pixels(region, image.size))


def crop_block(image: Image.Image, points_1000: list[tuple[int, int]]) -> Image.Image:
    """Crop a TeleOCR layout block from the original RGB image.

    Two points are a rectangle. More points are a polygon: crop its `cv2.boundingRect`
    and zero out the pixels outside the polygon.
    """
    w, h = image.size
    pts = [(p[0] * w / 1000, p[1] * h / 1000) for p in points_1000]
    if len(pts) == 2:
        (ax, ay), (bx, by) = points_1000
        return crop_region(image, (ax, ay, bx, by))
    poly = np.array([[round(x), round(y)] for x, y in pts], dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(poly)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + bw, w), min(y + bh, h)
    if x1 <= x0 or y1 <= y0:
        return image.crop((0, 0, 1, 1))
    arr = np.array(image.convert("RGB"))[y0:y1, x0:x1].copy()
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly - np.array([x0, y0], dtype=np.int32)], 255)
    arr[mask == 0] = 0
    return Image.fromarray(arr)


def rotate_crop(image: Image.Image, angle: int | None) -> Image.Image:
    """Rotate a block crop by its layout angle, as the reference client does."""
    if angle in (90, 180, 270):
        return image.rotate(angle, expand=True)
    return image


def resize_by_need(image: Image.Image, min_edge: int = 28, max_ratio: float = 50) -> Image.Image:
    """TeleOCR `resize_by_need`: pad centred with white to aspect ratio `max_ratio` when the
    image is more elongated, then upscale (bicubic) so the short edge is >= `min_edge`."""
    w, h = image.size
    if min(w, h) > 0 and max(w, h) / min(w, h) > max_ratio:
        new_w, new_h = w, h
        if w > h:
            new_h = math.ceil(w / max_ratio)
        else:
            new_w = math.ceil(h / max_ratio)
        canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        canvas.paste(image.convert("RGB"), ((new_w - w) // 2, (new_h - h) // 2))
        image = canvas
        w, h = image.size
    if min(w, h) < min_edge:
        scale = min_edge / min(w, h)
        image = image.resize((math.ceil(w * scale), math.ceil(h * scale)), Image.BICUBIC)
    return image


def downscale_if_huge(image: Image.Image, max_pixels: int = 64_000_000) -> Image.Image:
    """Downscale (bilinear) an image larger than `max_pixels` before cropping blocks."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
