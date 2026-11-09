"""Synthetic part images. Texture is generated, not photographed on a line."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def _value_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 4) -> NDArray[np.float64]:
    acc = np.zeros((h, w), dtype=np.float64)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        gh, gw = max(2, h // (2 ** (octaves - o))), max(2, w // (2 ** (octaves - o)))
        grid = rng.random((gh, gw))
        sampled = np.array(
            Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
            dtype=np.float64,
        ) / 255.0
        acc += amp * sampled
        total += amp
        amp *= 0.55
    return acc / total


def _circle_mask(h: int, w: int) -> NDArray[np.bool_]:
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = min(h, w) * 0.42
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def _segment_distance(h: int, w: int, y1: float, x1: float, y2: float, x2: float) -> NDArray[np.float64]:
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    vx, vy = x2 - x1, y2 - y1
    length = vx * vx + vy * vy + 1e-9
    t = np.clip(((xx - x1) * vx + (yy - y1) * vy) / length, 0.0, 1.0)
    px = x1 + t * vx
    py = y1 + t * vy
    return np.sqrt((xx - px) ** 2 + (yy - py) ** 2)


def render_part(defect_type: str | None, rng: np.random.Generator, size: int = 128) -> NDArray[np.uint8]:
    """Render a circular die. `defect_type=None` is a good part."""
    h = w = size
    mask = _circle_mask(h, w)
    noise = _value_noise(h, w, rng)
    base = 0.62 + 0.12 * (noise - 0.5)
    img = np.stack([base * 0.92, base * 0.95, base], axis=-1)
    img[~mask] = 0.06

    if defect_type == "scratch":
        for _ in range(int(rng.integers(1, 4))):
            y1, x1 = rng.uniform(h * 0.25, h * 0.75), rng.uniform(w * 0.2, w * 0.8)
            ang = rng.uniform(0, np.pi)
            length = rng.uniform(size * 0.25, size * 0.55)
            y2, x2 = y1 + length * np.sin(ang), x1 + length * np.cos(ang)
            dist = _segment_distance(h, w, y1, x1, y2, x2)
            width = rng.uniform(0.7, 1.6)
            scratch = np.clip(1.0 - dist / width, 0.0, 1.0)[..., None]
            img = img * (1.0 - 0.72 * scratch)
    elif defect_type == "stain":
        yy, xx = np.ogrid[:h, :w]
        cy, cx = rng.uniform(h * 0.3, h * 0.7), rng.uniform(w * 0.3, w * 0.7)
        sigma = rng.uniform(size * 0.06, size * 0.11)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))[..., None]
        tint = np.array([0.45, 0.22, 0.08])
        img = img * (1.0 - 0.65 * blob) + tint * (0.65 * blob)
    elif defect_type == "particle":
        for _ in range(int(rng.integers(4, 10))):
            py, px = int(rng.integers(h * 0.25, h * 0.75)), int(rng.integers(w * 0.25, w * 0.75))
            rad = int(rng.integers(1, 3))
            yy, xx = np.ogrid[:h, :w]
            spot = (yy - py) ** 2 + (xx - px) ** 2 <= rad**2
            img[spot] *= 0.15
    elif defect_type == "uneven":
        yy = np.linspace(0, 1, h)[:, None]
        band = 0.18 * np.sin(2 * np.pi * yy * rng.uniform(3.0, 6.0) + rng.uniform(0, 6))
        img = np.clip(img + band[..., None], 0.0, 1.0)
    elif defect_type is not None:
        raise ValueError(f"unknown defect_type {defect_type}")

    img[~mask] = 0.06
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def save_png(array: NDArray[np.uint8], path: str) -> None:
    Image.fromarray(array).save(path)
