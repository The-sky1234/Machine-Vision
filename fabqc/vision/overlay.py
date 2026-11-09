from __future__ import annotations

import base64
import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def png_data_url(array: NDArray[np.uint8]) -> str:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def colorize_heatmap(heat: NDArray[np.floating], size: tuple[int, int]) -> NDArray[np.uint8]:
    h = np.asarray(heat, dtype=np.float32)
    h = h / (float(h.max()) + 1e-9)
    img = Image.fromarray((h * 255).astype(np.uint8), mode="L").resize(size, Image.BILINEAR)
    x = np.asarray(img).astype(np.float32) / 255.0
    rgb = np.stack((x, x * 0.28, np.clip(1.0 - x, 0, 1) * 0.15), axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def overlay_heatmap(image: NDArray[np.uint8], heat: NDArray[np.floating]) -> NDArray[np.uint8]:
    color = colorize_heatmap(heat, (image.shape[1], image.shape[0])).astype(np.float32) / 255.0
    base = image.astype(np.float32) / 255.0
    h = color[..., 0]
    mix = base * (0.45 + 0.25 * (1.0 - h[..., None])) + color * (0.55 * h[..., None])
    return (np.clip(mix, 0, 1) * 255).astype(np.uint8)
