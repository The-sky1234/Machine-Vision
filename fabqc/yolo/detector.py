"""Inference wrapper for the NEU-DET YOLO model."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "neudet_yolo11n.pt"

CLASS_ZH = {
    "crazing": "龟裂",
    "inclusion": "夹杂物",
    "patches": "斑块",
    "pitted_surface": "点蚀",
    "rolled-in_scale": "压入氧化皮",
    "scratches": "划痕",
}


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    class_name_zh: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(slots=True)
class DetectionResult:
    detections: list[Detection]
    annotated_rgb: NDArray[np.uint8]
    inference_ms: float


_lock = threading.Lock()
_model = None


def model_available() -> bool:
    return MODEL_PATH.exists()


def get_model():
    global _model
    if not model_available():
        raise FileNotFoundError(MODEL_PATH)
    with _lock:
        if _model is None:
            from ultralytics import YOLO

            _model = YOLO(str(MODEL_PATH))
    return _model


def detect(image: Image.Image, confidence: float = 0.25) -> DetectionResult:
    model = get_model()
    rgb = np.asarray(image.convert("RGB"))
    started = time.perf_counter()
    result = model.predict(
        source=rgb,
        imgsz=256,
        conf=confidence,
        device="cpu",
        verbose=False,
    )[0]
    elapsed = (time.perf_counter() - started) * 1000.0
    detections: list[Detection] = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            name = str(result.names[class_id])
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=name,
                    class_name_zh=CLASS_ZH.get(name, name),
                    confidence=float(box.conf.item()),
                    box_xyxy=xyxy,
                )
            )
    # Ultralytics plot() is BGR because it targets OpenCV.
    annotated_bgr = result.plot(labels=True, conf=True)
    annotated_rgb = annotated_bgr[..., ::-1].copy()
    return DetectionResult(detections, annotated_rgb, elapsed)

