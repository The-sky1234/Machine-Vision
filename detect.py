#!/usr/bin/env python3
"""Detect objects in an image or a video file and draw boxes with OpenCV.

The default weight is Ultralytics' current nano detector, yolo26n.pt.
The first run downloads that file only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Ultralytics 8.4 default model for the detect task (YOLO26 nano).
DEFAULT_MODEL = "yolo26n.pt"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects with YOLO and draw boxes and labels with OpenCV."
    )
    parser.add_argument("mode", choices=("image", "video"), help="run on a still image or a video file")
    parser.add_argument("--source", required=True, type=Path, help="input image or video path")
    parser.add_argument("--output", required=True, type=Path, help="where to write the annotated file")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"weight file to load (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    return parser.parse_args()


def draw_detections(frame, result) -> int:
    """Draw one rectangle and label per detection. Returns how many were drawn."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0

    names = result.names
    drawn = 0
    for box in boxes:
        x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = f"{names[class_id]} {confidence:.2f}"

        color = (40, 180, 40)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        text_top = max(y1 - text_height - baseline - 4, 0)
        cv2.rectangle(
            frame,
            (x1, text_top),
            (x1 + text_width + 6, text_top + text_height + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 3, text_top + text_height + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        drawn += 1
    return drawn


def run_image(model: YOLO, source: Path, output: Path, conf: float) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise SystemExit(f"Could not read image: {source}")

    results = model.predict(frame, conf=conf, verbose=False)
    count = draw_detections(frame, results[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise SystemExit(f"Could not write image: {output}")
    print(f"image: {count} objects -> {output}")


def run_video(model: YOLO, source: Path, output: Path, conf: float) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise SystemExit(f"Video has no frames: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not open video writer: {output}")

    frames = 0
    boxes = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        results = model.predict(frame, conf=conf, verbose=False)
        boxes += draw_detections(frame, results[0])
        writer.write(frame)
        frames += 1

    capture.release()
    writer.release()
    if frames == 0:
        raise SystemExit(f"Video has no frames: {source}")
    print(f"video: {frames} frames, {boxes} boxes -> {output}")


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Source not found: {args.source}")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1")

    suffix = args.source.suffix.lower()
    if args.mode == "image" and suffix not in IMAGE_SUFFIXES:
        print(f"Warning: {args.source.name} does not look like an image.", file=sys.stderr)
    if args.mode == "video" and suffix not in VIDEO_SUFFIXES:
        print(f"Warning: {args.source.name} does not look like a video.", file=sys.stderr)

    try:
        model = YOLO(args.model)
    except Exception as exc:
        raise SystemExit(
            f"Could not load model {args.model!r}. "
            f"The default weight is the YOLO nano file {DEFAULT_MODEL}, "
            f"which Ultralytics downloads on first use. Error: {exc}"
        ) from exc

    if args.mode == "image":
        run_image(model, args.source, args.output, args.conf)
    else:
        run_video(model, args.source, args.output, args.conf)


if __name__ == "__main__":
    main()
