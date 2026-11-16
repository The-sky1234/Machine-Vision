#!/usr/bin/env python3
"""Run YOLO-PCB on one bare-board image and print a pass/fail line.

The model, the six class names, and weights/baseline_fpn_loss.pt come from
https://github.com/JiaLim98/YOLO-PCB (MIT, see LICENSE). Ultralytics YOLOv5
is the detector that repository adapted. This file stays inside pcb/ so it
does not replace the repository's detect.py.

The checkpoint pickles the whole network. On current PyTorch, loading it
requires weights_only=False (see models/experimental.py).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

CLASS_NAMES = (
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
)

# BGR colors, one per class, used only on the saved image.
CLASS_COLORS = (
    (60, 60, 220),
    (40, 160, 220),
    (40, 180, 40),
    (220, 140, 40),
    (180, 60, 180),
    (200, 80, 40),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect PCB defects with the YOLO-PCB YOLOv5 checkpoint and print PASS or FAIL."
    )
    parser.add_argument(
        "--weights",
        default=str(ROOT / "weights" / "baseline_fpn_loss.pt"),
        help="checkpoint path (default: pcb/weights/baseline_fpn_loss.pt)",
    )
    parser.add_argument(
        "--source",
        default=str(ROOT / "samples" / "missinghole_1.jpg"),
        help="one input image (default: pcb/samples/missinghole_1.jpg)",
    )
    parser.add_argument("--img-size", type=int, default=608, help="inference size in pixels")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", default="cpu", help="cuda device id, or cpu")
    parser.add_argument(
        "--output",
        default="",
        help="where to write the annotated image (default: pcb/outputs/<source name>)",
    )
    return parser.parse_args()


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names[class_id])
    return str(names[class_id])


def draw_box(image, xyxy, label: str, color) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_top = max(y1 - text_height - baseline - 4, 0)
    cv2.rectangle(
        image,
        (x1, text_top),
        (x1 + text_width + 6, text_top + text_height + baseline + 4),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 3, text_top + text_height + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.conf_thres <= 1.0:
        raise SystemExit("--conf-thres must be between 0 and 1")
    if not 0.0 <= args.iou_thres <= 1.0:
        raise SystemExit("--iou-thres must be between 0 and 1")

    weights = Path(args.weights)
    source = Path(args.source)
    if not weights.is_file():
        raise SystemExit(f"Weight file not found: {weights}")
    if not source.is_file():
        raise SystemExit(f"Image not found: {source}")

    image = cv2.imread(str(source))
    if image is None:
        raise SystemExit(f"Could not read image: {source}")

    device = select_device(args.device)
    model = attempt_load(str(weights), map_location=device)
    names = model.module.names if hasattr(model, "module") else model.names
    if isinstance(names, dict):
        loaded = [str(names[i]) for i in range(len(names))]
    else:
        loaded = [str(name) for name in names]
    if tuple(loaded) != CLASS_NAMES:
        raise SystemExit(f"Checkpoint classes {loaded} do not match {list(CLASS_NAMES)}")

    img_size = check_img_size(args.img_size, s=int(model.stride.max()))
    resized = letterbox(image, new_shape=img_size)[0]
    tensor = resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    tensor = torch.from_numpy(tensor.copy()).to(device).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        prediction = model(tensor)[0]
        detections = non_max_suppression(prediction, args.conf_thres, args.iou_thres)[0]

    counts: Counter[str] = Counter()
    if detections is not None and len(detections):
        detections = detections.clone()
        detections[:, :4] = scale_coords(tensor.shape[2:], detections[:, :4], image.shape).round()
        for *xyxy, conf, cls_id in detections.tolist():
            name = class_name(names, int(cls_id))
            box = [int(round(value)) for value in xyxy]
            counts[name] += 1
            print(f"{name} {conf:.3f} xyxy=[{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
            draw_box(image, box, f"{name} {conf:.2f}", CLASS_COLORS[int(cls_id) % len(CLASS_COLORS)])

    count_text = " ".join(f"{name}={counts[name]}" for name in CLASS_NAMES)
    verdict = "FAIL" if counts["missing_hole"] else "PASS"
    print(f"{verdict} {count_text}")

    output = Path(args.output) if args.output else ROOT / "outputs" / source.name
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise SystemExit(f"Could not write image: {output}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
