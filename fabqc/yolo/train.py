"""Train and evaluate a small YOLO detector on real NEU-DET images."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def train(
    data_yaml: Path,
    output_dir: Path,
    model_name: str = "yolo11n.pt",
    epochs: int = 20,
    image_size: int = 256,
    batch: int = 16,
) -> dict:
    from ultralytics import YOLO

    project = output_dir.resolve()
    model = YOLO(model_name)
    result = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device="cpu",
        workers=2,
        project=str(project),
        name="train",
        exist_ok=True,
        seed=42,
        deterministic=True,
        cache=False,
        plots=True,
        verbose=True,
    )
    best = project / "train" / "weights" / "best.pt"
    evaluated = YOLO(str(best)).val(
        data=str(data_yaml.resolve()),
        split="test",
        imgsz=image_size,
        batch=batch,
        device="cpu",
        workers=2,
        project=str(project),
        name="test",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    speed = evaluated.speed
    metrics = {
        "dataset": "NEU-DET official images; deterministic 80/10/10 split, seed 42",
        "model": model_name,
        "epochs": epochs,
        "image_size": image_size,
        "test_images": 180,
        "precision": float(evaluated.box.mp),
        "recall": float(evaluated.box.mr),
        "map50": float(evaluated.box.map50),
        "map50_95": float(evaluated.box.map),
        "per_class_map50_95": {
            evaluated.names[i]: float(value)
            for i, value in enumerate(evaluated.box.maps)
        },
        "speed_ms_per_image": {
            "preprocess": float(speed.get("preprocess", 0.0)),
            "inference": float(speed.get("inference", 0.0)),
            "postprocess": float(speed.get("postprocess", 0.0)),
        },
        "hardware": "CPU",
        "license_notes": {
            "dataset": "NEU-DET publisher states no explicit license; research use only.",
            "ultralytics": "AGPL-3.0 package; commercial closed-source use needs a compatible license.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    for name in (
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "val_batch0_pred.jpg",
        "val_batch0_labels.jpg",
    ):
        source = project / "test" / name
        if source.exists():
            shutil.copy2(source, artifacts / name)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/neudet_yolo"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    train(args.data, args.out, args.model, args.epochs, args.imgsz, args.batch)


if __name__ == "__main__":
    main()

