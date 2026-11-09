"""Convert the official NEU-DET VOC release into a deterministic YOLO split.

The dataset is intentionally not redistributed. Download it from the official
Northeastern University page, extract it, then point this command at the
directory containing ``IMAGES/`` and ``ANNOTATIONS/``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from fabqc.yolo import CLASSES


def _normalise_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def convert_annotation(xml_path: Path) -> tuple[str, list[str], Counter[str]]:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    size = root.find("size")
    if not filename or size is None:
        raise ValueError(f"invalid VOC annotation: {xml_path}")
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {xml_path}")

    rows: list[str] = []
    counts: Counter[str] = Counter()
    seen: set[tuple[int, float, float, float, float]] = set()
    for obj in root.findall("object"):
        name = _normalise_name(obj.findtext("name", ""))
        if name not in CLASSES:
            raise ValueError(f"unknown class {name!r} in {xml_path}")
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = max(0.0, float(box.findtext("xmin", "0")))
        ymin = max(0.0, float(box.findtext("ymin", "0")))
        xmax = min(width, float(box.findtext("xmax", "0")))
        ymax = min(height, float(box.findtext("ymax", "0")))
        if xmax <= xmin or ymax <= ymin:
            continue
        class_id = CLASSES.index(name)
        xc = ((xmin + xmax) / 2.0) / width
        yc = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        key = (class_id, round(xc, 8), round(yc, 8), round(bw, 8), round(bh, 8))
        if key in seen:
            continue
        seen.add(key)
        rows.append(f"{class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
        counts[name] += 1
    return filename, rows, counts


def _stratified_split(annotation_paths: list[Path], seed: int) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {name: [] for name in CLASSES}
    for path in annotation_paths:
        prefix = _normalise_name(path.stem.rsplit("_", 1)[0])
        if prefix not in groups:
            raise ValueError(f"cannot infer class from filename {path.name}")
        groups[prefix].append(path)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for paths in groups.values():
        paths = sorted(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        splits["train"].extend(paths[:n_train])
        splits["val"].extend(paths[n_train : n_train + n_val])
        splits["test"].extend(paths[n_train + n_val :])
    return splits


def prepare(raw_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    images_dir = raw_dir / "IMAGES"
    annotations_dir = raw_dir / "ANNOTATIONS"
    annotations = sorted(annotations_dir.glob("*.xml"))
    if not images_dir.is_dir() or len(annotations) != 1800:
        raise ValueError(
            f"expected official NEU-DET with 1800 XML files under {raw_dir}; "
            f"found {len(annotations)}"
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    splits = _stratified_split(annotations, seed)
    summary: dict = {
        "dataset": "NEU-DET",
        "source": "Northeastern University surface defect database",
        "license": "No license stated by dataset publisher; research use only.",
        "seed": seed,
        "classes": list(CLASSES),
        "splits": {},
    }
    total_boxes: Counter[str] = Counter()

    for split, paths in splits.items():
        out_images = output_dir / "images" / split
        out_labels = output_dir / "labels" / split
        out_images.mkdir(parents=True)
        out_labels.mkdir(parents=True)
        boxes: Counter[str] = Counter()
        for xml_path in paths:
            filename, rows, counts = convert_annotation(xml_path)
            source_image = images_dir / filename
            if not source_image.exists():
                source_image = images_dir / f"{xml_path.stem}.jpg"
            if not source_image.exists():
                raise FileNotFoundError(source_image)
            shutil.copy2(source_image, out_images / source_image.name)
            (out_labels / f"{source_image.stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else "")
            )
            boxes.update(counts)
        total_boxes.update(boxes)
        summary["splits"][split] = {
            "images": len(paths),
            "boxes": sum(boxes.values()),
            "boxes_by_class": dict(boxes),
        }

    yaml = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    yaml.extend(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (output_dir / "data.yaml").write_text("\n".join(yaml) + "\n")
    summary["total_images"] = sum(v["images"] for v in summary["splits"].values())
    summary["total_boxes"] = sum(total_boxes.values())
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed/neudet"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw, args.out, args.seed), indent=2))


if __name__ == "__main__":
    main()

