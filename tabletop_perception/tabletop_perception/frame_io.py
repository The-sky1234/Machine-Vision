"""Load and save one overhead-camera frame.

A frame directory holds rgb.png, depth.png (uint16 millimeters),
segmentation.png (instance ids), and meta.json (intrinsics and poses).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from tabletop_perception.model import Camera, RigidObject
from tabletop_perception.topics import COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC


class Frame:
    def __init__(
        self,
        name: str,
        camera: Camera,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        segmentation: np.ndarray,
        objects: list[RigidObject],
        table_z_m: float,
        table_half_extent_m: tuple[float, float],
        source: str,
    ) -> None:
        self.name = name
        self.camera = camera
        self.rgb = rgb
        self.depth_m = depth_m
        self.segmentation = segmentation
        self.objects = objects
        self.table_z_m = float(table_z_m)
        self.table_half_extent_m = (
            float(table_half_extent_m[0]),
            float(table_half_extent_m[1]),
        )
        self.source = source


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_frames_dir() -> Path:
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
    except ImportError:
        return package_root() / "frames"
    try:
        share = Path(get_package_share_directory("tabletop_perception")) / "frames"
    except PackageNotFoundError:
        return package_root() / "frames"
    if share.is_dir():
        return share
    return package_root() / "frames"


def list_frame_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"No frame directories with meta.json under {root}")
    found = [path for path in sorted(root.iterdir()) if (path / "meta.json").is_file()]
    if not found:
        raise FileNotFoundError(f"No frame directories with meta.json under {root}")
    return found


def save_frame(frame: Frame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(directory / "rgb.png"), rgb_bgr):
        raise RuntimeError(f"failed to write {directory / 'rgb.png'}")
    depth_mm = np.clip(np.rint(frame.depth_m * 1000.0), 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(directory / "depth.png"), depth_mm):
        raise RuntimeError(f"failed to write {directory / 'depth.png'}")
    if not cv2.imwrite(str(directory / "segmentation.png"), frame.segmentation):
        raise RuntimeError(f"failed to write {directory / 'segmentation.png'}")
    (directory / "meta.json").write_text(
        json.dumps(_meta(frame), indent=2) + "\n",
        encoding="utf-8",
    )


def load_frame(directory: Path) -> Frame:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    camera = _camera_from_meta(meta["camera"])
    rgb_bgr = _read(directory / "rgb.png", cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth_mm = _read(directory / "depth.png", cv2.IMREAD_UNCHANGED)
    segmentation = _read(directory / "segmentation.png", cv2.IMREAD_UNCHANGED)
    if segmentation.ndim == 3:
        segmentation = segmentation[:, :, 0]
    _check_shape("rgb", rgb, camera.height, camera.width, ndim=3)
    _check_shape("depth", depth_mm, camera.height, camera.width, ndim=2)
    _check_shape("segmentation", segmentation, camera.height, camera.width, ndim=2)
    table = meta["table"]
    half = table["half_extent_m"]
    return Frame(
        name=str(meta["name"]),
        camera=camera,
        rgb=rgb,
        depth_m=depth_mm.astype(np.float32) / 1000.0,
        segmentation=segmentation.astype(np.uint8),
        objects=[_object_from_meta(item) for item in meta["objects"]],
        table_z_m=float(table["z_m"]),
        table_half_extent_m=(float(half[0]), float(half[1])),
        source=str(meta.get("source", "saved_frame")),
    )


def _meta(frame: Frame) -> dict:
    camera = frame.camera
    return {
        "schema_version": 1,
        "name": frame.name,
        "source": frame.source,
        "topics": {
            "color": COLOR_TOPIC,
            "depth": DEPTH_TOPIC,
            "info": INFO_TOPIC,
        },
        "encodings": {
            "rgb": "png rgb8",
            "depth": "png uint16 millimeters, 0 is invalid",
            "segmentation": "png uint8 instance id, 0 is background",
        },
        "camera": {
            "width": camera.width,
            "height": camera.height,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "distortion": list(camera.distortion),
            "distortion_model": "plumb_bob",
            "optical_frame": camera.optical_frame,
        },
        "table": {
            "z_m": frame.table_z_m,
            "half_extent_m": list(frame.table_half_extent_m),
        },
        "objects": [_object_meta(obj) for obj in frame.objects],
    }


def _object_meta(obj: RigidObject) -> dict:
    return {
        "id": obj.object_id,
        "name": obj.name,
        "position_m": obj.position_m.tolist(),
        "quaternion_xyzw": obj.quaternion_xyzw.tolist(),
        "size_m": obj.size_m.tolist(),
        "color_rgb": list(obj.color_rgb),
    }


def _camera_from_meta(data: dict) -> Camera:
    return Camera(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        distortion=tuple(data["distortion"]),
        optical_frame=str(data["optical_frame"]),
    )


def _object_from_meta(data: dict) -> RigidObject:
    return RigidObject(
        object_id=int(data["id"]),
        name=str(data["name"]),
        position_m=data["position_m"],
        quaternion_xyzw=data["quaternion_xyzw"],
        size_m=data["size_m"],
        color_rgb=tuple(data["color_rgb"]),
    )


def _read(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _check_shape(label: str, image: np.ndarray, height: int, width: int, ndim: int) -> None:
    if image.ndim != ndim or image.shape[0] != height or image.shape[1] != width:
        raise ValueError(
            f"{label} shape {image.shape} does not match camera {width}x{height}"
        )
