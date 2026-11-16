"""Synthetic overhead frames for when Gazebo or Isaac is not available.

Each rigid object is an oriented box. Pixels are colored by casting a
pinhole ray into that box. Poses in meta.json are the ground truth the
projection check consumes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from tabletop_perception.frame_io import Frame, default_frames_dir, save_frame
from tabletop_perception.model import (
    Camera,
    RigidObject,
    quaternion_to_matrix,
    yaw_quaternion,
)

BACKGROUND_RGB = (36, 40, 46)
TABLE_RGB = (176, 164, 146)
SOURCE = "synthetic_pinhole"


class Scene:
    def __init__(
        self,
        name: str,
        camera: Camera,
        table_z_m: float,
        table_half_extent_m: tuple[float, float],
        objects: list[RigidObject],
    ) -> None:
        self.name = name
        self.camera = camera
        self.table_z_m = float(table_z_m)
        self.table_half_extent_m = table_half_extent_m
        self.objects = objects


def sample_scenes() -> list[Scene]:
    camera = Camera(width=320, height=240, fx=300.0, fy=300.0, cx=160.0, cy=120.0)
    table_z = 0.90
    table_half = (0.34, 0.26)
    return [
        Scene(
            name="sample_00",
            camera=camera,
            table_z_m=table_z,
            table_half_extent_m=table_half,
            objects=[
                RigidObject(
                    1,
                    "red_box",
                    [0.08, -0.05, 0.86],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.10, 0.08, 0.08],
                    (214, 48, 48),
                ),
                RigidObject(
                    2,
                    "blue_box",
                    [-0.11, 0.07, 0.85],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.07, 0.06, 0.10],
                    (48, 96, 214),
                ),
            ],
        ),
        Scene(
            name="sample_01",
            camera=camera,
            table_z_m=table_z,
            table_half_extent_m=table_half,
            objects=[
                RigidObject(
                    1,
                    "green_box",
                    [0.02, -0.02, 0.86],
                    yaw_quaternion(28.0),
                    [0.12, 0.06, 0.08],
                    (46, 160, 78),
                ),
                RigidObject(
                    2,
                    "yellow_box",
                    [-0.12, 0.09, 0.875],
                    yaw_quaternion(-18.0),
                    [0.05, 0.05, 0.05],
                    (214, 176, 48),
                ),
            ],
        ),
    ]


def render_scene(scene: Scene) -> Frame:
    camera = scene.camera
    height, width = camera.height, camera.width
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = BACKGROUND_RGB
    depth = np.full((height, width), np.inf, dtype=np.float64)
    segmentation = np.zeros((height, width), dtype=np.uint8)

    table_mask = _table_mask(scene)
    depth[table_mask] = scene.table_z_m
    rgb[table_mask] = TABLE_RGB

    rays = _camera_rays(camera)
    objects = [obj.clone() for obj in scene.objects]
    for obj in objects:
        t_hit = _object_ray_t(obj, rays)
        visible = t_hit < depth
        depth[visible] = t_hit[visible]
        segmentation[visible] = np.uint8(obj.object_id)
        rgb[visible] = np.asarray(obj.color_rgb, dtype=np.uint8)
        if int(np.count_nonzero(segmentation == obj.object_id)) < 20:
            raise RuntimeError(f"{scene.name}:{obj.name} is not visible")

    depth_m = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    return Frame(
        name=scene.name,
        camera=camera,
        rgb=rgb,
        depth_m=depth_m,
        segmentation=segmentation,
        objects=objects,
        table_z_m=scene.table_z_m,
        table_half_extent_m=scene.table_half_extent_m,
        source=SOURCE,
    )


def write_sample_frames(output_root: Path) -> list[Path]:
    written: list[Path] = []
    for scene in sample_scenes():
        destination = output_root / scene.name
        save_frame(render_scene(scene), destination)
        written.append(destination)
    return written


def first_hit_t(
    origin: np.ndarray,
    directions: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> np.ndarray:
    """Nearest t >= 0 where origin + t * direction hits an axis-aligned box.

    `directions` keeps its length. A miss is +inf. The ray parameter is not
    a metric distance unless the direction is a unit vector.
    """
    origin = np.asarray(origin, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)

    near = np.abs(directions) <= 1e-12
    safe = np.where(near, 1.0, directions)
    t_a = (bmin - origin) / safe
    t_b = (bmax - origin) / safe
    t_a = np.where(near, -np.inf, t_a)
    t_b = np.where(near, np.inf, t_b)
    t_enter = np.max(np.minimum(t_a, t_b), axis=-1)
    t_exit = np.min(np.maximum(t_a, t_b), axis=-1)
    inside = (origin >= bmin) & (origin <= bmax)
    parallel_miss = np.any(near & ~inside, axis=-1)
    hit = (~parallel_miss) & (t_exit >= np.maximum(t_enter, 0.0))
    t_hit = np.where(t_enter > 0.0, t_enter, 0.0)
    return np.where(hit, t_hit, np.inf)


def _camera_rays(camera: Camera) -> np.ndarray:
    columns = np.arange(camera.width, dtype=np.float64)
    rows = np.arange(camera.height, dtype=np.float64)
    u, v = np.meshgrid(columns, rows)
    directions = np.empty((camera.height, camera.width, 3), dtype=np.float64)
    directions[..., 0] = (u - camera.cx) / camera.fx
    directions[..., 1] = (v - camera.cy) / camera.fy
    directions[..., 2] = 1.0
    return directions


def _object_ray_t(obj: RigidObject, directions_camera: np.ndarray) -> np.ndarray:
    rotation = quaternion_to_matrix(obj.quaternion_xyzw)
    origin_object = -rotation.T @ obj.position_m
    directions_object = directions_camera @ rotation
    half = obj.size_m / 2.0
    # Camera rays are stored with z = 1. A pure rotation keeps the same ray
    # parameter, so t is the hit point's camera-frame Z.
    return first_hit_t(origin_object, directions_object, -half, half)


def _table_mask(scene: Scene) -> np.ndarray:
    half_x, half_y = scene.table_half_extent_m
    z = scene.table_z_m
    corners = np.array(
        [
            [-half_x, -half_y, z],
            [half_x, -half_y, z],
            [half_x, half_y, z],
            [-half_x, half_y, z],
        ],
        dtype=np.float64,
    )
    pixels = _project_camera_points(corners, scene.camera)
    hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
    mask = np.zeros((scene.camera.height, scene.camera.width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask == 1


def _project_camera_points(points_camera: np.ndarray, camera: Camera) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 1, 3)
    image_points, _ = cv2.projectPoints(
        points,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera.matrix(),
        np.asarray(camera.distortion, dtype=np.float64).reshape(-1, 1),
    )
    return image_points.reshape(-1, 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write synthetic overhead-camera sample frames.")
    parser.add_argument("--output", type=Path, default=default_frames_dir())
    args = parser.parse_args(argv)
    for path in write_sample_frames(args.output):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
