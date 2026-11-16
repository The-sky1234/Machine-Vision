"""Camera model and rigid-box poses in the camera optical frame.

Optical frame: x right, y down, z from the camera toward the table.
`quaternion_xyzw` rotates object-frame points into that camera frame.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class Camera:
    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        distortion: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
        optical_frame: str = "camera_color_optical_frame",
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.distortion = tuple(float(value) for value in distortion)
        self.optical_frame = optical_frame

    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


class RigidObject:
    def __init__(
        self,
        object_id: int,
        name: str,
        position_m: np.ndarray,
        quaternion_xyzw: np.ndarray,
        size_m: np.ndarray,
        color_rgb: tuple[int, int, int],
    ) -> None:
        self.object_id = int(object_id)
        self.name = name
        self.position_m = np.asarray(position_m, dtype=np.float64).reshape(3)
        self.quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
        self.size_m = np.asarray(size_m, dtype=np.float64).reshape(3)
        self.color_rgb = tuple(int(channel) for channel in color_rgb)

    def clone(self) -> RigidObject:
        return RigidObject(
            self.object_id,
            self.name,
            self.position_m.copy(),
            self.quaternion_xyzw.copy(),
            self.size_m.copy(),
            self.color_rgb,
        )


def yaw_quaternion(degrees: float) -> np.ndarray:
    """Rotation about the camera z axis. Upright boxes stay upright."""
    half = math.radians(degrees) * 0.5
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)


def quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("quaternion has zero length")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def box_corners(size_m: np.ndarray) -> np.ndarray:
    """Eight corners in the object frame. The origin is the box center."""
    half = np.asarray(size_m, dtype=np.float64).reshape(3) / 2.0
    signs = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float64,
    )
    return signs * half


def is_yaw_only(quaternion_xyzw: np.ndarray, tolerance: float = 1e-8) -> bool:
    x, y, _, _ = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    return abs(x) <= tolerance and abs(y) <= tolerance


def project_object_points(
    points_obj: np.ndarray,
    obj: RigidObject,
    camera: Camera,
) -> np.ndarray:
    """Project object-frame points with cv2.projectPoints. Returns (N, 2) pixels."""
    points = np.asarray(points_obj, dtype=np.float64).reshape(-1, 1, 3)
    rotation = quaternion_to_matrix(obj.quaternion_xyzw)
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = np.asarray(obj.position_m, dtype=np.float64).reshape(3, 1)
    distortion = np.asarray(camera.distortion, dtype=np.float64).reshape(-1, 1)
    image_points, _ = cv2.projectPoints(points, rvec, tvec, camera.matrix(), distortion)
    return image_points.reshape(-1, 2)
