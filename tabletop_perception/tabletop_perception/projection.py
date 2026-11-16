"""Project rigid-object points and compare them with the segmentation."""

from __future__ import annotations

import math

import cv2
import numpy as np

from tabletop_perception.frame_io import Frame
from tabletop_perception.model import (
    box_corners,
    is_yaw_only,
    project_object_points,
)

# Rasterized masks sit about a pixel inside the continuous silhouette.
DEFAULT_THRESHOLD_PX = 2.0


class ObjectReport:
    def __init__(
        self,
        name: str,
        object_id: int,
        center_error_px: float,
        bbox_error_px: float,
        max_outside_px: float,
        depth_error_m: float | None,
        projection_error_px: float,
        passed: bool,
    ) -> None:
        self.name = name
        self.object_id = object_id
        self.center_error_px = center_error_px
        self.bbox_error_px = bbox_error_px
        self.max_outside_px = max_outside_px
        self.depth_error_m = depth_error_m
        self.projection_error_px = projection_error_px
        self.passed = passed


class FrameReport:
    def __init__(
        self,
        frame_name: str,
        objects: list[ObjectReport],
        max_projection_error_px: float,
        threshold_px: float,
        passed: bool,
    ) -> None:
        self.frame_name = frame_name
        self.objects = objects
        self.max_projection_error_px = max_projection_error_px
        self.threshold_px = threshold_px
        self.passed = passed


def evaluate_frame(frame: Frame, threshold_px: float = DEFAULT_THRESHOLD_PX) -> FrameReport:
    if not frame.objects:
        return FrameReport(frame.name, [], math.inf, threshold_px, False)
    reports = [evaluate_object(frame, obj, threshold_px) for obj in frame.objects]
    worst = max(item.projection_error_px for item in reports)
    return FrameReport(
        frame_name=frame.name,
        objects=reports,
        max_projection_error_px=worst,
        threshold_px=threshold_px,
        passed=all(item.passed for item in reports),
    )


def evaluate_object(frame: Frame, obj, threshold_px: float) -> ObjectReport:
    binary = np.ascontiguousarray(frame.segmentation == obj.object_id, dtype=np.uint8)
    if not np.any(binary):
        return _failed(obj)

    points = np.vstack([np.zeros((1, 3), dtype=np.float64), box_corners(obj.size_m)])
    pixels = project_object_points(points, obj, frame.camera)
    center_px = pixels[0]
    corner_px = pixels[1:]

    moments = cv2.moments(binary, binaryImage=True)
    centroid = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )
    center_error = float(np.hypot(*(center_px - centroid)))
    bbox_error = _bbox_error(binary, corner_px)
    max_outside = _max_outside_px(binary, corner_px)
    projection_error = max(bbox_error, max_outside)
    return ObjectReport(
        name=obj.name,
        object_id=obj.object_id,
        center_error_px=center_error,
        bbox_error_px=bbox_error,
        max_outside_px=max_outside,
        depth_error_m=_depth_error_m(frame, obj, center_px),
        projection_error_px=projection_error,
        passed=math.isfinite(projection_error) and projection_error <= threshold_px,
    )


def _bbox_error(binary: np.ndarray, corner_px: np.ndarray) -> float:
    x, y, width, height = cv2.boundingRect(binary)
    mask_box = (float(x), float(y), float(x + width - 1), float(y + height - 1))
    projected = (
        float(np.min(corner_px[:, 0])),
        float(np.min(corner_px[:, 1])),
        float(np.max(corner_px[:, 0])),
        float(np.max(corner_px[:, 1])),
    )
    return max(abs(left - right) for left, right in zip(mask_box, projected))


def _max_outside_px(binary: np.ndarray, corner_px: np.ndarray) -> float:
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = found[0] if len(found) == 2 else found[1]
    if not contours:
        return math.inf
    contour = max(contours, key=cv2.contourArea)
    outside = []
    for u, v in corner_px:
        signed = cv2.pointPolygonTest(contour, (float(u), float(v)), True)
        outside.append(max(0.0, -float(signed)))
    return max(outside)


def _depth_error_m(frame: Frame, obj, center_px: np.ndarray) -> float | None:
    """Top-face depth for a yaw-only box. The top is the nearer Z face."""
    if not is_yaw_only(obj.quaternion_xyzw):
        return None
    x = int(math.floor(float(center_px[0]) + 0.5))
    y = int(math.floor(float(center_px[1]) + 0.5))
    camera = frame.camera
    if x < 0 or y < 0 or x >= camera.width or y >= camera.height:
        return None
    if int(frame.segmentation[y, x]) != obj.object_id:
        return None
    measured = float(frame.depth_m[y, x])
    if measured <= 0.0:
        return None
    top_z = float(obj.position_m[2] - obj.size_m[2] / 2.0)
    return abs(measured - top_z)


def _failed(obj) -> ObjectReport:
    return ObjectReport(
        name=obj.name,
        object_id=obj.object_id,
        center_error_px=math.inf,
        bbox_error_px=math.inf,
        max_outside_px=math.inf,
        depth_error_m=None,
        projection_error_px=math.inf,
        passed=False,
    )
