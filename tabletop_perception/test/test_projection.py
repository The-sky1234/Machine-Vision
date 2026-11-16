"""Projection of ground-truth box corners against the segmentation."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tabletop_perception.model import (
    Camera,
    RigidObject,
    box_corners,
    project_object_points,
    quaternion_to_matrix,
    yaw_quaternion,
)
from tabletop_perception.projection import DEFAULT_THRESHOLD_PX, evaluate_frame
from tabletop_perception.synthesize import first_hit_t, render_scene, sample_scenes
from tabletop_perception.frame_io import load_frame
from tabletop_perception.topics import COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_project_points_match_known_pixels():
    camera = Camera(320, 240, 300.0, 300.0, 160.0, 120.0)
    identity = RigidObject(1, "box", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.2], (0, 0, 0))
    yawed = identity.clone()
    yawed.quaternion_xyzw = yaw_quaternion(90.0)
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)

    identity_px = project_object_points(points, identity, camera)
    yawed_px = project_object_points(points, yawed, camera)

    np.testing.assert_allclose(identity_px[0], [160.0, 120.0], atol=1e-6)
    np.testing.assert_allclose(identity_px[1], [220.0, 120.0], atol=1e-6)
    np.testing.assert_allclose(yawed_px[1], [160.0, 180.0], atol=1e-6)
    np.testing.assert_allclose(
        quaternion_to_matrix(yaw_quaternion(90.0)),
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-8,
    )


def test_project_points_match_pinhole_equation():
    camera = Camera(320, 240, 280.0, 310.0, 150.0, 130.0)
    obj = RigidObject(
        1,
        "box",
        [0.04, -0.03, 0.8],
        yaw_quaternion(37.0),
        [0.1, 0.08, 0.06],
        (1, 2, 3),
    )
    points = np.vstack([np.zeros((1, 3)), box_corners(obj.size_m)])
    rotation = quaternion_to_matrix(obj.quaternion_xyzw)
    camera_points = points @ rotation.T + obj.position_m
    expected_u = camera.fx * camera_points[:, 0] / camera_points[:, 2] + camera.cx
    expected_v = camera.fy * camera_points[:, 1] / camera_points[:, 2] + camera.cy
    actual = project_object_points(points, obj, camera)
    np.testing.assert_allclose(actual[:, 0], expected_u, atol=1e-6)
    np.testing.assert_allclose(actual[:, 1], expected_v, atol=1e-6)


def test_ray_hits_near_face_and_misses_parallel_ray():
    box_min = np.array([-0.1, -0.1, -0.1])
    box_max = np.array([0.1, 0.1, 0.1])
    origin = np.array([0.0, 0.0, -1.0])
    hit = first_hit_t(origin, np.array([0.05, -0.02, 1.0]), box_min, box_max)
    miss = first_hit_t(origin, np.array([1.0, 0.0, 0.0]), box_min, box_max)
    assert float(hit) == pytest.approx(0.9)
    assert math.isinf(float(miss))


def test_synthetic_frames_project_inside_segmentation():
    for scene in sample_scenes():
        frame = render_scene(scene)
        assert frame.depth_m[0, 0] == 0.0
        areas = []
        for obj in frame.objects:
            area = int(np.count_nonzero(frame.segmentation == obj.object_id))
            assert area > 100
            areas.append(area)
        assert int(np.count_nonzero(frame.segmentation > 0)) == sum(areas)
        report = evaluate_frame(frame, DEFAULT_THRESHOLD_PX)
        assert report.passed, _dump(report)
        for item in report.objects:
            assert item.depth_error_m is not None
            assert item.depth_error_m < 1e-3
            assert item.projection_error_px <= DEFAULT_THRESHOLD_PX


def test_shifted_pose_fails_the_check():
    frame = render_scene(sample_scenes()[0])
    frame.objects[0].position_m = frame.objects[0].position_m + np.array([0.05, 0.0, 0.0])
    report = evaluate_frame(frame, DEFAULT_THRESHOLD_PX)
    assert report.objects[0].passed is False
    assert report.objects[0].projection_error_px > DEFAULT_THRESHOLD_PX
    assert report.passed is False


def test_saved_frames_match_renderer_and_pass():
    root = PACKAGE_ROOT / "frames"
    for scene in sample_scenes():
        fresh = render_scene(scene)
        saved = load_frame(root / scene.name)
        meta = json.loads((root / scene.name / "meta.json").read_text(encoding="utf-8"))
        assert meta["topics"] == {
            "color": COLOR_TOPIC,
            "depth": DEPTH_TOPIC,
            "info": INFO_TOPIC,
        }
        assert saved.name == fresh.name
        np.testing.assert_array_equal(saved.rgb, fresh.rgb)
        np.testing.assert_array_equal(saved.segmentation, fresh.segmentation)
        quantized = np.rint(fresh.depth_m * 1000.0) / 1000.0
        np.testing.assert_allclose(saved.depth_m, quantized, atol=1e-6)
        report = evaluate_frame(saved, DEFAULT_THRESHOLD_PX)
        assert report.passed, _dump(report)
        for item in report.objects:
            assert item.depth_error_m is not None
            assert item.depth_error_m <= 0.001


def test_camera_topic_names():
    assert COLOR_TOPIC == "/camera/color"
    assert DEPTH_TOPIC == "/camera/depth"
    assert INFO_TOPIC == "/camera/info"


def test_cli_prints_projection_error():
    completed = _run_module("tabletop_perception.check_node")
    assert completed.returncode == 0, completed.stderr
    assert "/camera/color" in completed.stdout
    assert "/camera/depth" in completed.stdout
    assert "/camera/info" in completed.stdout
    assert "max_projection_error_px=" in completed.stdout
    assert "status=PASS" in completed.stdout
    assert "status=FAIL" not in completed.stdout


def test_publish_node_reports_missing_ros():
    if importlib.util.find_spec("rclpy") is not None:
        pytest.skip("ROS 2 is installed")
    completed = _run_module("tabletop_perception.publish_node")
    assert completed.returncode == 1
    assert "/camera/color" in completed.stderr
    assert "/camera/depth" in completed.stderr
    assert "/camera/info" in completed.stderr


def _run_module(module: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _dump(report) -> str:
    parts = [f"{report.frame_name} max={report.max_projection_error_px}"]
    for item in report.objects:
        parts.append(
            f"{item.name} bbox={item.bbox_error_px:.3f} "
            f"outside={item.max_outside_px:.3f} center={item.center_error_px:.3f} "
            f"depth={item.depth_error_m}"
        )
    return "\n".join(parts)
