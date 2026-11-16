"""Publish one saved frame on /camera/color, /camera/depth, and /camera/info."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tabletop_perception.frame_io import Frame, default_frames_dir, load_frame
from tabletop_perception.topics import COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC


def main(argv: list[str] | None = None) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError:
        sys.stderr.write(
            "rclpy is not installed, so /camera/color, /camera/depth, and "
            "/camera/info cannot be published.\n"
            "Run the projection check instead:\n"
            "  python3 -m tabletop_perception\n"
        )
        return 1

    class TabletopCamera(Node):
        def __init__(self) -> None:
            super().__init__("tabletop_camera")
            self.declare_parameter("frame", "")
            self.declare_parameter("rate", 1.0)
            raw = self.get_parameter("frame").get_parameter_value().string_value
            rate = float(self.get_parameter("rate").get_parameter_value().double_value)
            if rate <= 0.0:
                rate = 1.0
            path = Path(raw) if raw else default_frames_dir() / "sample_00"
            self._frame = load_frame(path)
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self._color_pub = self.create_publisher(Image, COLOR_TOPIC, qos)
            self._depth_pub = self.create_publisher(Image, DEPTH_TOPIC, qos)
            self._info_pub = self.create_publisher(CameraInfo, INFO_TOPIC, qos)
            self.get_logger().info(
                f"Publishing {path} on {COLOR_TOPIC}, {DEPTH_TOPIC}, {INFO_TOPIC}"
            )
            self._publish()
            self.create_timer(1.0 / rate, self._publish)

        def _publish(self) -> None:
            stamp = self.get_clock().now().to_msg()
            frame = self._frame
            self._color_pub.publish(_color_message(Image, frame, stamp))
            self._depth_pub.publish(_depth_message(Image, frame, stamp))
            self._info_pub.publish(_info_message(CameraInfo, frame, stamp))

    rclpy.init(args=argv)
    try:
        node = TabletopCamera()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
    return 0


def _color_message(image_cls, frame: Frame, stamp):
    image = np.ascontiguousarray(frame.rgb)
    message = image_cls()
    message.header.stamp = stamp
    message.header.frame_id = frame.camera.optical_frame
    message.height = frame.camera.height
    message.width = frame.camera.width
    message.encoding = "rgb8"
    message.is_bigendian = 0
    message.step = frame.camera.width * 3
    message.data = image.tobytes()
    return message


def _depth_message(image_cls, frame: Frame, stamp):
    depth = np.array(frame.depth_m, dtype=np.float32, copy=True)
    depth[depth <= 0.0] = np.nan
    message = image_cls()
    message.header.stamp = stamp
    message.header.frame_id = frame.camera.optical_frame
    message.height = frame.camera.height
    message.width = frame.camera.width
    message.encoding = "32FC1"
    message.is_bigendian = 0
    message.step = frame.camera.width * 4
    message.data = np.ascontiguousarray(depth).tobytes()
    return message


def _info_message(info_cls, frame: Frame, stamp):
    camera = frame.camera
    message = info_cls()
    message.header.stamp = stamp
    message.header.frame_id = camera.optical_frame
    message.height = camera.height
    message.width = camera.width
    message.distortion_model = "plumb_bob"
    message.d = [float(value) for value in camera.distortion]
    message.k = [
        camera.fx, 0.0, camera.cx,
        0.0, camera.fy, camera.cy,
        0.0, 0.0, 1.0,
    ]
    message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    message.p = [
        camera.fx, 0.0, camera.cx, 0.0,
        0.0, camera.fy, camera.cy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    return message


if __name__ == "__main__":
    sys.exit(main())
