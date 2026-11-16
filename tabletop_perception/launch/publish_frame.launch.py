"""Publish frames/sample_00 on the camera topics."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="tabletop_perception",
                executable="publish_camera_frame",
                name="tabletop_camera",
                output="screen",
                parameters=[{"frame": "", "rate": 1.0}],
            )
        ]
    )
