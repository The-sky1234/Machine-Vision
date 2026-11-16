from pathlib import Path

from setuptools import find_packages, setup

package_name = "tabletop_perception"


def _share_files(folder: str) -> list[tuple[str, list[str]]]:
    entries: list[tuple[str, list[str]]] = []
    root = Path(folder)
    if not root.is_dir():
        return entries
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        destination = Path("share") / package_name / path.parent
        entries.append((destination.as_posix(), [path.as_posix()]))
    return entries


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        *_share_files("launch"),
        *_share_files("frames"),
    ],
    install_requires=["setuptools", "numpy", "opencv-python-headless"],
    zip_safe=True,
    maintainer="The-sky1234",
    maintainer_email="1254484692@qq.com",
    description="Overhead tabletop camera projection check against segmentation ground truth.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "check_projection = tabletop_perception.check_node:main",
            "publish_camera_frame = tabletop_perception.publish_node:main",
        ],
    },
)
