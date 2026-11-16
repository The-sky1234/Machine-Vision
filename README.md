# 固定俯视相机：桌面刚体投影检查

一台固定的俯视相机看向桌面上的几个刚体。每一帧包含 RGB、深度、相机内参、物体在相机坐标系下的位姿，以及实例分割。位姿和分割用真值，不训练检测模型。这个仓库不做抓取，也不做控制。

当前环境没有 Gazebo，也没有 Isaac Sim，示例帧不是仿真器渲染的。`frames/` 里的两帧由针孔模型和长方体求交生成，字段按以后从仿真器导出的方式来组织。投影检查只读这些文件。

## 目录

```
tabletop_perception/                 唯一的 ROS 2 Python 包
├── package.xml                      包名、依赖。有 ROS 2 时 colcon 用它
├── setup.py / setup.cfg             两个可执行文件：check_projection、publish_camera_frame
├── requirements.txt                 没有 ROS 时，跑投影检查要装的 Python 包
├── launch/                          把示例帧发到相机话题
├── frames/sample_00/                示例帧：红、蓝两个轴对齐盒子
├── frames/sample_01/                示例帧：绿、黄两个绕竖直轴转过的盒子
├── tabletop_perception/             读帧、OpenCV 投影、误差、两个节点
└── test/test_projection.py          投影是否落在分割上
```

`tabletop_perception/tabletop_perception/` 里各文件的分工：

| 文件 | 作用 |
| --- | --- |
| `topics.py` | `/camera/color`、`/camera/depth`、`/camera/info` |
| `model.py` | 内参、盒子位姿、`cv2.projectPoints` |
| `frame_io.py` | 读写一帧目录 |
| `synthesize.py` | 没有仿真器时生成示例帧 |
| `projection.py` | 角点对分割轮廓和包围盒的像素误差 |
| `check_node.py` | 读一帧并打印误差 |
| `publish_node.py` | 有 ROS 2 时把同一帧发到三个话题 |

每一帧目录里有四份文件：

| 文件 | 内容 |
| --- | --- |
| `rgb.png` | 彩色图 |
| `depth.png` | uint16，毫米。0 表示无效 |
| `segmentation.png` | uint8 实例编号。0 是桌面或背景，物体从 1 开始 |
| `meta.json` | 内参、桌面、每个刚体的尺寸，以及相机坐标系下的位置和四元数 |

## 话题

以后换真实相机时沿用这三个名字：

- `/camera/color`：彩色，`rgb8`
- `/camera/depth`：深度，米，`32FC1`。文件里的 0 在发布时写成 NaN
- `/camera/info`：内参，`sensor_msgs/CameraInfo`

分割是仿真真值，只放在 `segmentation.png`，不占用这三个话题。

## 运行投影检查

```bash
cd tabletop_perception
python3 -m pip install -r requirements.txt
python3 -m pytest
python3 -m tabletop_perception
```

最后一条命令读取 `frames/` 下的示例帧。对每个物体，用 `cv2.projectPoints` 投影盒子中心和 8 个角点，再与分割轮廓、包围盒比较，并打印像素误差。默认阈值是 2 像素。全部通过时返回码为 0。

只查一帧：

```bash
python3 -m tabletop_perception.check_node frames/sample_00
python3 -m tabletop_perception.check_node --threshold 2 frames/sample_01
```

重新生成示例帧：

```bash
python3 -m tabletop_perception.synthesize --output frames
```

一次通过的输出类似：

```
topics color=/camera/color depth=/camera/depth info=/camera/info
frame=sample_00 object=red_box id=1 center_error_px=1.227 bbox_error_px=1.000 max_outside_px=1.202 depth_error_m=0.0000 projection_error_px=1.202 status=PASS
frame=sample_00 max_projection_error_px=1.202 threshold_px=2.0 status=PASS
```

误差含义：

- `max_outside_px`：8 个角点落在分割轮廓外的最大距离
- `bbox_error_px`：角点包围盒与分割包围盒四条边的最大差
- `projection_error_px`：上面两项里较大的一个。不超过阈值即通过
- `center_error_px`：投影中心到分割质心的距离，只作参考。透视会让质心偏离几何中心
- `depth_error_m`：投影中心处的深度与盒子顶面深度之差。只在绕相机 z 轴旋转时计算

示例帧上的投影误差大约 1 像素，来自分割栅格化，不是标定残差。

## 有 ROS 2 时

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd <仓库根目录>
colcon build --packages-select tabletop_perception
source install/setup.bash
ros2 launch tabletop_perception publish_frame.launch.py
```

另一个终端：

```bash
ros2 topic echo /camera/info --once
```

没有安装 `rclpy` 时，发布节点会退出，并提示改跑投影检查。

## 坐标系

相机光学坐标系：x 向右，y 向下，z 指向桌面。`position_m` 是盒子中心，单位米。`quaternion_xyzw` 把物体坐标旋到相机坐标。示例物体只绕 z 轴偏航，底面贴着桌面。桌面是 z 为 0.9 米的一块矩形，不铺满整幅图。

## 限制

Gazebo 和 Isaac Sim 在当前环境里不能运行，ROS 2 也没有安装。示例图是同一套针孔模型下的真值帧，不是仿真器画面。接到仿真器时，按 `meta.json` 的字段写出 RGB、深度、内参、位姿和分割，检查命令不用改。三个话题名就是以后替换真实相机的接口。
