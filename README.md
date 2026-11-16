# Machine-Vision

用 Python、OpenCV 和 Ultralytics YOLO 检测一张静态图片或一个视频文件里的物体，画出边界框和类别标签，并把结果写到指定路径。不需要摄像头。

默认权重是 **`yolo26n.pt`**。这是 Ultralytics 8.4 检测任务自带的 YOLO nano 模型。第一次运行时程序只会下载这一份 nano 权重，不会再拉取更大的模型。

## 安装

需要 Python 3.8 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows 下激活虚拟环境用 `.venv\Scripts\activate`。

`requirements.txt` 固定了这两个直接依赖：

- `opencv-python==4.14.0.94`
- `ultralytics==8.4.215`

`ultralytics` 会顺便安装 PyTorch 等自身依赖。

## 准备示例图片

安装完成后，包内自带一张公开示例图 `bus.jpg`（一辆公交车和几个人）。查看它的路径：

```bash
python -c "from pathlib import Path; import ultralytics; print(Path(ultralytics.__file__).parent / 'assets' / 'bus.jpg')"
```

也可以自己下载同一张图：

```bash
mkdir -p samples
curl -L -o samples/bus.jpg https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg
```

视频模式读取本地视频文件。把你的文件放到例如 `samples/clip.mp4` 即可，仓库不附带视频。

## 检测图片

```bash
python detect.py image \
  --source samples/bus.jpg \
  --output outputs/bus.jpg
```

把 `--source` 换成上一步打印出的包内 `bus.jpg` 路径也可以。

## 检测视频

```bash
python detect.py video \
  --source samples/clip.mp4 \
  --output outputs/clip.mp4
```

输出视频使用 OpenCV 的 `mp4v` 编码。

## 可选参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `yolo26n.pt` | 权重文件。保持默认即可，只使用 nano |
| `--conf` | `0.25` | 置信度阈值，范围 0 到 1 |

终端会打印检出的目标数量，以及写出的文件路径。权重下载失败时，程序退出并打印模型名 `yolo26n.pt` 和具体错误，命令本身不需要改。
