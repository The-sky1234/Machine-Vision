# PCB 裸板缺陷检测

在一张裸板照片上跑 YOLOv5，框出六类缺陷。只要出现 `missing_hole`，最后一行就是 `FAIL`，同时打印六类数量。

这个目录不替换仓库根目录的 `detect.py`。根目录那个脚本用的是通用 YOLO nano，看的是人和车，不是电路板。

## 来源和许可

检测代码、六个类别和权重 `weights/baseline_fpn_loss.pt` 来自 [JiaLim98/YOLO-PCB](https://github.com/JiaLim98/YOLO-PCB)，上游提交 `d15641ec341a7403d85917114a0806afc0805ac9`。该仓库的 MIT 许可全文在 [LICENSE](LICENSE)，版权归 JiaLim98（2021）。

YOLOv5 本身来自 [Ultralytics](https://github.com/ultralytics/yolov5)。本目录用的是 YOLO-PCB 里那份改过的 YOLOv5s FPN（`models/yolov5s_fpn.yaml`），只保留了把权重加载并跑起来所需的文件，没有训练脚本。

公开数据集来自 Ding、Dai、Li、Liu 的 TDD-net，仓库在 [Tiny-Defect-Detection-for-PCB](https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB)。这里不附带那一万多张图，只附带上游的一张样例 `samples/missinghole_1.jpg` 和一份权重。这两份文件与上游 Git 对象一致：样例 `301076bb6c92c76973bdc4a6c0614e2ad8d3504f`，权重 `3e84604d238b049a3f229c927f1ab65d616737ff`。论文页面：[Results in Engineering](https://doi.org/10.1016/j.rineng.2023.100968)。下面不写准确率，因为本仓库没有那套测试集，也没有重新测过。

## 安装

需要 Python 3.8 或更高。在仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r pcb/requirements.txt
```

Windows 下激活虚拟环境用 `.venv\Scripts\activate`。

`pcb/requirements.txt` 从 PyTorch 的 CPU 索引安装 `torch` 和 `torchvision`，不需要显卡。根目录的 `requirements.txt` 是另一个演示，不要混用同一条命令。

## 运行

在 `pcb/` 目录执行：

```bash
python detect.py --weights weights/baseline_fpn_loss.pt --source samples/missinghole_1.jpg --img-size 608 --conf-thres 0.25 --device cpu
```

608 是上游对这批图使用的输入边长（原图 600×600）。0.25 是上游 `detect.py` 的默认置信度。`--device cpu` 固定走 CPU。

终端会先逐条打印框，例如 `missing_hole 0.881 xyxy=[2, 269, 21, 297]`，最后一行是判定和六类计数：

```text
FAIL missing_hole=2 mouse_bite=0 open_circuit=0 short=0 spur=0 spurious_copper=0
```

有任意一个 `missing_hole` 就是 `FAIL`，否则是 `PASS`。其他五类只计数，不单独把这张图判成失败。这是演示规则，不是出厂标准。

带框的图写到 `pcb/outputs/`。旧权重把整个网络 pickle 进去了，当前 PyTorch 的 `torch.load` 默认 `weights_only=True`，加载会失败。`models/experimental.py` 里对这份权重使用 `weights_only=False`。

上游 README 写的是 `--conf 0.9`。在这张样例、608 输入、这份 `baseline_fpn_loss.pt` 上，0.9 一个框都不会留下。所以演示命令用 0.25，终端才能看到检测结果。

`samples/missinghole_1.jpg` 是上游放在 README 里的结果图，像素里已经画着品红色的框和 `0.9x` 文字。那不是本脚本画的。本脚本另写 `pcb/outputs/missinghole_1.jpg`，新框的标签是白字。不要把图里原有的品红数字，或终端里这张图的框数，当成测试集分数。

## 六个类别

模型只在这六个名字里选一个，看的是铜皮和孔的样子：

| 名字 | 含义 |
| --- | --- |
| `missing_hole` | 缺孔。这个位置应该有一个钻孔，图上却没有孔。 |
| `mouse_bite` | 鼠咬。板边或线边缺了一小块，像被啃过。 |
| `open_circuit` | 开路。导线断了，两边没有连上。 |
| `short` | 短路。两条不该连在一起的铜连上了。 |
| `spur` | 毛刺。导线边上多出一根尖铜。 |
| `spurious_copper` | 杂铜。不该有铜的地方多了一块铜。 |

## 面试可以讲的五点

1. **模型看见什么。** 它看见的是一张彩色裸板照片，先按比例缩放到 608 附近，再用灰色把边补齐。它不读原理图、Gerber 或钻孔文件，只根据像素在六个类别里画框。缺孔在图上就是“焊盘还在，中间没有孔”。

2. **为什么贴着板边的框可能是误报。** 样例是裁过的板子，左边缘就是图像边界。被裁掉的焊盘只剩一块铜，看起来很像缺孔；补边用的灰条也会造出一条假边缘。演示命令会打出一个左边贴着图像边界的 `missing_hole` 框。这种框在产线上应该先当可疑，用板子轮廓丢掉贴边的框，再让人看，而不是直接当报废。

3. **置信度阈值。** 框后面的小数不是校准过的概率。同一张图、同一个权重，0.25 会留下贴边的框和板内的框，0.5 在 608 输入上只留下贴着左边缘的那只，0.9 一只都不留。阈值同时改变漏检和误报。0.25 只是为了让这张样例在终端打出框，不是在测试集上选出来的工作点。

4. **这不是钢材 NEU-DET 那个项目。** NEU-DET 是热轧钢板的表面图，缺陷是龟裂、夹杂、斑块、点蚀、压入氧化皮和划痕，看的是纹理。这边是绿色裸板上的孔和铜皮几何。类别、相机、光照、输入尺寸和权重都不能换着用。把钢材权重拿来跑电路板，要么加载失败，要么框上的名字没有意义。

5. **换一个新产品要改什么。** 用那条产线自己的相机、灯光和工作距离重新采图，不要直接用这份公共权重。和工艺确认缺陷名字：新产品可能没有鼠咬，但有连锡、偏位或字符。置信度和合格规则用这条线留出的图片来定，按漏检和误杀的代价取舍，不要沿用“只有缺孔才 FAIL”或者 0.25。先抠出板子轮廓，丢掉贴边的框，孔和开路再跟钻孔文件或 Gerber 对一下。权重换成当前框架能正常加载的格式，生产环境不要靠 `weights_only=False` 去反序列化 2021 年的整网文件。
