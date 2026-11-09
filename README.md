# 产线质检台：真实钢材 YOLO 检测 + 设备归因演习

网页现在有两部分：

1. **真实视觉质检**：上传钢材表面照片，YOLO11n 框出龟裂、夹杂物、斑块、点蚀、压入氧化皮和划痕。
2. **设备归因演习**：用模拟设备曲线验证“设备先异常、零件后出问题”的归因规则；拿不准就拒答，改参数需要人工批准。

两部分不能混为一谈：YOLO 使用真实工业图片；公开 NEU-DET 没有设备日志，所以根因归因仍是明确标注的模拟演习，不能冒充真实工厂 RCA。

## 怎么打开

需要 Python 3.11+。仓库带有训练好的 5.3 MB YOLO11n 权重，CPU 可以运行。

```bash
uv venv .venv-fabqc && source .venv-fabqc/bin/activate
uv pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
python -m fabqc.cli serve
```

浏览器打开 http://127.0.0.1:8765

建议先选「冷却水突然变热」，再试「零件有划痕但设备是正常的」（这时应该说不知道）。

## YOLO 真实测试结果

数据是东北大学发布的 [NEU-DET](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/)：

- 1800 张 200×200 热轧钢表面真实灰度图
- 4186 个缺陷框，六类缺陷
- 确定性分层切分：1440 train / 180 val / 180 test，seed=42
- YOLO11n，ImageNet/COCO 预训练权重起步，256px，CPU 训练 20 epochs

以下数字来自**从未参与训练和调参的 180 张 test 图**：

| 指标 | 实测值 |
|---|---:|
| Precision | 0.656 |
| Recall | 0.684 |
| mAP@50 | **0.716** |
| mAP@50–95 | **0.412** |
| CPU 推理（批量测试） | 3.84 ms/图 |

分类别 mAP@50–95：划痕 0.586、斑块 0.575、点蚀 0.537、夹杂物 0.429、压入氧化皮 0.232、龟裂 0.116。薄龟裂是明显短板，没有隐藏。

完整结果在 `results/yolo/metrics.json`，模型在 `models/neudet_yolo11n.pt`。

### 复现训练

NEU-DET 官方页面没有声明数据许可证，因此仓库**不重新分发图片**。研究使用请从官方页面下载并解压：

```bash
python -m fabqc.yolo.prepare_neudet \
  --raw /path/to/NEU-DET \
  --out data/processed/neudet

python -m fabqc.yolo.train \
  --data data/processed/neudet/data.yaml \
  --out runs/neudet_yolo \
  --epochs 20 --imgsz 256 --batch 16
```

Ultralytics 包使用 AGPL-3.0；如果做闭源商业产品，需要遵守 AGPL 或取得相容商业许可。

## 设备归因演习结果（seed=7）

模拟线上：决定给出原因时 97.5% 对；单故障从未指错原因；没有设备故障的划痕 100% 拒答。合成图上的 PatchCore AUROC 是 1.0，因为画上去的划痕太明显，**不能当成工厂成绩**。完整 JSON 在 `results/metrics.json`。
