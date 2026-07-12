# 第 11/12.1 轮：多帧 KITTI 批量评测

批量评测会逐帧调用单帧 evaluation pipeline，然后汇总整体结果。

```text
多个 frame_id
-> 逐帧调用单帧 evaluation pipeline
-> 汇总 TP / FP / FN / Precision / Recall / F1
-> 输出 JSON 汇总报告和 CSV 每帧明细
```

## 生成多帧 mini KITTI

```bash
PYTHONPATH=src python scripts/create_mini_kitti_sample.py --frame-id 000000 --num-frames 5
```

## 批量配置

配置文件：

```text
configs/kitti_eval_batch.yaml
```

核心内容：

```yaml
data:
  root: data/kitti
  frame_ids:
    - "000000"
    - "000001"
    - "000002"
    - "000003"
    - "000004"

detector:
  eps: 0.6
  min_points: 20
  oriented: true

nms:
  iou_threshold: 0.3

evaluation:
  iou_threshold: 0.5
  auxiliary_iou_thresholds:
    - 0.25
```

## 运行

```bash
PYTHONPATH=src python scripts/run_kitti_batch_eval_from_config.py --config configs/kitti_eval_batch.yaml
```

终端输出会包含主指标和每个 IoU 阈值的汇总：

```text
frames: 5
box mode: oriented_pca
total points: 41200
total gt boxes: 15
total detections after nms: 20
tp=10 fp=0 fn=0
precision=1.000 recall=1.000 f1=1.000
iou=0.50 tp=10 fp=0 fn=0 precision=1.000 recall=1.000 f1=1.000
iou=0.25 tp=10 fp=0 fn=0 precision=1.000 recall=1.000 f1=1.000
```

## JSON 输出

输出文件：

```text
outputs/reports/kitti_batch_eval_oriented.json
```

关键结构：

```json
{
  "metrics_by_iou": {
    "0.50": {
      "tp": 10,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0
    },
    "0.25": {
      "tp": 10,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0
    }
  }
}
```

## CSV 输出

输出文件：

```text
outputs/reports/kitti_batch_eval_frames_oriented.csv
```

CSV 会把每帧的双阈值指标展开：

```text
frame_id
tp_iou_0_50
fp_iou_0_50
fn_iou_0_50
precision_iou_0_50
recall_iou_0_50
f1_iou_0_50
tp_iou_0_25
fp_iou_0_25
fn_iou_0_25
precision_iou_0_25
recall_iou_0_25
f1_iou_0_25
```

## 说明

单帧 report 中本来就有：

```text
metrics            # 主 IoU=0.50
auxiliary["0.25"]  # 辅助 IoU=0.25
```

第 12.1 轮把 batch summary 和 CSV 也补齐为双阈值输出，避免只在单帧 JSON 里能看到辅助指标。
