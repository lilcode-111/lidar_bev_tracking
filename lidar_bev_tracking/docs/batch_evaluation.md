# 第 11 轮：多帧 KITTI 批量评测

第 10 轮已经支持通过 YAML 跑单帧 KITTI BEV 评测。第 11 轮在这个基础上新增批量评测能力：

```text
多个 frame_id
-> 逐帧调用单帧 evaluation pipeline
-> 汇总 TP / FP / FN
-> 计算整体 precision / recall
-> 输出 JSON 汇总报告和 CSV 每帧明细
```

## 为什么要做批量评测

真实感知评测不会只看一帧。单帧结果只能说明链路能跑通，但不能说明算法稳定性。批量评测可以观察一组数据上的整体效果，比如：

```text
总共多少 GT
总共多少检测框
整体 TP / FP / FN
整体 precision / recall
每一帧分别表现如何
```

这让项目更接近真实工程中的感知评测工具。

## 生成多帧 mini KITTI

当前没有真实 KITTI 数据时，可以先用 synthetic 数据生成 KITTI 目录格式的小样本：

```bash
PYTHONPATH=src python scripts/create_mini_kitti_sample.py --frame-id 000000 --num-frames 5
```

它会生成：

```text
data/kitti/
  training/
    velodyne/
      000000.bin
      000001.bin
      000002.bin
      000003.bin
      000004.bin
    label_2/
      000000.txt
      000001.txt
      000002.txt
      000003.txt
      000004.txt
    calib/
      000000.txt
      000001.txt
      000002.txt
      000003.txt
      000004.txt
```

注意：这只是 smoke test，用来验证批量评测链路，不是真实 KITTI benchmark。

## 批量配置

批量配置在：

```text
configs/kitti_eval_batch.yaml
```

核心是 `frame_ids`：

```yaml
data:
  root: data/kitti
  frame_ids:
    - "000000"
    - "000001"
    - "000002"
    - "000003"
    - "000004"
```

检测、NMS、评价参数仍然和单帧配置一致：

```yaml
detector:
  eps: 0.6
  min_points: 20
  oriented: true

nms:
  iou_threshold: 0.3

evaluation:
  iou_threshold: 0.25
```

## 运行批量评测

```bash
PYTHONPATH=src python scripts/run_kitti_batch_eval_from_config.py --config configs/kitti_eval_batch.yaml
```

终端输出类似：

```text
loaded config: configs/kitti_eval_batch.yaml
frames: 5
box mode: oriented_pca
total points: 41200
total gt boxes: 15
total detections after nms: 20
tp=15 fp=5 fn=0
precision=0.750 recall=1.000
saved outputs/reports/kitti_batch_eval_oriented.json
saved outputs/reports/kitti_batch_eval_frames_oriented.csv
```

## 输出文件

```text
outputs/reports/kitti_batch_eval_oriented.json
outputs/reports/kitti_batch_eval_frames_oriented.csv
```

JSON 是整体汇总，CSV 是每帧明细，方便后续做错误分析和参数对比。

第 11 轮的核心意义是：

```text
从“单帧可运行”升级为“多帧可评测、结果可汇总、实验可对比”。
```
