# Failure Analysis Core

第 14 轮将第 13 轮可信的 batch 报告转换为稳定、可追溯的失败样本清单，不重新运行 detection、NMS 或 evaluation。

## 运行

```bash
PYTHONPATH=src python scripts/run_failure_analysis.py \
  --run-dir outputs/kitti_batch_eval/<run_id> \
  --top-k 5
```

程序会读取并交叉校验：

```text
summary.json
frames.csv
frame_manifest.json
frames/*.json
```

帧集合、状态、`metric_valid`、逐帧报告路径或主 IoU 指标不一致时，程序会拒绝分析并返回非零退出码。

## 分类

```text
most_false_negatives
most_false_positives
lowest_recall
zero_detection_with_gt
highest_effective_car_detections
```

前四类是失败样本，`highest_effective_car_detections` 只作为诊断候选。正式 detection count 定义为：

```text
effective_car_detection_count = TP + FP + neutralized_detections @ IoU=0.50
```

默认每类输出前 5 帧。同分时最终按 `frame_id` 升序排序，保证重复运行结果稳定。

## 输出

结果会原子写入源 run 目录：

```text
outputs/kitti_batch_eval/<run_id>/failure_cases.json
```

报告包含源 run、`top_k`、类别计数和完整 FailureCase 列表。写入失败时不会用残缺内容覆盖已有正式报告。
