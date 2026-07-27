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

默认每类输出前 5 帧。每类使用冻结的多级排序规则，同分时最终按 `frame_id` 升序排序，保证重复运行结果稳定。筛选使用主指标 IoU=0.50，但每个 FailureCase 会同时保留 IoU=0.50 和 IoU=0.25 的完整指标，便于区分“完全漏检”和“定位不准”。

## 输出

结果会原子写入源 run 目录：

```text
outputs/kitti_batch_eval/<run_id>/failure_cases.json
```

报告采用 `14.0` schema，包含：

```text
schema_version / analysis_version
source / config / validation
category_definitions / generated_categories
categories
summary / generation
```

`categories` 中五类始终存在。每类分别记录：

```text
eligible_count
selected_count
sort_rule
diagnostic_only
cases
```

即使某一类没有候选帧，也会输出 `eligible_count=0`、`selected_count=0` 和 `cases=[]`。`summary` 同时记录 case 条目总数和去重后的失败帧数量，因为同一帧允许进入多个类别。

除 `generation.generated_at` 外，相同输入和配置重复运行会产生相同的业务内容。写入使用原子替换，失败时不会用残缺内容覆盖已有正式报告。
