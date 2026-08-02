# 第 15.1 轮 Day3：全 GT 候选覆盖证据

## 目标

Day2 已经证明 `intensity_min=0.38` 会大量删除 z-filter 后的点，但全局点数不能说明这些点是否真正帮助车辆 GT 形成候选框。

Day3 不修改检测、NMS 或评测算法，只补充所有 positive Car GT 的候选链路记录和汇总。

## 单个 GT 的候选链路

每个 positive GT 都记录：

```text
各过滤阶段框内点数
-> 关联 cluster
-> raw detection
-> NMS 前 Car candidate
-> NMS 后 Car candidate
-> 最佳 BEV IoU
-> 0.25 / 0.50 是否匹配
```

输出位于每帧 FailureEvidence 的：

```text
gt_candidate_records
```

该列表包含 TP 和 FN，避免只分析 FN 导致覆盖率缺少完整分母。

## GT candidate coverage

每帧和 25 帧汇总都输出：

```text
num_positive_gt
gt_with_cluster
gt_with_raw_detection
gt_with_car_detection_before_nms
gt_with_car_detection_after_nms
gt_with_best_iou_ge_0_10
gt_with_best_iou_ge_0_15
gt_with_best_iou_ge_0_25
gt_with_best_iou_ge_0_50
```

同时输出以 `num_positive_gt` 为分母的 ratios。

`zero_detection_with_gt_eligible_count` 在本轮定义为：没有任何关联的 NMS 后 Car candidate 的 positive GT 数量。它是 GT 级诊断量，不等同于第 14 轮的帧级 failure category。

## Candidate generation

算法阶段总量分开记录：

```text
cluster_count
raw_detection_count
car_candidate_count_before_nms
non_car_candidate_count
nms_suppressed_count
car_nms_suppressed_count
final_car_detection_count
neutralized_detection_count
effective_car_detection_count
```

其中当前聚类检测器对每个 cluster 固定生成一个 raw detection，因此自动检查：

```text
cluster_count == raw_detection_count
car candidate before NMS + non-Car candidate == raw detection
Car candidate before NMS - Car NMS suppressed == final Car detection
final Car detection == effective Car detection
```

## I0/I1 输出

重新运行原 Day2 命令后，`i0_i1_comparison.json` 新增：

```text
comparison.candidate_generation_totals
comparison.gt_candidate_coverage_counts
comparison.candidate_outcome_counts
comparison.zero_car_candidate_gt_count
```

Day3 先回答候选覆盖和候选生成总量。逐帧 Top 5、距离分段和 GT 级 I0/I1 增量连接放在 Day4。
