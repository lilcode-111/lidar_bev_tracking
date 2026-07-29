# 第 15 轮 Day 1：FailureEvidence 基础

## 目标

Day 1 只建立可复现的诊断基础，不修改聚类、分类、PCA、NMS 或评测策略。

A0 默认过滤参数冻结为：

```yaml
detector:
  z_min: -0.9
  intensity_min: 0.38
```

## 诊断集

诊断集文件为 `configs/kitti_diagnostic_frames.txt`，包含 25 个互不重复的真实 KITTI 帧。

来源：

```text
run_id: 20260728T150755Z_kitti_car_batch_4d8c5173
primary_iou: 0.50
auxiliary_iou: 0.25
```

选择规则：

1. 每类 Failure Analysis top-5，按冻结类别顺序去重；
2. 加入 IoU=0.25 下存在 TP 的帧作为正向对照；
3. 按同一冻结排序从 rank 6 开始补充失败帧，直到 25 帧；
4. 最终 manifest 按 frame_id 升序保存。

诊断集只用于 FailureEvidence 和聚类方案选择，不作为正式收益指标。

## Primary reason 优先级

```text
NO_RAW_POINTS_IN_GT
REMOVED_BY_ROI
REMOVED_BY_Z_FILTER
REMOVED_BY_INTENSITY_FILTER
INSUFFICIENT_POINTS_FOR_CLUSTERING
CLUSTER_FRAGMENTATION
CLUSTER_MERGING
REJECTED_BY_CAR_CLASSIFICATION
REMOVED_BY_NMS
FINAL_IOU_BELOW_THRESHOLD
UNRESOLVED
```

一个 FN 只能有一个 primary reason；其他同时成立的现象放入 supporting flags。
