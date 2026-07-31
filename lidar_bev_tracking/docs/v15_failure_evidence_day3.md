# 第 15 轮 Day3：单帧 FailureEvidence Replay

## 目标

Day3 将单帧 Car 漏检沿检测链路回放，并为每个 FN 生成一条结构化证据：

```text
raw GT points
-> ROI
-> z filter
-> intensity filter
-> clustering
-> Car classification
-> NMS
-> final IoU
```

每个 FN 只有一个 `primary_reason`，其他同时出现的现象放入
`supporting_flags`，避免失败原因重复统计。

当过滤前 GT 框内点数不低于 `min_points`，过滤后点数降到
`min_points` 以下，并且最终没有关联 cluster 时，主原因归属于造成门槛
跨越的 ROI、z 或 intensity 过滤阶段；`insufficient_points_for_clustering`
仅作为下游 supporting flag 保留。

## 几何检查修正

中心和 yaw 不再共用一个无单位阈值：

```text
center_tolerance_m = 0.0001 m
yaw_tolerance_rad = 0.0002 rad
```

运行：

```bash
PYTHONPATH=src python scripts/run_geometry_sanity.py \
  --data-root data/kitti/real_100 \
  --frame-id 000424 \
  --center-tolerance-m 0.0001 \
  --yaw-tolerance-rad 0.0002
```

## 单帧失败证据回放

```bash
PYTHONPATH=src python scripts/run_failure_evidence_replay.py \
  --data-root data/kitti/real_100 \
  --frame-id 000424 \
  --oriented
```

输出：

```text
outputs/failure_evidence/failure_evidence_000424.json
```

该输出是本地诊断产物，不应提交到 Git。

## 当前边界

Day3 冻结 detector、NMS 和 Evaluation Policy，只增加诊断能力。聚类分裂和
粘连采用保守证据：同一 GT 与多个 cluster 相交视为分裂候选，同一 cluster
同时包含多个 positive GT 的点视为粘连候选。后续诊断集运行会验证这些规则。
