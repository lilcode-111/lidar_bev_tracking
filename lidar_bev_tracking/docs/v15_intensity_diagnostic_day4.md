# 第 15.1 轮 Day4：I0/I1 失败证据对比

## 目标

Day4 读取 Day3 已生成的 I0/I1 JSON，不重新运行检测、聚类、NMS 或评测。它将同一帧和同一 GT 的记录一一连接，形成正式评审需要的逐帧、分距离和 GT 级增量证据。

## 逐帧分析

对 IoU=0.50 和 IoU=0.25 分别统计：

```text
TP增加、不变、下降帧数
FP增加、不变、下降帧数
TP增益Top 5
FP增长Top 5
effective Car detection增长Top 5
```

同时输出新增 fragmentation、merging 和 `FINAL_IOU_BELOW_THRESHOLD` 集中的帧。所有 Top 排序使用变化量降序、`frame_id` 升序兜底。

## 距离分析

距离固定为：

```text
near：[0,15)
mid：[15,30)
far：[30,+∞)
```

每个距离段分别输出 GT 数量、双 IoU TP/FN、cluster coverage、Car candidate coverage 和 zero-Car-candidate GT。

## GT级增量证据

使用 `(frame_id, gt_id)` 连接 I0/I1，逐 GT 输出：

```text
I1恢复的框内点数
是否新形成cluster
是否新形成最终Car candidate
best IoU变化
IoU=0.50/0.25下是否FN转TP
candidate outcome状态迁移
```

连接时自动检查 I0/I1 的 raw、ROI、z-filter 框内点数、距离和距离段保持不变。

## 使用现有报告生成Day4结果

```bash
PYTHONPATH=src python scripts/build_intensity_diagnostic_analysis.py
```

该命令读取：

```text
outputs/intensity_diagnostic/i0_intensity_038.json
outputs/intensity_diagnostic/i1_intensity_000.json
```

并更新：

```text
outputs/intensity_diagnostic/i0_i1_comparison.json
```

原有 manifest、config difference、I0 reproduction 和 invariant 门禁会再次执行。
