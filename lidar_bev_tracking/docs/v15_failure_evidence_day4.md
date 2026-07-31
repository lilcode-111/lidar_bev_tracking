# 第 15 轮 Day4：25 帧诊断集与低 IoU 几何证据

## 目标

Day4 不改变检测算法，只将单帧 FailureEvidence 扩展到冻结的25帧诊断集，并为
`FINAL_IOU_BELOW_THRESHOLD` 保存最佳候选框及几何偏差：

```text
dx / dy
BEV center error
length error
width error
yaw error (mod pi)
```

PCA 框旋转180度时 BEV 几何等价，因此 yaw error 按 `pi` 周期计算。

## 运行

```bash
PYTHONPATH=src python scripts/run_failure_evidence_diagnostic_batch.py \
  --data-root data/kitti/real_100 \
  --manifest configs/kitti_diagnostic_frames.txt \
  --source-run-id 20260728T150755Z_kitti_car_batch_4d8c5173
```

输出：

```text
outputs/failure_evidence/diagnostic_25_failure_evidence.json
```

报告同时保存每帧几何 sanity、每个 FN 的完整证据、唯一主原因汇总和低 IoU
候选框的几何误差摘要。该文件是本地实验产物，不提交到 Git。
