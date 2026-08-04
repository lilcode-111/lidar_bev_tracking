# 第 15.2 轮 Day4：C0 replay gate

本日新增 C0 对账工具 `bev_tracking.c0_replay`。

它对同一帧分别运行：

1. legacy `euclidean_cluster()`；
2. fixed policy 的统一 `cluster_points()` 入口。

然后比较：

- filter stage point counts；
- intensity-filtered point hash；
- cluster signature、数量和顺序；
- raw detection、Car candidate、NMS 后数量；
- IoU=0.50/0.25 指标；
- GT candidate records；
- FN primary reason 和 supporting flags。

`replay_mismatch_count=0` 才表示 C0 通过。Day4 的单测使用 synthetic frame 验证门禁逻辑；真实 25 帧回放由用户在 WSL 中执行，避免把本地 KITTI 输出纳入代码包。

## 真实 25 帧回放

```bash
PYTHONPATH=src python scripts/run_c0_replay.py \
  --data-root data/kitti/real_100 \
  --manifest configs/kitti_diagnostic_frames.txt \
  --output outputs/clustering_diagnostic/c0_replay.json
```

通过标准：

```text
C0 replay mismatch count: 0
passed frames: 25/25
```
