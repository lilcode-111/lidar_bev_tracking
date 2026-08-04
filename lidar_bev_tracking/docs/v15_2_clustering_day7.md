# 第 15.2 轮 Day7：C0/C1/C2/C3 运行入口

本日新增 `scripts/run_clustering_ablation.py`，从显式 YAML 读取 C0/C1/C2/C3，按固定 manifest 逐帧运行，并输出：

- 预注册配置；
- eligible positive GT 一致性 gate；
- IoU=0.50 主指标；
- IoU=0.25 辅助指标；
- zero-detection、merging rate；
- 确定性 variant 排序。

运行方式：

```bash
PYTHONPATH=src python scripts/run_clustering_ablation.py \
  --config configs/experiments/v15/c123_preregistered.yaml \
  --data-root data/kitti/real_100 \
  --manifest configs/kitti_diagnostic_frames.txt \
  --output outputs/clustering_diagnostic/c0_c3_comparison.json
```

配置文件必须显式包含 `variants.C0`、`C1`、`C2`、`C3`。本日不提供未经评审确认的 C1/C2/C3 数值。
