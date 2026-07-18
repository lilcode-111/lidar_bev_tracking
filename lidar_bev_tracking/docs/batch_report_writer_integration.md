# Batch Report Writer Integration

第 13 轮 Day6 把 `report_writer` 接入 batch 配置入口。

之前的 batch 入口主要输出旧版文件：

```text
outputs/reports/kitti_batch_eval_oriented.json
outputs/reports/kitti_batch_eval_frames_oriented.csv
```

Day6 后推荐入口是：

```bash
PYTHONPATH=src python scripts/run_kitti_batch_eval_from_config.py --config configs/kitti_eval_batch.yaml
```

它会先读取 YAML 配置，再运行 batch 评测，最后调用 `write_batch_report()` 输出完整 run 目录：

```text
outputs/kitti_batch_eval/<run_id>/
  summary.json
  frames.csv
  config_input.yaml
  config_effective.json
  git.json
  frame_manifest.json
  frames/
    000000.json
    000001.json
```

`summary.json` 是总览入口，适合评审和自动化系统读取。

`frames.csv` 是逐帧表格，适合排查哪一帧成功、跳过、失败，以及每一帧的 TP / FP / FN。

`frames/*.json` 保留每一帧的完整结构化结果，后续可以继续做失败可视化、错误聚类和报告平台接入。
