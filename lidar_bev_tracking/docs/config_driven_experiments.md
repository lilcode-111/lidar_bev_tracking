# Config Driven Experiments

第十轮目标是把 KITTI BEV 评价流程配置化。

之前运行评价时，需要在命令行里手动传参数：

```bash
PYTHONPATH=src python scripts/run_kitti_eval_demo.py --frame-id 000000 --oriented
```

第十轮新增 YAML 配置文件：

```text
configs/kitti_eval.yaml
```

示例：

```yaml
data:
  root: data/kitti
  frame_id: "000000"

detector:
  eps: 0.6
  min_points: 20
  oriented: true

nms:
  iou_threshold: 0.3

evaluation:
  iou_threshold: 0.5
  auxiliary_iou_thresholds:
    - 0.25

outputs:
  report_dir: outputs/reports
```

运行配置化评价：

```bash
PYTHONPATH=src python scripts/run_kitti_eval_from_config.py --config configs/kitti_eval.yaml
```

新增模块：

```text
src/bev_tracking/config.py
```

负责读取 YAML，并和默认配置合并。

```text
src/bev_tracking/pipeline.py
```

负责组织完整评价流程：

```text
读取 KITTI 点云 / label / calib
-> 聚类检测
-> NMS
-> GT label 转 LiDAR box
-> BEV IoU 匹配评价
-> 保存 JSON report
```

这样脚本层只负责入口，核心流程放在 `src/bev_tracking` 模块里复用。

第十轮的价值不是新增算法，而是让实验更可复现：

```text
同一个 config
-> 同一套检测参数
-> 同一套评价阈值
-> 同一份输出报告
```

后续参数扫描可以继续基于这个配置结构扩展。
