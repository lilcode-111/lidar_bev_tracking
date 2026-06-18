# LiDAR BEV Tracking

基于 LiDAR BEV 表示的自动驾驶目标感知与多目标跟踪项目。

当前版本实现最小闭环：

- 生成模拟 LiDAR 点云
- 生成车辆、行人、锥桶目标框
- 构建 BEV 栅格图
- 在 BEV 图上绘制目标框
- 输出单帧可视化结果

后续会继续加入 BEV IoU、NMS、Kalman Filter、Hungarian Matching 和多目标跟踪。

## Quick Start

```bash
conda activate ad-perception
pip install -r requirements.txt

PYTHONPATH=src python scripts/generate_sample.py
PYTHONPATH=src python scripts/visualize_bev.py
```

输出图片：

```text
outputs/figures/bev_frame_000001.png
```


lidar_bev_tracking/
├── README.md
├── requirements.txt
├── data/
│   └── sample/
├── outputs/
│   └── figures/
├── scripts/
│   ├── generate_sample.py
│   └── visualize_bev.py
└── src/
    └── bev_tracking/
        ├── __init__.py
        ├── synthetic.py
        ├── bev.py
        └── visualization.py