# LiDAR BEV Tracking

A lightweight autonomous-driving perception project for learning LiDAR BEV representation, detection post-processing, and multi-object tracking.

## Current Features

- Generate synthetic LiDAR point clouds.
- Generate sample car, pedestrian, and cone boxes.
- Convert point clouds into a BEV intensity map.
- Visualize BEV boxes on top of the point cloud map.
- Generate noisy detection boxes.
- Run BEV IoU based NMS.
- Save before/after NMS visualization images.

## Quick Start

```bash
conda activate ad-perception
pip install -r requirements.txt

PYTHONPATH=src python scripts/generate_sample.py
PYTHONPATH=src python scripts/visualize_bev.py
PYTHONPATH=src python scripts/run_nms_demo.py
```

## Outputs

```text
outputs/figures/bev_frame_000001.png
outputs/figures/nms_before.png
outputs/figures/nms_after.png
```

## Roadmap

- Add BEV detection metrics.
- Add Kalman Filter prediction.
- Add Hungarian matching.
- Add track lifecycle management.
- Integrate real KITTI or nuScenes-style data.
