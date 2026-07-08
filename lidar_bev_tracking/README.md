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
- Generate a short multi-frame sequence.
- Associate detections across frames with BEV IoU.
- Maintain track ids and visualize trajectories.
- Detect objects directly from point clouds with a simple clustering baseline.
- Load KITTI Object Detection LiDAR `.bin` files and run the clustering detector on real point clouds.

## Quick Start

```bash
conda activate ad-perception
pip install -r requirements.txt

PYTHONPATH=src python scripts/generate_sample.py
PYTHONPATH=src python scripts/visualize_bev.py
PYTHONPATH=src python scripts/run_nms_demo.py
PYTHONPATH=src python scripts/run_tracking_demo.py
PYTHONPATH=src python scripts/run_clustering_detection_demo.py
PYTHONPATH=src python scripts/run_kitti_clustering_demo.py --frame-id 000000
```

## Outputs

```text
outputs/figures/bev_frame_000001.png
outputs/figures/nms_before.png
outputs/figures/nms_after.png
outputs/figures/tracking_frame_000000.png
outputs/figures/tracking_frame_000007.png
outputs/figures/clustering_detection.png
outputs/figures/kitti_clustering_000000.png
```

## KITTI Data Layout

Place KITTI Object Detection files under:

```text
data/kitti/
  training/
    velodyne/
      000000.bin
    label_2/
      000000.txt
```

`velodyne/*.bin` is required for the KITTI demo. `label_2/*.txt` is optional and is parsed for later evaluation/visualization work.

## Roadmap

- Add BEV detection metrics.
- Add Kalman Filter prediction.
- Add Hungarian matching.
- Replace axis-aligned cluster boxes with PCA-oriented boxes.
- Add KITTI label-based detection metrics.
- Integrate nuScenes-style data.
