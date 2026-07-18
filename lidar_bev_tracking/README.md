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
- Parse KITTI calibration files and overlay LiDAR-frame GT boxes with detections in BEV.
- Evaluate BEV detections with IoU matching, TP/FP/FN, precision, recall, and F1.
- Estimate PCA-oriented boxes for clustering-based LiDAR detections.
- Run KITTI BEV evaluation from YAML configs for reproducible experiments.
- Run multi-frame KITTI BEV batch evaluation and export summary JSON plus per-frame CSV.
- Evaluate Car-only BEV detections with explicit positive/neutral/excluded policy, primary IoU=0.5, auxiliary IoU=0.25, fixed neutral IoU=0.5, and deterministic det_index ordering.
- Write batch evaluation runs as reproducible report directories with summary JSON, per-frame CSV, config snapshots, Git metadata, frame manifest, and per-frame JSON.

## Quick Start

```bash
conda activate ad-perception
pip install -r requirements.txt

PYTHONPATH=src python scripts/generate_sample.py
PYTHONPATH=src python scripts/visualize_bev.py
PYTHONPATH=src python scripts/run_nms_demo.py
PYTHONPATH=src python scripts/run_tracking_demo.py
PYTHONPATH=src python scripts/run_clustering_detection_demo.py
PYTHONPATH=src python scripts/run_oriented_clustering_demo.py
PYTHONPATH=src python scripts/create_mini_kitti_sample.py --frame-id 000000
PYTHONPATH=src python scripts/run_kitti_clustering_demo.py --frame-id 000000
PYTHONPATH=src python scripts/run_kitti_gt_overlay_demo.py --frame-id 000000
PYTHONPATH=src python scripts/run_kitti_eval_demo.py --frame-id 000000
PYTHONPATH=src python scripts/run_kitti_eval_demo.py --frame-id 000000 --oriented
PYTHONPATH=src python scripts/run_kitti_eval_from_config.py --config configs/kitti_eval.yaml
PYTHONPATH=src python scripts/create_mini_kitti_sample.py --frame-id 000000 --num-frames 5
PYTHONPATH=src python scripts/run_kitti_batch_eval_from_config.py --config configs/kitti_eval_batch.yaml
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Outputs

```text
outputs/figures/bev_frame_000001.png
outputs/figures/nms_before.png
outputs/figures/nms_after.png
outputs/figures/tracking_frame_000000.png
outputs/figures/tracking_frame_000007.png
outputs/figures/clustering_detection.png
outputs/figures/oriented_clustering_detection.png
outputs/figures/kitti_clustering_000000.png
outputs/figures/kitti_gt_overlay_000000.png
outputs/reports/kitti_eval_000000_axis_aligned.json
outputs/reports/kitti_eval_000000_oriented.json
outputs/reports/kitti_batch_eval_oriented.json
outputs/reports/kitti_batch_eval_frames_oriented.csv
outputs/kitti_batch_eval/<run_id>/summary.json
outputs/kitti_batch_eval/<run_id>/frames.csv
outputs/kitti_batch_eval/<run_id>/frames/000000.json
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
    calib/
      000000.txt
```

`velodyne/*.bin` is required for the KITTI clustering demo. `label_2/*.txt` and `calib/*.txt` are required for GT overlay because KITTI labels are stored in camera coordinates and must be converted into the LiDAR frame before BEV visualization.

For a tiny smoke test without downloading KITTI, generate a synthetic KITTI-layout frame:

```bash
PYTHONPATH=src python scripts/create_mini_kitti_sample.py --frame-id 000000
PYTHONPATH=src python scripts/run_kitti_clustering_demo.py --frame-id 000000
PYTHONPATH=src python scripts/run_kitti_gt_overlay_demo.py --frame-id 000000
PYTHONPATH=src python scripts/run_kitti_eval_demo.py --frame-id 000000
PYTHONPATH=src python scripts/run_kitti_eval_demo.py --frame-id 000000 --oriented
PYTHONPATH=src python scripts/run_kitti_eval_from_config.py --config configs/kitti_eval.yaml
PYTHONPATH=src python scripts/create_mini_kitti_sample.py --frame-id 000000 --num-frames 5
PYTHONPATH=src python scripts/run_kitti_batch_eval_from_config.py --config configs/kitti_eval_batch.yaml
```

This only validates the file layout, reader path, simplified calibration parsing, GT overlay path, and BEV evaluation flow; it is not a real KITTI benchmark result.

## Evaluation Policy

The current evaluation policy is Car-only:

```text
positive GT: Car
neutral GT: Van, Truck
excluded GT: Pedestrian, Cyclist, Person_sitting, Tram, Misc
ROI: x=[0,40), y=[-20,20)
primary IoU: 0.5
auxiliary IoU: 0.25
neutral IoU: 0.5
```

`DontCare` labels are preserved by the parser and counted in raw labels, but entries without valid 3D boxes are not converted into BEV GT boxes. Missing label files raise `FileNotFoundError`; empty label files are valid zero-GT frames.

## Roadmap

- Add Kalman Filter prediction.
- Add Hungarian matching.
- Add KITTI label-based detection metrics.
- Integrate nuScenes-style data.
