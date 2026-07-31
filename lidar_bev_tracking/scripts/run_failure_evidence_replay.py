import argparse
import json
from pathlib import Path

from bev_tracking.failure_evidence import build_failure_evidence_report
from bev_tracking.kitti import (
    load_kitti_labels,
    load_kitti_point_cloud,
    resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib


def parse_args():
    parser = argparse.ArgumentParser(description="Replay one KITTI frame and generate per-FN failure evidence.")
    parser.add_argument("--data-root", default="data/kitti")
    parser.add_argument("--frame-id", default="000000")
    parser.add_argument("--eps", type=float, default=0.6)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--z-min", type=float, default=-0.9)
    parser.add_argument("--intensity-min", type=float, default=0.38)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument("--oriented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs/failure_evidence")
    parser.add_argument("--source-run-id")
    return parser.parse_args()


def main():
    args = parse_args()
    frame_id = str(args.frame_id).zfill(6)
    velodyne_path, label_path = resolve_kitti_paths(args.data_root, frame_id)
    calib_path = resolve_kitti_calib_path(args.data_root, frame_id)

    points = load_kitti_point_cloud(velodyne_path)
    labels = load_kitti_labels(label_path)
    calib = load_kitti_calib(calib_path)
    gt_boxes = kitti_labels_to_lidar_boxes(labels, calib)
    report = build_failure_evidence_report(
        points,
        gt_boxes,
        frame_id=frame_id,
        eps=args.eps,
        min_points=args.min_points,
        oriented=args.oriented,
        z_min=args.z_min,
        intensity_min=args.intensity_min,
        nms_iou_threshold=args.nms_iou_threshold,
        eval_iou_threshold=args.eval_iou_threshold,
        source_run_id=args.source_run_id,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"failure_evidence_{frame_id}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"frame: {frame_id}")
    print(f'positive GT: {summary["num_positive_gt"]}')
    print(f'false negatives: {summary["num_false_negatives"]}')
    print(f'primary reasons: {summary["primary_reason_counts"]}')
    print(f"saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
