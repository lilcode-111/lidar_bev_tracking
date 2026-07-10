import argparse
import json
from pathlib import Path

from bev_tracking.clustering_detector import detect_objects_from_points
from bev_tracking.evaluation import evaluate_detections
from bev_tracking.kitti import (
    load_kitti_labels,
    load_kitti_point_cloud,
    resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib
from bev_tracking.nms import nms_bev


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KITTI BEV detections against LiDAR-frame GT boxes.")
    parser.add_argument("--data-root", default="data/kitti", help="KITTI root containing training folders.")
    parser.add_argument("--frame-id", default="000000", help="KITTI frame id, for example 000000.")
    parser.add_argument("--eps", type=float, default=0.6, help="Euclidean clustering radius in meters.")
    parser.add_argument("--min-points", type=int, default=20, help="Minimum points for a valid cluster.")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3, help="BEV NMS IoU threshold.")
    parser.add_argument("--eval-iou-threshold", type=float, default=0.25, help="BEV IoU threshold for TP matching.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    velodyne_path, label_path = resolve_kitti_paths(args.data_root, args.frame_id)
    calib_path = resolve_kitti_calib_path(args.data_root, args.frame_id)

    try:
        points = load_kitti_point_cloud(velodyne_path)
        labels = load_kitti_labels(label_path)
        calib = load_kitti_calib(calib_path)
    except FileNotFoundError as exc:
        print(exc)
        print("Expected layout: data/kitti/training/{velodyne,label_2,calib}/000000.*")
        raise SystemExit(1)

    raw_detections = detect_objects_from_points(points, eps=args.eps, min_points=args.min_points)
    detections = nms_bev(raw_detections, iou_threshold=args.nms_iou_threshold)
    gt_boxes = kitti_labels_to_lidar_boxes(labels, calib)
    evaluation = evaluate_detections(detections, gt_boxes, iou_threshold=args.eval_iou_threshold)

    frame_id = str(args.frame_id).zfill(6)
    report = {
        "frame_id": frame_id,
        "num_points": len(points),
        "num_labels": len(labels),
        "num_gt_boxes": len(gt_boxes),
        "num_raw_detections": len(raw_detections),
        "num_detections_after_nms": len(detections),
        **evaluation,
    }

    output_path = report_dir / f"kitti_eval_{frame_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    metrics = report["metrics"]
    print(f"loaded points: {len(points)}")
    print(f"gt boxes in lidar frame: {len(gt_boxes)}")
    print(f"detections after nms: {len(detections)}")
    print(f'eval iou threshold: {report["iou_threshold"]:.2f}')
    print(f'tp={metrics["tp"]} fp={metrics["fp"]} fn={metrics["fn"]}')
    print(f'precision={metrics["precision"]:.3f} recall={metrics["recall"]:.3f}')
    print(f"saved {output_path}")
