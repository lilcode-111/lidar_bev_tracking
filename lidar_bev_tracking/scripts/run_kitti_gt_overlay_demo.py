import argparse
from pathlib import Path

import cv2

from bev_tracking.bev import points_to_bev
from bev_tracking.clustering_detector import detect_objects_from_points
from bev_tracking.kitti import (
    load_kitti_labels,
    load_kitti_point_cloud,
    resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib
from bev_tracking.nms import nms_bev
from bev_tracking.visualization import draw_detections_and_gt


def parse_args():
    parser = argparse.ArgumentParser(description="Draw KITTI detections and GT boxes in LiDAR BEV.")
    parser.add_argument("--data-root", default="data/kitti", help="KITTI root containing training folders.")
    parser.add_argument("--frame-id", default="000000", help="KITTI frame id, for example 000000.")
    parser.add_argument("--eps", type=float, default=0.6, help="Euclidean clustering radius in meters.")
    parser.add_argument("--min-points", type=int, default=20, help="Minimum points for a valid cluster.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="BEV NMS IoU threshold.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    detections = nms_bev(raw_detections, iou_threshold=args.iou_threshold)
    gt_boxes = kitti_labels_to_lidar_boxes(labels, calib)

    bev = points_to_bev(points)
    image = draw_detections_and_gt(bev, detections, gt_boxes)

    frame_id = str(args.frame_id).zfill(6)
    output_path = output_dir / f"kitti_gt_overlay_{frame_id}.png"
    cv2.imwrite(str(output_path), image)

    print(f"loaded points: {len(points)}")
    print(f"loaded labels: {len(labels)}")
    print(f"gt boxes in lidar frame: {len(gt_boxes)}")
    print(f"raw detections: {len(raw_detections)}")
    print(f"detections after nms: {len(detections)}")
    print(f"saved {output_path}")
