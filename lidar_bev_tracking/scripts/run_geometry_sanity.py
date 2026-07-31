import argparse
import json
from pathlib import Path

from bev_tracking.geometry_sanity import (
    DEFAULT_CENTER_TOLERANCE_M,
    DEFAULT_YAW_TOLERANCE_RAD,
    build_geometry_sanity_report,
)
from bev_tracking.kitti import (
    load_kitti_labels,
    load_kitti_point_cloud,
    resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import load_kitti_calib


def parse_args():
    parser = argparse.ArgumentParser(description="Run KITTI camera/LiDAR geometry sanity checks for one frame.")
    parser.add_argument("--data-root", default="data/kitti", help="KITTI root containing training folders.")
    parser.add_argument("--frame-id", default="000000", help="KITTI frame id.")
    parser.add_argument("--output-dir", default="outputs/geometry_sanity", help="Directory for JSON reports.")
    parser.add_argument(
        "--center-tolerance-m",
        type=float,
        default=DEFAULT_CENTER_TOLERANCE_M,
        help="Maximum coordinate center round-trip error in meters.",
    )
    parser.add_argument(
        "--yaw-tolerance-rad",
        type=float,
        default=DEFAULT_YAW_TOLERANCE_RAD,
        help="Maximum yaw round-trip error in radians.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame_id = str(args.frame_id).zfill(6)
    velodyne_path, label_path = resolve_kitti_paths(args.data_root, frame_id)
    calib_path = resolve_kitti_calib_path(args.data_root, frame_id)

    points = load_kitti_point_cloud(velodyne_path)
    labels = load_kitti_labels(label_path)
    calib = load_kitti_calib(calib_path)
    report = build_geometry_sanity_report(
        points,
        labels,
        calib,
        frame_id=frame_id,
        center_tolerance_m=args.center_tolerance_m,
        yaw_tolerance_rad=args.yaw_tolerance_rad,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"geometry_sanity_{frame_id}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    round_trip = report["coordinate_round_trip"]
    print(f'frame: {report["frame_id"]}')
    print(f'geometry sanity passed: {report["passed"]}')
    print(f'coordinate max round-trip error: {round_trip["max_error_m"]:.8f} m')
    print(f'valid 3D boxes: {summary["num_valid_3d_boxes"]}')
    print(f'car boxes with points: {summary["car_boxes_with_points"]}/{summary["num_car_boxes"]}')
    print(f"saved {output_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
