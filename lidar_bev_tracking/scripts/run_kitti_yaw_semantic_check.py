import argparse
import json
from pathlib import Path

from bev_tracking.kitti import load_kitti_labels, resolve_kitti_calib_path, resolve_kitti_paths
from bev_tracking.kitti_calib import load_kitti_calib
from bev_tracking.kitti_yaw_validation import (
    DEFAULT_CORNER_TOLERANCE_M,
    DEFAULT_YAW_SEMANTIC_TOLERANCE_RAD,
    build_kitti_yaw_semantic_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate KITTI rotation_y against transformed box corners.")
    parser.add_argument("--data-root", default="data/kitti/real_100")
    parser.add_argument("--frame-id", default="000424")
    parser.add_argument("--output-dir", default="outputs/geometry_sanity")
    parser.add_argument("--yaw-tolerance-rad", type=float, default=DEFAULT_YAW_SEMANTIC_TOLERANCE_RAD)
    parser.add_argument("--corner-tolerance-m", type=float, default=DEFAULT_CORNER_TOLERANCE_M)
    return parser.parse_args()


def main():
    args = parse_args()
    frame_id = str(args.frame_id).zfill(6)
    _, label_path = resolve_kitti_paths(args.data_root, frame_id)
    calib_path = resolve_kitti_calib_path(args.data_root, frame_id)
    labels = load_kitti_labels(label_path)
    calib = load_kitti_calib(calib_path)
    report = build_kitti_yaw_semantic_report(
        labels,
        calib,
        frame_id=frame_id,
        yaw_tolerance_rad=args.yaw_tolerance_rad,
        corner_tolerance_m=args.corner_tolerance_m,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"kitti_yaw_semantics_{frame_id}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"frame: {report['frame_id']}")
    print(f"KITTI yaw semantics passed: {report['passed']}")
    print(f"boxes passed: {summary['num_passed_boxes']}/{summary['num_valid_boxes']}")
    print(f"max yaw semantic error: {summary['max_yaw_semantic_error_rad']:.8f} rad")
    print(f"max corner alignment error: {summary['max_corner_alignment_error_m']:.8f} m")
    print(f"saved {output_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
