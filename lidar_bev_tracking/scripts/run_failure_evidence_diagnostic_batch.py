import argparse
from pathlib import Path

from bev_tracking.failure_evidence_batch import (
    load_diagnostic_manifest,
    run_kitti_diagnostic_failure_evidence,
)
from bev_tracking.kitti_yaw_validation import (
    DEFAULT_CORNER_TOLERANCE_M,
    DEFAULT_YAW_SEMANTIC_TOLERANCE_RAD,
)
from bev_tracking.report_writer import atomic_write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run FailureEvidence on the frozen KITTI diagnostic frame manifest.")
    parser.add_argument("--data-root", default="data/kitti/real_100")
    parser.add_argument("--manifest", default="configs/kitti_diagnostic_frames.txt")
    parser.add_argument("--output", default="outputs/failure_evidence/diagnostic_25_failure_evidence.json")
    parser.add_argument("--eps", type=float, default=0.6)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--z-min", type=float, default=-0.9)
    parser.add_argument("--intensity-min", type=float, default=0.38)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument("--center-tolerance-m", type=float, default=0.0001)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=0.0002)
    parser.add_argument(
        "--yaw-semantic-tolerance-rad",
        type=float,
        default=DEFAULT_YAW_SEMANTIC_TOLERANCE_RAD,
    )
    parser.add_argument("--corner-tolerance-m", type=float, default=DEFAULT_CORNER_TOLERANCE_M)
    parser.add_argument("--oriented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-run-id")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_diagnostic_manifest(args.manifest)
    report = run_kitti_diagnostic_failure_evidence(
        data_root=args.data_root,
        frame_ids=manifest["frame_ids"],
        eps=args.eps,
        min_points=args.min_points,
        oriented=args.oriented,
        z_min=args.z_min,
        intensity_min=args.intensity_min,
        nms_iou_threshold=args.nms_iou_threshold,
        eval_iou_threshold=args.eval_iou_threshold,
        center_tolerance_m=args.center_tolerance_m,
        yaw_tolerance_rad=args.yaw_tolerance_rad,
        yaw_semantic_tolerance_rad=args.yaw_semantic_tolerance_rad,
        corner_tolerance_m=args.corner_tolerance_m,
        manifest_metadata=manifest,
        source_run_id=args.source_run_id,
        progress_callback=lambda index, total, frame_id: print(f"[{index:02d}/{total:02d}] replay {frame_id}"),
    )

    output_path = Path(args.output)
    atomic_write_json(output_path, report)
    summary = report["summary"]
    print(f'geometry passed: {summary["geometry_passed_frames"]}/{summary["num_frames"]}')
    print(f'positive GT: {summary["num_positive_gt"]}')
    print(f'false negatives: {summary["num_false_negatives"]}')
    print(f'primary reasons: {summary["primary_reason_counts"]}')
    for metric_name, metric in summary["geometry_errors"].items():
        worst = metric["worst_object"]
        worst_id = f'{worst["frame_id"]}/{worst["gt_id"]}' if worst is not None else "none"
        print(
            f'{metric_name}: mean={metric["mean"]} max={metric["max"]} '
            f'tolerance={metric["tolerance"]} worst={worst_id}'
        )
    print(f"saved {output_path}")
    return 0 if summary["geometry_failed_frames"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
