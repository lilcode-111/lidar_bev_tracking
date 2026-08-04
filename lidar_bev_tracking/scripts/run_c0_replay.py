import argparse
from pathlib import Path

from bev_tracking.c0_replay import aggregate_c0_replay, run_c0_frame_replay
from bev_tracking.failure_evidence_batch import load_diagnostic_manifest
from bev_tracking.kitti import load_kitti_labels, load_kitti_point_cloud, resolve_kitti_paths
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib
from bev_tracking.report_writer import atomic_write_json
from bev_tracking.kitti import resolve_kitti_calib_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed C0 clustering replay gate.")
    parser.add_argument("--data-root", default="data/kitti/real_100")
    parser.add_argument("--manifest", default="configs/kitti_diagnostic_frames.txt")
    parser.add_argument("--output", default="outputs/clustering_diagnostic/c0_replay.json")
    parser.add_argument("--eps", type=float, default=0.6)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--z-min", type=float, default=-0.9)
    parser.add_argument("--intensity-min", type=float, default=0.38)
    parser.add_argument("--oriented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument("--source-run-id")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_diagnostic_manifest(args.manifest)
    frame_reports = []
    total = len(manifest["frame_ids"])

    for index, frame_id in enumerate(manifest["frame_ids"], start=1):
        print(f"[C0 {index:02d}/{total}] replay {frame_id}")
        velodyne_path, label_path = resolve_kitti_paths(args.data_root, frame_id)
        calib_path = resolve_kitti_calib_path(args.data_root, frame_id)
        points = load_kitti_point_cloud(velodyne_path)
        labels = load_kitti_labels(label_path)
        calib = load_kitti_calib(calib_path)
        gt_boxes = kitti_labels_to_lidar_boxes(labels, calib)
        frame_reports.append(
            run_c0_frame_replay(
                points,
                gt_boxes,
                frame_id,
                eps=args.eps,
                min_points=args.min_points,
                oriented=args.oriented,
                z_min=args.z_min,
                intensity_min=args.intensity_min,
                nms_iou_threshold=args.nms_iou_threshold,
                eval_iou_threshold=args.eval_iou_threshold,
                auxiliary_iou_thresholds=(0.25,),
                source_run_id=args.source_run_id,
            )
        )

    report = aggregate_c0_replay(
        frame_reports,
        manifest_metadata=manifest,
        source_run_id=args.source_run_id,
    )
    output_path = Path(args.output)
    atomic_write_json(output_path, report)
    summary = report["summary"]
    print(f"C0 replay mismatch count: {summary['replay_mismatch_count']}")
    print(f"passed frames: {summary['passed_frames']}/{summary['num_frames']}")
    print(f"saved {output_path}")
    return 0 if summary["replay_mismatch_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
