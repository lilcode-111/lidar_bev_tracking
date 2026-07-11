import argparse

from bev_tracking.pipeline import format_eval_summary, run_kitti_bev_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KITTI BEV detections against LiDAR-frame GT boxes.")
    parser.add_argument("--data-root", default="data/kitti", help="KITTI root containing training folders.")
    parser.add_argument("--frame-id", default="000000", help="KITTI frame id, for example 000000.")
    parser.add_argument("--eps", type=float, default=0.6, help="Euclidean clustering radius in meters.")
    parser.add_argument("--min-points", type=int, default=20, help="Minimum points for a valid cluster.")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3, help="BEV NMS IoU threshold.")
    parser.add_argument("--eval-iou-threshold", type=float, default=0.25, help="BEV IoU threshold for TP matching.")
    parser.add_argument("--oriented", action="store_true", help="Use PCA-oriented clustering boxes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        report, output_path = run_kitti_bev_evaluation(
            data_root=args.data_root,
            frame_id=args.frame_id,
            eps=args.eps,
            min_points=args.min_points,
            oriented=args.oriented,
            nms_iou_threshold=args.nms_iou_threshold,
            eval_iou_threshold=args.eval_iou_threshold,
        )
    except FileNotFoundError as exc:
        print(exc)
        print("Expected layout: data/kitti/training/{velodyne,label_2,calib}/000000.*")
        raise SystemExit(1)

    print(format_eval_summary(report, output_path))
