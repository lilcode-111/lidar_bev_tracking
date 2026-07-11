import argparse

from bev_tracking.batch_pipeline import format_batch_summary, run_kitti_batch_evaluation_from_config
from bev_tracking.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-frame KITTI BEV evaluation from a YAML config.")
    parser.add_argument("--config", default="configs/kitti_eval_batch.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)

    try:
        summary, summary_path, csv_path = run_kitti_batch_evaluation_from_config(config)
    except FileNotFoundError as exc:
        print(exc)
        print("Expected layout: data/kitti/training/{velodyne,label_2,calib}/000000.*")
        raise SystemExit(1)

    print(f"loaded config: {args.config}")
    print(format_batch_summary(summary, summary_path, csv_path))
