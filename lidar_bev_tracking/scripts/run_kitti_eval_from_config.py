import argparse

from bev_tracking.config import load_yaml_config
from bev_tracking.pipeline import format_eval_summary, run_kitti_bev_evaluation_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run KITTI BEV evaluation from a YAML config.")
    parser.add_argument("--config", default="configs/kitti_eval.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)

    try:
        report, output_path = run_kitti_bev_evaluation_from_config(config)
    except FileNotFoundError as exc:
        print(exc)
        print("Expected layout: data/kitti/training/{velodyne,label_2,calib}/000000.*")
        raise SystemExit(1)

    print(f"loaded config: {args.config}")
    print(format_eval_summary(report, output_path))
