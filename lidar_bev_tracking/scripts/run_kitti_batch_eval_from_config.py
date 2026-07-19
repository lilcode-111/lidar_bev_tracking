import argparse
import sys

from bev_tracking.batch_pipeline import format_batch_report_summary, run_kitti_batch_report_from_config
from bev_tracking.config import load_yaml_config
from bev_tracking.report_writer import ReportWriteError


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-frame KITTI BEV evaluation and write a batch report.")
    parser.add_argument("--config", default="configs/kitti_eval_batch.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)
    command = " ".join(sys.argv)
    try:
        batch_result, paths = run_kitti_batch_report_from_config(config, config_input_path=args.config, command=command)
    except ReportWriteError as exc:
        print(f"report write failed: {exc.error_code.value}", file=sys.stderr)
        print(f"output path: {exc.output_path}", file=sys.stderr)
        print(f"cause: {exc.cause}", file=sys.stderr)
        raise SystemExit(1)

    print(f"loaded config: {args.config}")
    print(format_batch_report_summary(batch_result, paths))
