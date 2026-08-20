import argparse
from copy import deepcopy
import sys

from bev_tracking.batch_pipeline import format_batch_report_summary, run_kitti_batch_report_from_config
from bev_tracking.config import load_yaml_config
from bev_tracking.report_writer import ReportWriteError


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-frame KITTI BEV evaluation and write a batch report.")
    parser.add_argument("--config", default="configs/kitti_eval_batch.yaml", help="Path to YAML config.")
    parser.add_argument("--variant", choices=("T0", "T2", "GESR-v1"), help="Frozen GESR-v1 Phase-2 variant.")
    return parser.parse_args()


def apply_phase2_variant(config, variant):
    if variant is None:
        return config
    effective = deepcopy(config)
    detector = effective["detector"]
    detector["intensity_min"] = 0.15 if variant == "T2" else 0.38
    detector["gesr_enabled"] = variant == "GESR-v1"
    detector["gesr_reason_attribution"] = True
    effective["phase2"] = {"variant": variant}
    effective.setdefault("outputs", {})["batch_report_root"] = f"outputs/gesr_v1/phase2/{variant}"
    return effective


if __name__ == "__main__":
    args = parse_args()
    config = apply_phase2_variant(load_yaml_config(args.config), args.variant)
    command = " ".join(sys.argv)
    progress_label = args.variant or "BATCH"
    try:
        batch_result, paths = run_kitti_batch_report_from_config(
            config,
            config_input_path=args.config,
            command=command,
            progress_callback=lambda index, total, frame_id: print(
                f"[{progress_label} {index:02d}/{total:02d}] {frame_id}",
                flush=True,
            ),
        )
    except ReportWriteError as exc:
        print(f"report write failed: {exc.error_code.value}", file=sys.stderr)
        print(f"output path: {exc.output_path}", file=sys.stderr)
        print(f"cause: {exc.cause}", file=sys.stderr)
        raise SystemExit(1)

    print(f"loaded config: {args.config}")
    print(format_batch_report_summary(batch_result, paths))
