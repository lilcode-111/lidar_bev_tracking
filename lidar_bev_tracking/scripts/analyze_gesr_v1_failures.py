import argparse
import json
from pathlib import Path

from bev_tracking.gesr_v1_failure_analysis import run_failure_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only GESR-v1 analysis on T2 material-recovery delta-22 GTs."
    )
    parser.add_argument("--data-root", default="data/kitti/real_100")
    parser.add_argument(
        "--phase2-gate-result",
        default="outputs/gesr_v1/phase2/phase2_gate_result.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/gesr_v1/failure_analysis/t2_material_recovery_gt.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_failure_analysis(
        data_root=args.data_root,
        gate_result_path=args.phase2_gate_result,
        progress_callback=lambda index, total, frame_id, targets: print(
            f"[{index:02d}/{total:02d}] {frame_id} "
            + ",".join(target["gt_id"] for target in targets),
            flush=True,
        ),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    summary = result["summary"]
    print(f'target GTs: {summary["target_gt_count"]}')
    print(f'unique frames: {summary["unique_frame_count"]}')
    print(
        "T2-added GT points: "
        f'{summary["point_counts"].get("T2_added_GT", 0)}'
    )
    print(
        "GESR accepted / missed: "
        f'{summary["point_counts"].get("GESR_accepted_from_T2_added", 0)} / '
        f'{summary["point_counts"].get("GESR_missed_from_T2_added", 0)}'
    )
    print(
        "missed terminal reasons: "
        + json.dumps(summary["missed_terminal_reason_counts"], sort_keys=True)
    )
    print(f'dominant missed reason: {summary["dominant_missed_terminal_reason"]}')
    print(f'saved {output_path}')
