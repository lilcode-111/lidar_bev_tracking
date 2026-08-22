import argparse
import json

from bev_tracking.fragment_phase0 import run_fragment_phase0, write_phase0_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Run read-only Fragment Phase-0 analysis.")
    parser.add_argument("--data-root", default="data/kitti/real_100")
    parser.add_argument(
        "--phase2-gate-result",
        default="outputs/gesr_v1/phase2/phase2_gate_result.json",
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_geometry_recovery/phase0"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_fragment_phase0(
        args.data_root,
        args.phase2_gate_result,
        progress_callback=lambda index, total, frame_id, targets: print(
            f"[{index:02d}/{total:02d}] {frame_id} "
            + ",".join(item["gt_id"] for item in targets),
            flush=True,
        ),
    )
    paths = write_phase0_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], indent=2))
    print(f'saved {paths["summary"]}')
    print(f'saved {paths["fragments"]}')
    print(f'saved {paths["pairs"]}')
