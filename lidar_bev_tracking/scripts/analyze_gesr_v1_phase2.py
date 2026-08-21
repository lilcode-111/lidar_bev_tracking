import argparse
import json
from pathlib import Path

from bev_tracking.gesr_phase2_gate import analyze_phase2_runs


def parse_args():
    parser = argparse.ArgumentParser(description="Compute frozen GESR-v1 Phase-2 Gate0 and Gate A.")
    parser.add_argument("--t0-run", required=True)
    parser.add_argument("--t2-run", required=True)
    parser.add_argument("--gesr-run", required=True)
    parser.add_argument(
        "--identity",
        default="configs/experiments/v15_4/pre_run_identity.json",
    )
    parser.add_argument(
        "--gate",
        default="configs/experiments/v15_5/gesr_v1/gesr_v1_release_gate.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/gesr_v1/phase2/phase2_gate_result.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = analyze_phase2_runs(
        {"T0": args.t0_run, "T2": args.t2_run, "GESR-v1": args.gesr_run},
        identity_path=args.identity,
        gate_path=args.gate,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    metrics = result["Gate_A"]["metrics"]
    print(f'actual commit: {result["actual_commit"]}')
    print(f'Gate0_Phase2: {result["Gate0_Phase2"]["result"]}')
    print(f'Gate A: {result["Gate_A"]["result"]}')
    print(f'material recovery: {metrics["material_recovery_count"]} / 22')
    print(f'median IoU gain: {metrics["median_iou_gain"]}')
    print(
        "IoU >= 0.25: "
        f'T0={metrics["T0_iou_ge_0_25_count"]} '
        f'GESR={metrics["GESR_iou_ge_0_25_count"]}'
    )
    print(f'material regression: {metrics["material_regression_count"]}')
    print(f'recovery frames: {metrics["material_recovery_frame_count"]}')
    print(f'saved {output_path}')
