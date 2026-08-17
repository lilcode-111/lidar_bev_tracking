import argparse
import json
from pathlib import Path

from bev_tracking.v15_5_seed_support import build_phase0_day1, build_phase0_day2


def build_parser():
    parser = argparse.ArgumentParser(description="Run the read-only 15.5 Phase-0 SSSR analysis")
    commands = parser.add_subparsers(dest="command", required=True)

    day1 = commands.add_parser("day1", help="Build frozen Band identities and point-level seed metrics")
    day1.add_argument("--output", default="outputs/seed_support_selectivity/v15_5_phase0_day1.json")
    day1.add_argument("--cache-dir", default="outputs/seed_support_selectivity/day1_cache")

    day2 = commands.add_parser("day2", help="Summarize VRR/BRR, distance strata, and frame stability")
    day2.add_argument("--day1", default="outputs/seed_support_selectivity/v15_5_phase0_day1.json")
    day2.add_argument("--output", default="outputs/seed_support_selectivity/v15_5_phase0_day2.json")
    return parser


def main():
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "day1":
        result = build_phase0_day1(
            repo_root=root,
            output_path=args.output,
            cache_dir=args.cache_dir,
        )
        summary = {
            "schema_version": result["schema_version"],
            "day1_complete": result["day1_complete"],
            "identity_validation": result["identity_validation"],
            "band_totals": result["band_totals"],
            "gt_oracle_leakage": result["contracts"]["gt_oracle_leakage"],
            "formal_pipeline_rerun": result["formal_pipeline_rerun"],
            "saved": args.output,
        }
    else:
        result = build_phase0_day2(
            repo_root=root,
            day1_path=args.day1,
            output_path=args.output,
        )
        summary = {
            "schema_version": result["schema_version"],
            "day2_complete": result["day2_complete"],
            "primary_H_plus_M": result["primary_H_plus_M"],
            "frame_stability": result["frame_stability"],
            "descriptive_observation": result["descriptive_observation"],
            "gt_oracle_leakage": result["gt_oracle_leakage"],
            "formal_pipeline_rerun": result["formal_pipeline_rerun"],
            "saved": args.output,
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
