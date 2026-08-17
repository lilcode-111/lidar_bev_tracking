import argparse
import json
from pathlib import Path

from bev_tracking.v15_5_seed_support import build_phase0_day1, build_phase0_day2, build_phase0_day3
from bev_tracking.v15_5_geometry_critical import build_geometry_critical_analysis


def build_parser():
    parser = argparse.ArgumentParser(description="Run the read-only 15.5 Phase-0 SSSR analysis")
    commands = parser.add_subparsers(dest="command", required=True)

    day1 = commands.add_parser("day1", help="Build frozen Band identities and point-level seed metrics")
    day1.add_argument("--output", default="outputs/seed_support_selectivity/v15_5_phase0_day1.json")
    day1.add_argument("--cache-dir", default="outputs/seed_support_selectivity/day1_cache")

    day2 = commands.add_parser("day2", help="Summarize VRR/BRR, distance strata, and frame stability")
    day2.add_argument("--day1", default="outputs/seed_support_selectivity/v15_5_phase0_day1.json")
    day2.add_argument("--output", default="outputs/seed_support_selectivity/v15_5_phase0_day2.json")

    day3 = commands.add_parser("day3", help="Compare delta-22 T0, seed-supported H/M, and T2 representations")
    day3.add_argument("--day1", default="outputs/seed_support_selectivity/v15_5_phase0_day1.json")
    day3.add_argument("--day2", default="outputs/seed_support_selectivity/v15_5_phase0_day2.json")
    day3.add_argument("--output", default="outputs/seed_support_selectivity/v15_5_phase0_day3.json")

    geometry = commands.add_parser("geometry-critical", help="Analyze frozen far selectivity and marginal geometry gain")
    geometry.add_argument("--day1", default="outputs/seed_support_selectivity/v15_5_phase0_day1.json")
    geometry.add_argument("--day2", default="outputs/seed_support_selectivity/v15_5_phase0_day2.json")
    geometry.add_argument("--day3", default="outputs/seed_support_selectivity/v15_5_phase0_day3.json")
    geometry.add_argument("--output", default="outputs/seed_support_selectivity/v15_5_phase0_geometry_critical.json")
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
    elif args.command == "day2":
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
    elif args.command == "day3":
        result = build_phase0_day3(
            repo_root=root,
            day1_path=args.day1,
            day2_path=args.day2,
            output_path=args.output,
        )
        summary = {
            "schema_version": result["schema_version"],
            "day3_complete": result["day3_complete"],
            "delta_22_summary": result["delta_22"]["summary"],
            "phase0_observation": result["phase0_observation"],
            "gt_oracle_leakage": result["gt_oracle_leakage"],
            "formal_pipeline_rerun": result["formal_pipeline_rerun"],
            "saved": args.output,
        }
    else:
        result = build_geometry_critical_analysis(
            repo_root=root,
            day1_path=args.day1,
            day2_path=args.day2,
            day3_path=args.day3,
            output_path=args.output,
        )
        marginal = result["dropped_vehicle_points"]["summary"]
        summary = {
            "schema_version": result["schema_version"],
            "far_H_M_frame_stability": {
                band: result["far_H_M_frame_stability"][band]["frame_stability"]
                for band in ("H", "M")
            },
            "T2_material_recovery_GT_range_distribution": result["T2_material_recovery_GT_range_distribution"],
            "far_uniform_vs_far_seed": result["far_uniform_vs_far_seed"]["modes"],
            "marginal_geometry_gain": {
                "G_p_distribution": marginal["G_p_distribution"],
                "runtime_feature_spearman_correlation": marginal["runtime_feature_spearman_correlation"],
                "offline_oracle_extent_spearman_correlation": marginal["offline_oracle_extent_spearman_correlation"],
                "top_G_cases_descriptive_only": marginal["top_G_cases_descriptive_only"],
            },
            "GT_oracle_leakage": result["GT_oracle_leakage"],
            "r_seed_search": result["r_seed_search"],
            "formal_pipeline_rerun": result["formal_pipeline_rerun"],
            "formal_results_modified": result["formal_results_modified"],
            "saved": args.output,
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
