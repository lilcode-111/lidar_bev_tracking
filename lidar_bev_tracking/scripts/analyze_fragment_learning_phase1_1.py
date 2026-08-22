import argparse
import json

from bev_tracking.fragment_learning_failure_analysis import run_phase1_1_failure_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Run read-only Phase-1.1 OOF failure analysis.")
    parser.add_argument("--output-dir", default="outputs/fragment_learning_dev_v1")
    parser.add_argument(
        "--data-root",
        default="outputs/fragment_learning_dev_v1/selected_input_cache",
        help="Existing selected-64 KITTI cache; it is read only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result, result_path, pair_path, point_path = run_phase1_1_failure_analysis(
        args.output_dir, args.data_root
    )
    compact = {
        "group_counts": result["group_counts"],
        "scene_concentration": {
            key: result["scene_concentration"][key]
            for key in (
                "total_FP", "total_FN", "positive_support_frame_count",
                "recovered_positive_frame_count", "zero_recovery_frame_count",
                "zero_recovery_frames", "top_frame_concentration",
            )
        },
        "FP_CLUSTERING_BY_EXISTING_FEATURE": result["FP_CLUSTERING_BY_EXISTING_FEATURE"],
        "BAD_FOLD_DISTRIBUTION_SHIFT": result["BAD_FOLD_DISTRIBUTION_SHIFT"],
        "matched_pairs_summary": result["matched_pairs_summary"],
        "structural_review_status": result["structural_review_status"],
    }
    print(json.dumps(compact, indent=2))
    print(f"saved {result_path}")
    print(f"saved {pair_path}")
    print(f"saved {point_path}")
    print("READ-ONLY ANALYSIS COMPLETE; NO MODEL OR FORMAL DATA WAS MODIFIED")
