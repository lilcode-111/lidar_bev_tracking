import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a frozen read-only fragment-learning mechanism analysis."
    )
    parser.add_argument(
        "--phase",
        choices=("1.1", "1.2"),
        default="1.1",
        help="Analysis phase; existing Phase 1.1 behavior remains the default.",
    )
    parser.add_argument("--output-dir", default="outputs/fragment_learning_dev_v1")
    parser.add_argument(
        "--data-root",
        default="outputs/fragment_learning_dev_v1/selected_input_cache",
        help="Existing selected-64 KITTI cache; it is read only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "1.1":
        from bev_tracking.fragment_learning_failure_analysis import (
            run_phase1_1_failure_analysis,
        )

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
        paths = (result_path, pair_path, point_path)
    else:
        from bev_tracking.fragment_learning_multiseed_analysis import (
            run_multiseed_context_analysis,
        )

        result, result_path, record_path, evidence_path = run_multiseed_context_analysis(
            args.output_dir,
            args.data_root,
            progress_callback=lambda index, total, frame_id: print(
                f"[{index:02d}/{total:02d}] {frame_id}", flush=True
            ),
        )
        compact = {
            "group_counts": result["group_counts"],
            "valid_seed_relation_count": {
                group: result["group_distributions"][group]["distributions"]["valid_seed_relation_count"]
                for group in ("P_HIGH", "P_MISS", "N0_FP")
            },
            "singleton_sample_counts": {
                group: result["singleton_results"][group]["sample_count"]
                for group in ("P_HIGH", "P_MISS", "N0_FP")
            },
            "signal_families": result["cross_frame_signal"]["families"],
            "MULTI_SEED_CONTEXT_SIGNAL": result["MULTI_SEED_CONTEXT_SIGNAL"],
        }
        paths = (result_path, record_path, evidence_path)
    print(json.dumps(compact, indent=2))
    for path in paths:
        print(f"saved {path}")
    print("READ-ONLY ANALYSIS COMPLETE; NO MODEL OR FORMAL DATA WAS MODIFIED")
