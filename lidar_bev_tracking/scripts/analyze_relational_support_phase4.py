import argparse
import json

from bev_tracking.relational_support_phase4 import analyze_relational_support


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run read-only Phase-4 relational support coverage statistics."
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1",
        help="Existing frozen 64-frame development artifact directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result, result_path, group_path, sample_path = analyze_relational_support(args.output_dir)
    print(json.dumps({
        "relational_group_counts": result["relational_group_counts"],
        "mixed_group_support_overall": result["P_N1_mixed_group_support"]["overall"],
        "cross_frame_fold_support": result["cross_frame_fold_support"],
        "supervised_targets_per_group": result["supervised_targets_per_group"],
        "local_set_cardinality": result["local_set_cardinality"],
        "context_label_composition": result["context_label_composition_offline_diagnostic"],
        "RELATIONAL_SUPPORT_OBSERVATION": result["RELATIONAL_SUPPORT_OBSERVATION"],
        "INDEPENDENT_MANIFEST_STATUS": result["INDEPENDENT_MANIFEST_STATUS"],
    }, indent=2))
    print(f"saved {result_path}")
    print(f"saved {group_path}")
    print(f"saved {sample_path}")
    print("PHASE 4 READ-ONLY STATISTICS COMPLETE; NO MODEL TRAINING PERFORMED")
