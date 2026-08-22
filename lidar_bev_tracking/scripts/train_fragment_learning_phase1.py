import argparse
import json

from bev_tracking.fragment_learning_training import train_fragment_learning_phase1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the frozen L0/M1 5-fold grouped-OOF feasibility training."
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1",
        help="Existing frozen dataset/split directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result, result_path, oof_path = train_fragment_learning_phase1(args.output_dir)
    compact = {
        "M1": result["M1"], "L0": result["L0"],
        "positive_prevalence": result["positive_prevalence"],
        "M1_AP_minus_prevalence": result["M1_AP_minus_prevalence"],
        "M1_AP_minus_L0_AP": result["M1_AP_minus_L0_AP"],
        "M1_vs_L0_fold_win_count": result["M1_vs_L0_fold_win_count"],
        "P_vs_N0": result["subsets"]["P_vs_N0"],
        "P_vs_N1": result["subsets"]["P_vs_N1"],
        "Success_Gates": result["Success_Gates"],
        "LEARNED_FRAGMENT_RELATION": result["LEARNED_FRAGMENT_RELATION"],
    }
    print(json.dumps(compact, indent=2))
    print(f"saved {result_path}")
    print(f"saved {oof_path}")
    print("PHASE 1 COMPLETE; NO PARAMETER SEARCH OR FIXED-100 EXECUTION PERFORMED")
