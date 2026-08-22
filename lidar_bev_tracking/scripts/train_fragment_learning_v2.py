import argparse
import json

from bev_tracking.fragment_learning_v2_training import train_fragment_learning_v2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed one-feature LightGBM-v2 development confirmation."
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1",
        help="Existing frozen dataset, split, and M1 OOF directory.",
    )
    parser.add_argument(
        "--data-root",
        default="outputs/fragment_learning_dev_v1/selected_input_cache",
        help="Existing selected 64-frame KITTI cache used only for GT-free margin replay.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    def progress(index, total, frame_id):
        print(f"[margin {index:02d}/{total:02d}] {frame_id}", flush=True)

    result, result_path, oof_path, margin_path = train_fragment_learning_v2(
        args.output_dir, args.data_root, progress_callback=progress
    )
    compact = {
        "M1": result["M1"], "M2": result["M2"],
        "Delta_AP": result["Delta_AP"], "N0_FP": result["N0_FP"],
        "P_vs_N1": result["P_vs_N1"],
        "M2_AP_ge_M1_AP_fold_count": result["M2_AP_ge_M1_AP_fold_count"],
        "median_fold_Delta_AP": result["median_fold_Delta_AP"],
        "margin_coverage": result["margin_coverage"],
        "Development_Gates": result["Development_Gates"],
        "M2_DEVELOPMENT_CONFIRMATION": result["M2_DEVELOPMENT_CONFIRMATION"],
    }
    print(json.dumps(compact, indent=2))
    print(f"saved {result_path}")
    print(f"saved {oof_path}")
    print(f"saved {margin_path}")
    print("M2 DEVELOPMENT COMPLETE; NO FINAL FIT OR FIXED-100 EXECUTION PERFORMED")
