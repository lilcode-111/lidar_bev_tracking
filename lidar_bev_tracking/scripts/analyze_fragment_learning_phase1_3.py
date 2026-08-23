import argparse
import json

from bev_tracking.fragment_learning_neighbor_context import analyze_neighbor_context


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the read-only Phase-1.3 neighboring-fragment context analysis."
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1",
        help="Existing frozen dataset/OOF directory.",
    )
    parser.add_argument(
        "--data-root", default="outputs/fragment_learning_dev_v1/selected_input_cache",
        help="Existing selected 64-frame cache for deterministic graph replay.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    def progress(index, total, frame_id):
        print(f"[neighbor {index:02d}/{total:02d}] {frame_id}", flush=True)

    result, result_path, record_path = analyze_neighbor_context(
        args.output_dir, args.data_root, progress_callback=progress
    )
    compact = {
        "counts": result["counts"],
        "signal_families": result["signal_evaluation"]["families"],
        "Fold1_Fold5": {
            field: {
                "overall": item["overall"], "Fold1": item["folds"][1],
                "Fold5": item["folds"][5],
            }
            for field, item in result["signal_evaluation"]["fields"].items()
        },
        "singleton_counts": {
            group: item["sample_count"]
            for group, item in result["singleton_distributions"].items()
        },
        "frame_concentration": result["frame_concentration"]["top_frame_target_share"],
        "NEIGHBORING_FRAGMENT_CONTEXT_SIGNAL": result["NEIGHBORING_FRAGMENT_CONTEXT_SIGNAL"],
        "LOCAL_FRAGMENT_SUPPORT_COMPLEMENTARITY": result["LOCAL_FRAGMENT_SUPPORT_COMPLEMENTARITY"],
        "LOCAL_FRAGMENT_ISOLATION_CONTEXT": result["LOCAL_FRAGMENT_ISOLATION_CONTEXT"],
        "REPRESENTATION_DIAGNOSIS": result["REPRESENTATION_DIAGNOSIS"],
    }
    print(json.dumps(compact, indent=2))
    print(f"saved {result_path}")
    print(f"saved {record_path}")
    print("PHASE 1.3 READ-ONLY ANALYSIS COMPLETE; NO FEATURE OR MODEL WAS CHANGED")
