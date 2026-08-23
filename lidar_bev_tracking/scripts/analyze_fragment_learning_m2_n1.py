import argparse
import json

from bev_tracking.fragment_learning_n1_degradation import analyze_m2_n1_degradation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only M2 N1 ranking-degradation diagnostic."
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1",
        help="Existing frozen dataset, split, M1/M2 OOF, and margin directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result, result_path, record_path = analyze_m2_n1_degradation(args.output_dir)
    compact = {
        "N1_score_movement": result["N1_score_movement"],
        "N1_materiality_continuum": result["N1_materiality_continuum"],
        "margin_label_relationship": result["margin_label_relationship"],
        "P_vs_N1_per_fold": result["P_vs_N1_per_fold"],
        "scene_concentration": result["scene_concentration"]["concentration"],
        "final_diagnosis": result["final_diagnosis"],
    }
    print(json.dumps(compact, indent=2))
    print(f"saved {result_path}")
    print(f"saved {record_path}")
    print("READ-ONLY N1 DIAGNOSTIC COMPLETE; NO MODEL WAS RETRAINED")
