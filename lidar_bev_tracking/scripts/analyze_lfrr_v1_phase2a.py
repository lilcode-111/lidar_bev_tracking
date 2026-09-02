import argparse
import json

from bev_tracking.lfrr_v1_phase2a import run_phase2a_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run read-only LFRR-v1 Phase 2A checkpoint diagnostics."
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1",
        help="Frozen LFRR-v1 development artifacts and 25 checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    def progress(seed, fold):
        print(f"[checkpoint] seed={seed} fold={fold}", flush=True)

    report, result_path, records_path = run_phase2a_analysis(
        args.output_dir, progress_callback=progress
    )
    print(json.dumps({
        "checkpoint_execution": f"{report['checkpoint_count']}/25 SUCCESS",
        "SCORE_COLLAPSE_MODE": report["score_distribution"]["SCORE_COLLAPSE_MODE"],
        **{
            key: value for key, value in report["formal_answers"].items()
            if key != "decision_evidence"
        },
        "artifact_limitations": report["training_dynamics"]["artifact_limitations"],
        "RETRAINING": report["authorization"]["RETRAINING"],
        "OPTIMIZER_STEP_EXECUTED": report["authorization"]["OPTIMIZER_STEP_EXECUTED"],
        "INDEPENDENT_MANIFEST_STATUS": report["authorization"]["INDEPENDENT_MANIFEST_STATUS"],
    }, indent=2))
    print(f"saved {result_path}")
    print(f"saved {records_path}")
    print("PHASE 2A READ-ONLY ANALYSIS COMPLETE")
