import argparse
import json

from bev_tracking.lfrr_v1_phase2b import train_target_only_control


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the frozen LFRR-v1 Phase 2B Target-Only Neural Control."
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1",
        help="Frozen 64-frame development artifacts.",
    )
    return parser.parse_args()


def show_progress(seed, fold, epoch, total, loss):
    print(
        f"[seed {seed} fold {fold}] epoch {epoch:03d}/{total:03d} "
        f"loss={loss:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    args = parse_args()
    report, result_path, oof_path = train_target_only_control(
        args.output_dir, progress_callback=show_progress
    )
    print(json.dumps({
        "CONTROL_CONTRACT_VALIDATOR": report["CONTROL_CONTRACT_VALIDATOR"]["result"],
        "model_execution": f"{report['model_success_count']}/25 SUCCESS",
        **report["aggregate"],
        "PRELIMINARY_CASE": report["PRELIMINARY_CASE"],
        "case_interpretation": report["case_interpretation"],
        "INDEPENDENT_MANIFEST_STATUS": report["INDEPENDENT_MANIFEST_STATUS"],
    }, indent=2))
    print(f"saved {result_path}")
    print(f"saved {oof_path}")
    print("PHASE 2B COMPLETE; NO OTHER CONTROL OR INDEPENDENT EVALUATION PERFORMED")
