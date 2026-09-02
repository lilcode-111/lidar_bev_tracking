import argparse
import json

from bev_tracking.lfrr_v1_training import train_lfrr_v1_development


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run frozen LFRR-v1 5-seed x 5-fold development training."
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1",
        help="Frozen development dataset, split, M2 baseline and LFRR representation.",
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
    report, result_path, oof_path, frame_path = train_lfrr_v1_development(
        args.output_dir, progress_callback=show_progress
    )
    print(json.dumps({
        "model_execution": (
            f"{report['model_success_count']}/"
            f"{report['training_protocol']['model_count']} SUCCESS"
        ),
        "seed_metric_aggregate": report["seed_metric_aggregate"],
        "Development_Gates": {
            name: value["result"]
            for name, value in report["Development_Gates"].items()
        },
        "LFRR_V1_DEVELOPMENT_SIGNAL": report["LFRR_V1_DEVELOPMENT_SIGNAL"],
    }, indent=2))
    print(f"saved {result_path}")
    print(f"saved {oof_path}")
    print(f"saved {frame_path}")
    print("LFRR-v1 DEVELOPMENT COMPLETE; NO FINAL FIT OR INDEPENDENT EVALUATION PERFORMED")
