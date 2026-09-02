import argparse
import json

from bev_tracking.lfrr_v1_validators import run_pretraining_validators


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all five mandatory LFRR-v1 pre-training hard validators."
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1",
        help="Frozen development artifacts plus completed LFRR-v1 Day-1 representation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report, path = run_pretraining_validators(args.output_dir)
    print(json.dumps({
        name: item["result"] for name, item in report["validators"].items()
    }, indent=2))
    print(f"TRAINING_READINESS = {report['TRAINING_READINESS']}")
    print("TRAINING_STARTED = false")
    print("OPTIMIZER_STEP_EXECUTED = false")
    print(f"saved {path}")
    if report["TRAINING_READINESS"] != "PASS":
        raise SystemExit(1)
