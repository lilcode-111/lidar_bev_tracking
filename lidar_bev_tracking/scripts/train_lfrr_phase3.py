import argparse
import json

from bev_tracking.lfrr_phase3 import train_phase3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run frozen Phase-3 M2-anchored incremental context development."
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1",
        help="Frozen development dataset and representation directory.",
    )
    return parser.parse_args()


def progress(*values):
    if values[0] == "M2_OUTER":
        print(f"[M2 outer {values[1]}/5]", flush=True)
    elif values[0] == "M2_PAIR":
        print(f"[M2 pair-exclusion {values[1]}/10]", flush=True)
    else:
        _, seed, fold, epoch, total, loss = values
        print(
            f"[context seed {seed} fold {fold}] epoch {epoch:03d}/{total:03d} "
            f"loss={loss:.6f}", flush=True,
        )


if __name__ == "__main__":
    args = parse_args()
    result, result_path, oof_path, anchor_path = train_phase3(
        args.output_dir, progress_callback=progress
    )
    print(json.dumps({
        "M2_replica_execution": result["M2_replica_execution"],
        "pretrain_validators": result["pretrain_validators"],
        "PHASE3_TRAINING_READINESS": result["PHASE3_TRAINING_READINESS"],
        "context_model_execution": result["context_model_execution"],
        "Development_Gates": {
            name: value["result"] for name, value in result["Development_Gates"].items()
        },
        "PHASE3_DEVELOPMENT_SIGNAL": result["PHASE3_DEVELOPMENT_SIGNAL"],
        "INDEPENDENT_MANIFEST_STATUS": result["INDEPENDENT_MANIFEST_STATUS"],
    }, indent=2))
    print(f"saved {result_path}")
    print(f"saved {oof_path}")
    print(f"saved {anchor_path}")
    print("PHASE 3 DEVELOPMENT COMPLETE; NO FINAL FIT OR INDEPENDENT EVALUATION PERFORMED")
