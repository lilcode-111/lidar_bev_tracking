import argparse
import json
from pathlib import Path

from bev_tracking.fragment_learning_construction import collect_complete_archive_frame_ids
from bev_tracking.lfrr_v1 import (
    build_lfrr_development_representation,
    register_independent_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build frozen LFRR-v1 Day-1 representation and register identities only."
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1",
        help="Existing frozen fragment_learning_dev_v1 artifact directory.",
    )
    parser.add_argument(
        "--data-root",
        default="outputs/fragment_learning_dev_v1/selected_input_cache",
        help="Existing selected 64-frame cache used only for deterministic runtime replay.",
    )
    parser.add_argument(
        "--archive-dir", required=True,
        help="Directory containing official KITTI training archives; only entry names are read.",
    )
    parser.add_argument(
        "--learning-manifest",
        default="outputs/fragment_learning_dev_v1/fragment_learning_dev_manifest.txt",
    )
    parser.add_argument("--fixed100-manifest", default="configs/kitti_100_frames.txt")
    parser.add_argument(
        "--phase0-evidence",
        default="outputs/fragment_geometry_recovery/phase0/fragment_gt_pairs.csv",
        help="Frozen Phase-0 evidence containing exactly its 9 unique frame identities.",
    )
    parser.add_argument(
        "--independent-manifest",
        default="configs/lfrr_independent_confirm_v1_manifest.txt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    def progress(index, total, frame_id):
        print(f"[LFRR Day1 {index:02d}/{total:02d}] {frame_id}", flush=True)

    representation, summary_path = build_lfrr_development_representation(
        args.output_dir, args.data_root, progress_callback=progress
    )
    complete_ids = collect_complete_archive_frame_ids(args.archive_dir)
    independent = register_independent_manifest(
        complete_ids,
        args.learning_manifest,
        args.fixed100_manifest,
        args.phase0_evidence,
        args.independent_manifest,
    )
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    summary["independent_manifest_registration"] = independent
    Path(summary_path).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "representation": representation,
        "independent_manifest_registration": independent,
    }, indent=2))
    print(f"saved {summary_path}")
    print("LFRR-v1 DAY 1 COMPLETE")
    print("TRAINING_STARTED = false")
    print("OPTIMIZER_STEP_EXECUTED = false")
    print("INDEPENDENT_RESULTS_OBSERVED = false")
