import argparse
import json

from bev_tracking.fragment_learning_construction import (
    prepare_and_construct_archive_dataset,
    prepare_and_construct_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct fragment_learning_dev_v1 without model training."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data-root",
        help="KITTI root containing training/velodyne, label_2, and calib.",
    )
    source.add_argument(
        "--archive-dir",
        help="Directory containing the three official KITTI object ZIP files.",
    )
    parser.add_argument(
        "--fixed-100-manifest", default="configs/kitti_100_frames.txt"
    )
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1"
    )
    parser.add_argument(
        "--cache-root",
        default="outputs/fragment_learning_dev_v1/selected_input_cache",
        help="WSL-local cache used only with --archive-dir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.archive_dir:
        summary, summary_path = prepare_and_construct_archive_dataset(
            args.archive_dir,
            args.fixed_100_manifest,
            args.output_dir,
            args.cache_root,
            staging_progress_callback=lambda index, total, frame_id: print(
                f"[extract {index:02d}/{total:02d}] {frame_id}", flush=True
            ),
            construction_progress_callback=lambda index, total, frame_id: print(
                f"[build {index:02d}/{total:02d}] {frame_id}", flush=True
            ),
        )
    else:
        summary, summary_path = prepare_and_construct_dataset(
            args.data_root,
            args.fixed_100_manifest,
            args.output_dir,
            progress_callback=lambda index, total, frame_id: print(
                f"[build {index:02d}/{total:02d}] {frame_id}", flush=True
            ),
        )
    print(json.dumps(summary, indent=2))
    print(f"saved {summary_path}")
    print("MODEL TRAINING = NOT PERFORMED")
