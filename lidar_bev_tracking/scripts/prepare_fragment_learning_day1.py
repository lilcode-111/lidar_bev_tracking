import argparse
import json

from bev_tracking.fragment_learning_dataset import (
    feature_schema_payload,
    select_fragment_learning_manifest,
    write_feature_schema,
    write_fragment_learning_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the frozen fragment_learning_dev_v1 manifest and schema."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fixed-100-manifest", default="configs/kitti_100_frames.txt")
    parser.add_argument(
        "--output-dir", default="outputs/fragment_learning_dev_v1"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frames = select_fragment_learning_manifest(
        args.data_root, args.fixed_100_manifest
    )
    manifest = write_fragment_learning_manifest(
        frames, f"{args.output_dir}/fragment_learning_dev_manifest.txt"
    )
    schema = write_feature_schema(
        f"{args.output_dir}/fragment_feature_schema_v1.json"
    )
    print("DATASET_CONSTRUCTION = DAY1_PREPARED")
    print(f"selected frames = {len(frames)}")
    print(f"saved {manifest}")
    print(f"saved {schema}")
    print(json.dumps(feature_schema_payload()["model_feature_fields"], indent=2))
