import argparse
import copy
import json
from pathlib import Path

import yaml

from bev_tracking.adaptive_clustering import cluster_point_indices
from bev_tracking.adaptive_experiment import variant_specs_from_config
from bev_tracking.clustering_detector import split_obstacle_filter_stages_with_indices
from bev_tracking.kitti import (
    load_kitti_labels, load_kitti_point_cloud, resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib
from bev_tracking.o2_point_identity_fix import (
    build_o2_fix_result, correct_fragment_record,
    rebuild_day3_with_corrected_o2, write_sha256_sidecar,
)
from bev_tracking.report_writer import atomic_write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correct v15.3.2 O2 point identity without rerunning evaluation."
    )
    parser.add_argument("--source", required=True, help="Existing 15.3.2 result JSON.")
    parser.add_argument("--config", required=True, help="Frozen C0/C1/C2/C3 YAML.")
    parser.add_argument("--data-root", default="data/kitti/real_100")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def record_index(records):
    return {
        (str(item["frame_id"]).zfill(6), str(item["gt_id"])): item
        for item in records
    }


def main():
    args = parse_args()
    source = load_json(args.source)
    specs = variant_specs_from_config(load_yaml(args.config))
    c1 = specs["C1"]
    old_day2 = source["point_retention_day2"]
    delta = record_index(old_day2["delta_records"])
    controls = record_index(old_day2["p1_control_records"])
    overlap = sorted(set(delta) & set(controls))
    if overlap:
        raise ValueError(f"delta and P1 control cohorts overlap: {overlap}")
    selected = {**delta, **controls}
    selected_frames = sorted({key[0] for key in selected})
    corrected = {}

    for index, frame_id in enumerate(selected_frames, start=1):
        print(f"[C1 provenance {index:02d}/{len(selected_frames)}] replay {frame_id}")
        velodyne_path, label_path = resolve_kitti_paths(args.data_root, frame_id)
        calib_path = resolve_kitti_calib_path(args.data_root, frame_id)
        raw_points = load_kitti_point_cloud(velodyne_path)
        gt_boxes = kitti_labels_to_lidar_boxes(
            load_kitti_labels(label_path), load_kitti_calib(calib_path)
        )
        gt_by_id = {str(box["id"]): box for box in gt_boxes}
        stages, stage_indices = split_obstacle_filter_stages_with_indices(
            raw_points, z_min=c1.z_min, intensity_min=c1.intensity_min
        )
        local_indices = cluster_point_indices(stages["intensity_filter"], c1.policy)
        clusters = [stages["intensity_filter"][items] for items in local_indices]
        source_indices = [stage_indices["intensity_filter"][items] for items in local_indices]
        for key, old_record in selected.items():
            if key[0] != frame_id:
                continue
            if key[1] not in gt_by_id:
                raise ValueError(f"selected GT missing from frozen frame: {key}")
            corrected[key] = correct_fragment_record(
                old_record, gt_box=gt_by_id[key[1]], raw_points=raw_points,
                clusters=clusters, cluster_source_indices=source_indices,
            )

    if set(corrected) != set(selected):
        raise ValueError("provenance replay did not correct every selected GT")
    corrected_day2 = copy.deepcopy(old_day2)
    corrected_day2["point_identity"] = "raw_lidar_point_index"
    corrected_day2["provenance_only_replay"] = True
    corrected_day2["delta_records"] = [corrected[key] for key in delta]
    corrected_day2["p1_control_records"] = [corrected[key] for key in controls]
    corrected_day3 = rebuild_day3_with_corrected_o2(
        source["point_retention_day3"], corrected_day2
    )
    result = build_o2_fix_result(source, corrected_day2, corrected_day3)
    output_path = Path(args.output)
    atomic_write_json(output_path, result)
    digest, sidecar = write_sha256_sidecar(output_path)
    print(f"saved {output_path}")
    print(f"sha256 {digest}")
    print(f"saved {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
