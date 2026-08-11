import argparse
import json
from pathlib import Path

import yaml

from bev_tracking.adaptive_experiment import (
    aggregate_variant_reports,
    build_variant_diagnostics,
    preregister_variant_specs,
    rank_variant_summaries,
    run_variant_frame,
    validate_variant_batch_results,
    variant_specs_from_config,
)
from bev_tracking.candidate_conversion import build_candidate_conversion_delta
from bev_tracking.cluster_separability import (
    build_cluster_feature_day2,
    build_cluster_group_day1,
    build_cluster_separability_day3,
)
from bev_tracking.review_supplement import build_review_supplement_day2
from bev_tracking.point_retention import (
    build_point_retention_day1,
    build_point_retention_day2,
    build_point_retention_day3,
)
from bev_tracking.failure_evidence_batch import load_diagnostic_manifest
from bev_tracking.kitti import (
    load_kitti_labels,
    load_kitti_point_cloud,
    resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib
from bev_tracking.report_writer import atomic_write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run preregistered C0/C1/C2/C3 clustering ablation.")
    parser.add_argument("--config", required=True, help="YAML containing variants.C0/C1/C2/C3.")
    parser.add_argument("--data-root", default="data/kitti/real_100")
    parser.add_argument("--manifest", default="configs/kitti_diagnostic_frames.txt")
    parser.add_argument("--output", default="outputs/clustering_diagnostic/c0_c3_comparison.json")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    specs = variant_specs_from_config(config)
    preregistration = preregister_variant_specs(specs)
    manifest = load_diagnostic_manifest(args.manifest)
    reports_by_variant = {name: [] for name in specs}
    total = len(manifest["frame_ids"])

    for variant_name, variant in specs.items():
        for index, frame_id in enumerate(manifest["frame_ids"], start=1):
            print(f"[{variant_name} {index:02d}/{total}] replay {frame_id}")
            velodyne_path, label_path = resolve_kitti_paths(args.data_root, frame_id)
            calib_path = resolve_kitti_calib_path(args.data_root, frame_id)
            points = load_kitti_point_cloud(velodyne_path)
            labels = load_kitti_labels(label_path)
            calib = load_kitti_calib(calib_path)
            gt_boxes = kitti_labels_to_lidar_boxes(labels, calib)
            reports_by_variant[variant_name].append(
                run_variant_frame(
                    points,
                    gt_boxes,
                    frame_id,
                    variant,
                    oriented=True,
                    nms_iou_threshold=0.3,
                    eval_iou_threshold=0.5,
                    auxiliary_iou_thresholds=(0.25,),
                )
            )

    gate = validate_variant_batch_results(reports_by_variant)
    summaries = [
        aggregate_variant_reports(name, reports)
        for name, reports in reports_by_variant.items()
    ]
    cluster_separability_day1 = build_cluster_group_day1(reports_by_variant)
    cluster_separability_day2 = build_cluster_feature_day2(cluster_separability_day1)
    cluster_separability_diagnostic = build_cluster_separability_day3(
        cluster_separability_day1, cluster_separability_day2
    )
    point_retention_day1 = build_point_retention_day1(reports_by_variant)
    point_retention_day2 = build_point_retention_day2(point_retention_day1, reports_by_variant)
    point_retention_day3 = build_point_retention_day3(
        point_retention_day1, point_retention_day2, reports_by_variant
    )
    output = {
        "schema_version": "15.2-ablation",
        "source": {
            "data_root": str(args.data_root),
            "manifest": manifest,
        },
        "preregistration": preregistration,
        "gates": gate,
        "summaries": summaries,
        "ranking": rank_variant_summaries(summaries),
        "diagnostics": {
            name: build_variant_diagnostics(reports)
            for name, reports in reports_by_variant.items()
        },
        "delta_cohort": build_candidate_conversion_delta(reports_by_variant),
        "cluster_separability_diagnostic": cluster_separability_diagnostic,
        "point_retention_day1": point_retention_day1,
        "point_retention_day2": point_retention_day2,
        "point_retention_day3": point_retention_day3,
        "review_supplement": build_review_supplement_day2(reports_by_variant),
    }
    output_path = Path(args.output)
    atomic_write_json(output_path, output)
    print(f"eligible GT gate passed: {gate['passed']}")
    print(f"saved {output_path}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
