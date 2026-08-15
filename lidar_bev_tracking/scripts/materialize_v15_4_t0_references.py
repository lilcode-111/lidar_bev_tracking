import argparse
from pathlib import Path

from bev_tracking.experiment_gate import load_a0_prime_declaration
from bev_tracking.failure_evidence_batch import (
    load_diagnostic_manifest,
    run_kitti_diagnostic_failure_evidence,
)
from bev_tracking.intensity_diagnostic import load_intensity_config
from bev_tracking.report_writer import atomic_write_json
from bev_tracking.v15_4_materialization import (
    build_effective_config_matrix,
    build_t0_reference_artifacts,
    load_json,
    raw_file_sha256,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Materialize frozen T0-only references for 15.4")
    parser.add_argument("--schedule", default="configs/experiments/v15_4/threshold_schedule.json")
    parser.add_argument("--identity", default="configs/experiments/v15_4/pre_run_identity.json")
    parser.add_argument("--t0-config", default="configs/experiments/v15/i0_intensity_038.yaml")
    parser.add_argument("--baseline-declaration", default="configs/experiments/v15/a0_prime_geometry_corrected.json")
    parser.add_argument("--historical-15-3-2", default="outputs/clustering_diagnostic/v15_3_2_o2_point_identity_corrected.json")
    parser.add_argument("--output-dir", default="outputs/intensity_filter_ablation/pre_run")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    schedule = load_json(args.schedule)
    identity = load_json(args.identity)
    historical = load_json(args.historical_15_3_2)
    declaration = load_a0_prime_declaration(args.baseline_declaration, repo_root=repo_root)
    base_config = load_intensity_config(args.t0_config)
    effective = build_effective_config_matrix(base_config, schedule)
    t0_config = effective["T0"]

    manifest = load_diagnostic_manifest(identity["formal_100"]["manifest_path"])
    if manifest["sha256"] != identity["formal_100"]["manifest_sha256"]:
        raise ValueError("formal 100-frame manifest hash differs from frozen identity")
    delta_keys = identity["delta_22"]["ordered_identity_list"]

    print("running T0-only 100-frame reference materialization")
    report = run_kitti_diagnostic_failure_evidence(
        data_root=t0_config["data"]["root"],
        frame_ids=manifest["frame_ids"],
        eps=t0_config["detector"]["eps"],
        min_points=t0_config["detector"]["min_points"],
        oriented=t0_config["detector"]["oriented"],
        z_min=t0_config["detector"]["z_min"],
        intensity_min=t0_config["detector"]["intensity_min"],
        nms_iou_threshold=t0_config["nms"]["iou_threshold"],
        eval_iou_threshold=t0_config["evaluation"]["iou_threshold"],
        auxiliary_iou_thresholds=tuple(t0_config["evaluation"]["auxiliary_iou_thresholds"]),
        manifest_metadata=manifest,
        source_run_id="v15_4_phase1_t0_reference",
        source_point_identity_gt_keys=delta_keys,
        progress_callback=lambda index, total, frame_id: print(
            f"[T0 {index:03d}/{total:03d}] {frame_id}"
        ),
    )
    output_dir = Path(args.output_dir)
    report_path = output_dir / "t0_100_materialization_report.json"
    t0_25_path = output_dir / "t0_25_reference.json"
    t0_100_path = output_dir / "t0_100_reference.json"
    atomic_write_json(report_path, report)
    print(f"saved diagnostic report before reference validation: {report_path}")
    t0_25, t0_100 = build_t0_reference_artifacts(report, historical, identity, declaration)
    atomic_write_json(t0_25_path, t0_25)
    atomic_write_json(t0_100_path, t0_100)
    index = {
        "schema_version": "15.4-t0-reference-index-v1",
        "status": "MATERIALIZED",
        "formal_variants_run": ["T0"],
        "non_t0_results_observed": False,
        "artifacts": {
            "t0_100_materialization_report": {"path": report_path.as_posix(), "sha256": raw_file_sha256(report_path)},
            "t0_25_reference": {"path": t0_25_path.as_posix(), "sha256": raw_file_sha256(t0_25_path)},
            "t0_100_reference": {"path": t0_100_path.as_posix(), "sha256": raw_file_sha256(t0_100_path)},
        },
        "formal_run_authorized": False,
        "next_required_step": "Freeze these artifact hashes and complete remaining Phase-1 validators before any non-T0 run.",
    }
    index_path = output_dir / "t0_reference_index.json"
    atomic_write_json(index_path, index)
    print(f"saved {t0_25_path}")
    print(f"saved {t0_100_path}")
    print(f"saved {index_path}")
    print("formal_run_authorized false")


if __name__ == "__main__":
    main()
