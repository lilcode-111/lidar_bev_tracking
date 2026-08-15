import argparse
import json
import subprocess
from pathlib import Path

from bev_tracking.experiment_gate import load_a0_prime_declaration
from bev_tracking.failure_evidence_batch import load_diagnostic_manifest, run_kitti_diagnostic_failure_evidence
from bev_tracking.intensity_diagnostic import load_intensity_config
from bev_tracking.report_writer import atomic_write_json
from bev_tracking.v15_4_authorization import build_formal_run_authorization, validate_formal_run_authorization
from bev_tracking.v15_4_formal import build_formal_run_plan, build_matrix_identity_audit, require_passed_t0_gate, validate_both_t0_replays
from bev_tracking.v15_4_materialization import build_effective_config_matrix, build_t0_reference_artifacts, load_json
from bev_tracking.v15_4_low_memory_audit import build_low_memory_matrix_audit


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Authorize, inspect, or execute the frozen 15.4 experiment")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--authorize", action="store_true")
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--execute-t0", action="store_true")
    modes.add_argument("--execute-matrix", action="store_true")
    modes.add_argument("--resume-audit", action="store_true")
    parser.add_argument("--authorization", default="outputs/intensity_filter_ablation/pre_run/formal_run_authorization.json")
    parser.add_argument("--registry", default="configs/experiments/v15_4/t0_reference_registry.json")
    parser.add_argument("--base-config", default="configs/experiments/v15/i0_intensity_038.yaml")
    parser.add_argument("--output-dir", default="outputs/intensity_filter_ablation/formal/T0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = git(root, "rev-parse", "HEAD")
    clean = git(root, "status", "--porcelain") == ""
    if args.authorize:
        record = build_formal_run_authorization(repo_root=root, commit=commit, working_tree_clean=clean,
            schedule_path="configs/experiments/v15_4/threshold_schedule.json", gate_path="configs/experiments/v15_4/v15_4_release_gate.json",
            identity_path="configs/experiments/v15_4/pre_run_identity.json", reference_registry_path=args.registry)
        atomic_write_json(root / args.authorization, record)
        print(f"formal_run_authorized true\nformal_comparison_commit {commit}\nsaved {root / args.authorization}")
        return
    authorization = load_json(root / args.authorization)
    if args.resume_audit:
        if not clean:
            raise RuntimeError("working tree must be clean for recovery audit")
        matrix_root = (root / args.output_dir).parent
        report_paths = {
            "T0": matrix_root / "T0" / "t0_replay_report.json",
            "T1": matrix_root / "T1" / "formal_report.json",
            "T2": matrix_root / "T2" / "formal_report.json",
            "T_off": matrix_root / "T_off" / "formal_report.json",
        }
        audit = build_low_memory_matrix_audit(report_paths, matrix_root / "audit_cache", authorization["formal_comparison_commit"])
        audit["audit_recovery_commit"] = commit
        atomic_write_json(matrix_root / "matrix_identity_audit.json", audit)
        print(f"low-memory matrix audit PASS\nsaved {matrix_root / 'matrix_identity_audit.json'}")
        return
    validate_formal_run_authorization(authorization, current_commit=commit, working_tree_clean=clean, repo_root=root)
    schedule = load_json(root / "configs/experiments/v15_4/threshold_schedule.json")
    base_config = load_intensity_config(root / args.base_config)
    if args.plan_only:
        print(json.dumps(build_formal_run_plan(schedule, base_config, authorization), indent=2, sort_keys=True))
        return
    if args.execute_t0:
        execute_t0(root, args, commit, schedule, base_config)
    else:
        execute_matrix(root, args, commit, schedule, base_config)


def execute_t0(root, args, commit, schedule, base_config):
    identity = load_json(root / "configs/experiments/v15_4/pre_run_identity.json")
    registry = load_json(root / args.registry)
    config = build_effective_config_matrix(base_config, schedule)["T0"]
    manifest = load_diagnostic_manifest(root / identity["formal_100"]["manifest_path"])
    report = run_kitti_diagnostic_failure_evidence(data_root=config["data"]["root"], frame_ids=manifest["frame_ids"],
        eps=config["detector"]["eps"], min_points=config["detector"]["min_points"], oriented=config["detector"]["oriented"],
        z_min=config["detector"]["z_min"], intensity_min=config["detector"]["intensity_min"], nms_iou_threshold=config["nms"]["iou_threshold"],
        eval_iou_threshold=config["evaluation"]["iou_threshold"], auxiliary_iou_thresholds=tuple(config["evaluation"]["auxiliary_iou_thresholds"]),
        manifest_metadata=manifest, source_run_id=f"v15_4_formal_t0_{commit[:12]}", source_point_identity_gt_keys=identity["delta_22"]["ordered_identity_list"],
        progress_callback=lambda i, n, frame: print(f"[T0 {i:03d}/{n:03d}] {frame}"))
    output = root / args.output_dir
    atomic_write_json(output / "t0_replay_report.json", report)
    historical = load_json(root / identity["delta_22"]["source_artifact"]["path"])
    declaration = load_a0_prime_declaration(root / identity["t0_100_reference"]["declaration_path"], repo_root=root)
    replay_25, replay_100 = build_t0_reference_artifacts(report, historical, identity, declaration)
    atomic_write_json(output / "t0_25_replay.json", replay_25)
    atomic_write_json(output / "t0_100_replay.json", replay_100)
    reference_25 = load_json(root / registry["artifacts"]["t0_25_reference"]["path"])
    reference_100 = load_json(root / registry["artifacts"]["t0_100_reference"]["path"])
    gate = validate_both_t0_replays(reference_25, replay_25, reference_100, replay_100)
    atomic_write_json(output / "t0_replay_gate.json", gate)
    print(f"T0 replay PASS\nnon_t0_interpretation_allowed true\nsaved {output / 't0_replay_gate.json'}")


def execute_matrix(root, args, commit, schedule, base_config):
    t0_output = root / args.output_dir
    require_passed_t0_gate(load_json(t0_output / "t0_replay_gate.json"))
    identity = load_json(root / "configs/experiments/v15_4/pre_run_identity.json")
    manifest = load_diagnostic_manifest(root / identity["formal_100"]["manifest_path"])
    matrix = build_effective_config_matrix(base_config, schedule)
    reports = {"T0": load_json(t0_output / "t0_replay_report.json")}
    matrix_root = t0_output.parent
    for name in ("T1", "T2", "T_off"):
        config = matrix[name]
        report = run_kitti_diagnostic_failure_evidence(data_root=config["data"]["root"], frame_ids=manifest["frame_ids"],
            eps=config["detector"]["eps"], min_points=config["detector"]["min_points"], oriented=config["detector"]["oriented"],
            z_min=config["detector"]["z_min"], intensity_min=config["detector"]["intensity_min"], nms_iou_threshold=config["nms"]["iou_threshold"],
            eval_iou_threshold=config["evaluation"]["iou_threshold"], auxiliary_iou_thresholds=tuple(config["evaluation"]["auxiliary_iou_thresholds"]),
            manifest_metadata=manifest, source_run_id=f"v15_4_formal_{name.lower()}_{commit[:12]}", source_point_identity_gt_keys=identity["delta_22"]["ordered_identity_list"],
            progress_callback=lambda i, n, frame, variant=name: print(f"[{variant} {i:03d}/{n:03d}] {frame}"))
        reports[name] = report
        atomic_write_json(matrix_root / name / "formal_report.json", report)
    audit = build_matrix_identity_audit(reports)
    atomic_write_json(matrix_root / "matrix_identity_audit.json", audit)
    print(f"matrix execution complete\nsource_point_monotonicity PASS\nsaved {matrix_root / 'matrix_identity_audit.json'}")


if __name__ == "__main__":
    main()
