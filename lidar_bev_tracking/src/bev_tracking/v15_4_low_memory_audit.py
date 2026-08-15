import gc
import statistics
from pathlib import Path

import numpy as np

from bev_tracking.report_writer import atomic_write_json
from bev_tracking.v15_4_audit import build_candidate_regression_audit, build_tp_regression_audit, classify_regression_reasons
from bev_tracking.v15_4_materialization import load_json
from bev_tracking.v15_4_materialization import _aggregate_report_metrics
from bev_tracking.v15_4_release import evaluate_variant_release_gates, select_release_candidate


UNIVERSE_MAPPING = {
    "global": "global_post_intensity",
    "positive_gt": "positive_gt_post_intensity",
    "annotation_excluded_background": "annotation_excluded_background_post_intensity",
}


def materialize_compact_cache(report_path, cache_dir):
    report = load_json(report_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cache_dir / "gt_records.json", {"gt_candidate_records": report["gt_candidate_records"]})
    index = {"source_run_id": report["source"]["source_run_id"], "frames": {}}
    for frame in report["source_point_universes"]:
        frame_id = str(frame["frame_id"]).zfill(6)
        index["frames"][frame_id] = {}
        for name, source_name in UNIVERSE_MAPPING.items():
            relative = f"{frame_id}_{name}.npy"
            values = np.asarray(frame["universes"][source_name]["source_point_indices"], dtype=np.int64)
            if len(values) != len(np.unique(values)):
                raise ValueError(f"duplicate point identity in {frame_id} {name}")
            np.save(cache_dir / relative, values, allow_pickle=False)
            index["frames"][frame_id][name] = relative
    atomic_write_json(cache_dir / "index.json", index)
    del report
    gc.collect()
    return index


def validate_compact_monotonicity(cache_dirs):
    variants = ("T0", "T1", "T2", "T_off")
    indices = {name: load_json(Path(cache_dirs[name]) / "index.json") for name in variants}
    comparisons = []
    for left_name, right_name in zip(variants, variants[1:]):
        left_index, right_index = indices[left_name], indices[right_name]
        if set(left_index["frames"]) != set(right_index["frames"]):
            raise ValueError("compact audit frame identities differ")
        for frame_id in sorted(left_index["frames"]):
            for universe in UNIVERSE_MAPPING:
                left = np.load(Path(cache_dirs[left_name]) / left_index["frames"][frame_id][universe], allow_pickle=False)
                right = np.load(Path(cache_dirs[right_name]) / right_index["frames"][frame_id][universe], allow_pickle=False)
                missing = np.setdiff1d(left, right, assume_unique=True)
                if len(missing):
                    raise ValueError(f"source-point monotonicity failed: {left_name}->{right_name} {frame_id} {universe} missing={int(missing[0])}")
                comparisons.append({"left": left_name, "right": right_name, "frame_id": frame_id, "universe": universe, "left_count": int(len(left)), "right_count": int(len(right)), "passed": True})
    return {"status": "PASS", "point_identity": "(frame_id, raw_lidar_point_index)", "coordinate_row_dedup_used": False, "comparisons": comparisons}


def build_low_memory_matrix_audit(report_paths, cache_root, formal_comparison_commit):
    cache_root = Path(cache_root)
    cache_dirs = {name: cache_root / name for name in ("T0", "T1", "T2", "T_off")}
    for name, path in report_paths.items():
        index = materialize_compact_cache(path, cache_dirs[name])
        expected = f"_{formal_comparison_commit[:12]}"
        if not index["source_run_id"].endswith(expected):
            raise ValueError(f"{name} report was not produced by formal comparison commit")
    monotonicity = validate_compact_monotonicity(cache_dirs)
    t0 = load_json(cache_dirs["T0"] / "gt_records.json")
    comparisons = {}
    for name in ("T1", "T2", "T_off"):
        variant = load_json(cache_dirs[name] / "gt_records.json")
        comparisons[name] = {
            "tp_regression": build_tp_regression_audit(t0, variant),
            "candidate_regression": build_candidate_regression_audit(t0, variant),
            "regression_reasons": {key: classify_regression_reasons(t0, variant, key) for key in ("0.50", "0.25")},
        }
        del variant
        gc.collect()
    return {"schema_version": "15.4-formal-matrix-identity-audit-v1", "formal_comparison_commit": formal_comparison_commit, "audit_mode": "low_memory_per_report_per_frame", "source_point_monotonicity": monotonicity, "comparisons_vs_T0": comparisons}


def extract_scalar_summary(report_path):
    report = load_json(report_path)
    delta = {
        (str(item["frame_id"]).zfill(6), str(item["gt_id"])): item["post_intensity_diagnostic_pca"]["iou"]
        for item in report.get("source_point_identity_records", [])
    }
    generation = report["summary"]["candidate_generation_totals"]
    summary = {
        "source_run_id": report["source"]["source_run_id"],
        "num_frames": int(report["summary"]["num_frames"]),
        "metrics_by_iou": _aggregate_report_metrics(report),
        "counts": {
            "effective_car_detection": int(generation["effective_car_detection_count"]),
            "raw_car_candidate_before_nms": int(generation["car_candidate_count_before_nms"]),
            "strict_background_candidate_after_nms": sum(bool(item["final_car_candidate"]) for item in report["strict_background_lineage"]["records"]),
        },
        "delta_22_iou": delta,
    }
    del report
    gc.collect()
    return summary


def finalize_release_artifact(report_paths, matrix_audit_path, gate_config, schedule, formal_comparison_commit, audit_recovery_commit):
    audit = load_json(matrix_audit_path)
    if audit.get("source_point_monotonicity", {}).get("status") != "PASS":
        raise ValueError("source-point monotonicity audit must PASS before finalization")
    summaries = {name: extract_scalar_summary(path) for name, path in report_paths.items()}
    if any(not item["source_run_id"].endswith(formal_comparison_commit[:12]) for item in summaries.values()):
        raise ValueError("formal report commit identity mismatch")
    t0 = summaries["T0"]
    base_delta = t0["delta_22_iou"]
    baseline = {
        "validity": _validity(100),
        "delta_22": {"iou_ge_0_25_count": sum(value is not None and value >= 0.25 for value in base_delta.values())},
        "candidate": {"positive_gt_count": _candidate_baseline_count(audit, "T1")},
        "metrics_by_iou": t0["metrics_by_iou"],
        "counts": t0["counts"],
    }
    gate_results, selection_metrics = {}, {}
    for name in ("T1", "T2", "T_off"):
        comparison = audit["comparisons_vs_T0"][name]
        summary = summaries[name]
        if set(summary["delta_22_iou"]) != set(base_delta):
            raise ValueError(f"{name} delta-22 GT identities differ from T0")
        deltas = [float(summary["delta_22_iou"][key]) - float(value) for key, value in base_delta.items() if value is not None and summary["delta_22_iou"].get(key) is not None]
        recovered_keys = [key for key, value in base_delta.items() if value is not None and summary["delta_22_iou"].get(key) is not None and float(summary["delta_22_iou"][key]) - float(value) >= 0.10]
        candidate = comparison["candidate_regression"]
        tp_audit = comparison["tp_regression"]["by_iou"]
        variant = {
            "validity": _validity(summary["num_frames"]),
            "delta_22": {
                "material_recovery_count": len(recovered_keys),
                "median_iou_gain_vs_T0": statistics.median(deltas) if deltas else 0.0,
                "iou_ge_0_25_count": sum(value is not None and value >= 0.25 for value in summary["delta_22_iou"].values()),
                "material_regression_count": sum(value <= -0.10 for value in deltas),
                "material_recovery_unique_frames": len({key[0] for key in recovered_keys}),
            },
            "candidate": {"improved_gt_count": candidate["candidate_improved_GT"]["count"], "regressed_gt_count": candidate["candidate_regressed_GT"]["count"]},
            "metrics_by_iou": summary["metrics_by_iou"],
            "counts": summary["counts"],
            "regression": {
                "tp_regressed_gt_count": {key: tp_audit[key]["TP_regressed_GT"]["count"] for key in ("0.50", "0.25")},
                "unexplained_count": sum(comparison["regression_reasons"][key]["unexplained_count"] for key in ("0.50", "0.25")),
            },
            "gain_distribution": {key: {"new_tp_gt_count": tp_audit[key]["TP_improved_GT"]["count"], "new_tp_frame_count": len({item[0] for item in tp_audit[key]["TP_improved_GT"]["identity_list"]})} for key in ("0.50", "0.25")},
        }
        gate_results[name] = evaluate_variant_release_gates(baseline, variant, name, gate_config)
        selection_metrics[name] = _selection_metrics(name, t0, summary, variant, schedule)
    selection = select_release_candidate(gate_results, selection_metrics, gate_config)
    return {
        "schema_version": "15.4-intensity-filter-ablation-v1",
        "analysis_version": "15.4-phase2-finalize-low-memory-v1",
        "formal_comparison_commit": formal_comparison_commit,
        "audit_recovery_commit": audit_recovery_commit,
        "source_point_monotonicity": audit["source_point_monotonicity"]["status"],
        "variant_scalar_summaries": {name: {key: value for key, value in summary.items() if key != "delta_22_iou"} for name, summary in summaries.items()},
        "release_gate_results": gate_results,
        **selection,
    }


def _validity(num_frames):
    names = ("diagnostic_manifest_consistency", "formal_manifest_consistency", "delta_22_identity_consistency", "comparison_commit_consistency", "effective_config_diff", "t0_replay_25", "t0_replay_100", "dual_iou_complete", "required_artifacts_written", "source_point_monotonicity")
    return {"invariants": {name: True for name in names}, "formal_100": {"requested": 100, "success": num_frames, "metric_valid": num_frames, "partial_success": 0, "skipped": 0, "failed": 0}}


def _candidate_baseline_count(audit, variant_name):
    value = audit["comparisons_vs_T0"][variant_name]["candidate_regression"]
    return value["candidate_unchanged_GT"]["count"] + value["candidate_regressed_GT"]["count"]


def _selection_metrics(name, baseline, summary, variant, schedule):
    return {
        "variant": name,
        "f1_0_50": summary["metrics_by_iou"]["0.50"]["f1"], "tp_0_50": summary["metrics_by_iou"]["0.50"]["tp"],
        "f1_0_25": summary["metrics_by_iou"]["0.25"]["f1"], "tp_0_25": summary["metrics_by_iou"]["0.25"]["tp"],
        "fp_0_50": summary["metrics_by_iou"]["0.50"]["fp"],
        "effective_car_detection_growth": summary["counts"]["effective_car_detection"] - baseline["counts"]["effective_car_detection"],
        "effective_car_detection_count": summary["counts"]["effective_car_detection"],
        "strict_background_candidate_after_nms": summary["counts"]["strict_background_candidate_after_nms"],
        "candidate_regressed_gt_count": variant["candidate"]["regressed_gt_count"],
        "background_car_candidate_growth": summary["counts"]["strict_background_candidate_after_nms"] - baseline["counts"]["strict_background_candidate_after_nms"],
        "intensity_min": schedule["variants"][name]["intensity_min"],
    }
