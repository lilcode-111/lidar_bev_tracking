from copy import deepcopy

from bev_tracking.config import load_yaml_config
from bev_tracking.failure_evidence_batch import run_kitti_diagnostic_failure_evidence


INTENSITY_DIAGNOSTIC_SCHEMA_VERSION = "15.1"
REQUIRED_IOU_KEYS = ("0.50", "0.25")
ALLOWED_CONFIG_DIFFERENCE = "detector.intensity_min"


class IntensityDiagnosticError(ValueError):
    pass


def load_intensity_config(path):
    config = load_yaml_config(path)
    require_mapping(config.get("data"), "data")
    require_mapping(config.get("detector"), "detector")
    require_mapping(config.get("nms"), "nms")
    require_mapping(config.get("evaluation"), "evaluation")
    require_mapping(config.get("geometry"), "geometry")
    return config


def config_differences(config_a, config_b):
    flat_a = flatten_config(config_a)
    flat_b = flatten_config(config_b)
    differences = []
    for path in sorted(set(flat_a) | set(flat_b)):
        if flat_a.get(path) != flat_b.get(path):
            differences.append({"path": path, "i0": flat_a.get(path), "i1": flat_b.get(path)})
    return differences


def validate_intensity_config_pair(i0_config, i1_config):
    differences = config_differences(i0_config, i1_config)
    if [item["path"] for item in differences] != [ALLOWED_CONFIG_DIFFERENCE]:
        changed = ", ".join(item["path"] for item in differences) or "none"
        raise IntensityDiagnosticError(f"I0/I1 config difference must be intensity_min only: {changed}")

    i0_value = float(i0_config["detector"]["intensity_min"])
    i1_value = float(i1_config["detector"]["intensity_min"])
    if i0_value != 0.38 or i1_value != 0.0:
        raise IntensityDiagnosticError("I0/I1 intensity_min values must be 0.38 and 0.0")
    return differences


def validate_manifest_against_declaration(manifest, declaration):
    dataset = declaration["dataset"]
    if manifest["sha256"] != dataset["diagnostic_manifest_sha256"]:
        raise IntensityDiagnosticError("diagnostic manifest hash does not match A0 prime declaration")
    if manifest["num_frames"] != dataset["diagnostic_num_frames"]:
        raise IntensityDiagnosticError("diagnostic manifest frame count does not match A0 prime declaration")
    return {
        "passed": True,
        "sha256": manifest["sha256"],
        "num_frames": manifest["num_frames"],
        "frame_ids": list(manifest["frame_ids"]),
    }


def run_diagnostic_from_config(config, manifest, source_run_id=None, progress_callback=None):
    detector = config["detector"]
    evaluation = config["evaluation"]
    geometry = config["geometry"]
    return run_kitti_diagnostic_failure_evidence(
        data_root=config["data"]["root"],
        frame_ids=manifest["frame_ids"],
        eps=detector["eps"],
        min_points=detector["min_points"],
        oriented=detector["oriented"],
        z_min=detector["z_min"],
        intensity_min=detector["intensity_min"],
        nms_iou_threshold=config["nms"]["iou_threshold"],
        eval_iou_threshold=evaluation["iou_threshold"],
        auxiliary_iou_thresholds=tuple(evaluation["auxiliary_iou_thresholds"]),
        center_tolerance_m=geometry["center_tolerance_m"],
        yaw_tolerance_rad=geometry["yaw_tolerance_rad"],
        yaw_semantic_tolerance_rad=geometry["yaw_semantic_tolerance_rad"],
        corner_tolerance_m=geometry["corner_tolerance_m"],
        manifest_metadata=manifest,
        source_run_id=source_run_id,
        progress_callback=progress_callback,
    )


def validate_report_invariants(report, config, manifest, require_intensity_identity=False):
    expected_ids = list(manifest["frame_ids"])
    report_ids = [str(item["frame_id"]).zfill(6) for item in report.get("frames", [])]
    source_manifest = report.get("source", {}).get("diagnostic_manifest") or {}
    if report_ids != expected_ids or report.get("source", {}).get("requested_frame_ids") != expected_ids:
        raise IntensityDiagnosticError("diagnostic report frame order does not match frozen manifest")
    if source_manifest.get("sha256") != manifest["sha256"]:
        raise IntensityDiagnosticError("diagnostic report manifest hash mismatch")
    if report.get("summary", {}).get("geometry_failed_frames") != 0:
        raise IntensityDiagnosticError("diagnostic report contains geometry failures")

    aggregate_stage_counts = {"raw": 0, "roi": 0, "z_filter": 0, "intensity_filter": 0}
    aggregate_metrics = {
        iou_key: {"tp": 0, "fp": 0, "fn": 0, "neutralized_detections": 0, "effective_car_detection_count": 0}
        for iou_key in REQUIRED_IOU_KEYS
    }
    intensity_identity_frames = 0
    total_positive_gt = 0
    aggregate_candidate_generation = {
        "cluster_count": 0,
        "raw_detection_count": 0,
        "car_candidate_count_before_nms": 0,
        "non_car_candidate_count": 0,
        "nms_suppressed_count": 0,
        "car_nms_suppressed_count": 0,
        "final_car_detection_count": 0,
        "neutralized_detection_count": 0,
        "effective_car_detection_count": 0,
    }

    for frame in report["frames"]:
        frame_id = str(frame["frame_id"]).zfill(6)
        summary = frame["failure_evidence"]["summary"]
        parameters = frame["failure_evidence"]["parameters"]
        if float(parameters["intensity_min"]) != float(config["detector"]["intensity_min"]):
            raise IntensityDiagnosticError(f"intensity_min mismatch in frame report: {frame_id}")

        stage_counts = summary["stage_point_counts"]
        ordered_counts = [int(stage_counts[name]) for name in aggregate_stage_counts]
        if not all(left >= right for left, right in zip(ordered_counts, ordered_counts[1:])):
            raise IntensityDiagnosticError(f"filter stage point conservation failed: {frame_id}")
        for stage_name in aggregate_stage_counts:
            aggregate_stage_counts[stage_name] += int(stage_counts[stage_name])

        identical = bool(summary["z_to_intensity_identical"])
        intensity_identity_frames += int(identical)
        if require_intensity_identity and (
            not identical or int(stage_counts["z_filter"]) != int(stage_counts["intensity_filter"])
        ):
            raise IntensityDiagnosticError(f"I1 z/intensity point identity failed: {frame_id}")

        metrics_by_iou = summary["metrics_by_iou"]
        if any(iou_key not in metrics_by_iou for iou_key in REQUIRED_IOU_KEYS):
            raise IntensityDiagnosticError(f"required IoU metrics missing: {frame_id}")
        positive_gt = int(summary["num_positive_gt"])
        total_positive_gt += positive_gt
        effective_counts = []
        for iou_key in REQUIRED_IOU_KEYS:
            metrics = metrics_by_iou[iou_key]
            if int(metrics["tp"]) + int(metrics["fn"]) != positive_gt:
                raise IntensityDiagnosticError(f"TP+FN conservation failed: {frame_id} {iou_key}")
            effective = int(metrics["tp"]) + int(metrics["fp"]) + int(metrics["neutralized_detections"])
            if effective != int(metrics["effective_car_detection_count"]):
                raise IntensityDiagnosticError(f"effective detection count mismatch: {frame_id} {iou_key}")
            effective_counts.append(effective)
            for field in aggregate_metrics[iou_key]:
                aggregate_metrics[iou_key][field] += int(metrics[field])
        if len(set(effective_counts)) != 1:
            raise IntensityDiagnosticError(f"effective detection count differs across IoU thresholds: {frame_id}")

        candidate_generation = summary["candidate_generation"]
        validate_candidate_generation(candidate_generation, effective_counts[0], frame_id)
        for field in aggregate_candidate_generation:
            aggregate_candidate_generation[field] += int(candidate_generation[field])

    candidate_coverage = report.get("summary", {}).get("candidate_coverage")
    if not isinstance(candidate_coverage, dict):
        raise IntensityDiagnosticError("diagnostic report missing GT candidate coverage")
    coverage_counts = candidate_coverage.get("counts", {})
    if int(coverage_counts.get("num_positive_gt", -1)) != total_positive_gt:
        raise IntensityDiagnosticError("GT candidate coverage denominator mismatch")

    return {
        "passed": True,
        "num_frames": len(report_ids),
        "manifest_sha256": manifest["sha256"],
        "stage_point_counts": aggregate_stage_counts,
        "metrics_by_iou": aggregate_metrics,
        "candidate_generation_totals": aggregate_candidate_generation,
        "candidate_coverage": candidate_coverage,
        "z_to_intensity_identical_frames": intensity_identity_frames,
        "require_intensity_identity": bool(require_intensity_identity),
    }


def validate_candidate_generation(candidate_generation, effective_car_detection_count, frame_id):
    required_fields = {
        "cluster_count",
        "raw_detection_count",
        "car_candidate_count_before_nms",
        "non_car_candidate_count",
        "nms_suppressed_count",
        "car_nms_suppressed_count",
        "final_car_detection_count",
        "neutralized_detection_count",
        "effective_car_detection_count",
    }
    missing = sorted(required_fields - set(candidate_generation))
    if missing:
        raise IntensityDiagnosticError(f"candidate generation fields missing: {frame_id} {', '.join(missing)}")

    values = {field: int(candidate_generation[field]) for field in required_fields}
    if values["cluster_count"] != values["raw_detection_count"]:
        raise IntensityDiagnosticError(f"cluster/raw detection conservation failed: {frame_id}")
    if values["car_candidate_count_before_nms"] + values["non_car_candidate_count"] != values["raw_detection_count"]:
        raise IntensityDiagnosticError(f"Car/non-Car candidate conservation failed: {frame_id}")
    if values["car_candidate_count_before_nms"] - values["car_nms_suppressed_count"] != values["final_car_detection_count"]:
        raise IntensityDiagnosticError(f"Car NMS candidate conservation failed: {frame_id}")
    if values["effective_car_detection_count"] != effective_car_detection_count:
        raise IntensityDiagnosticError(f"candidate/evaluation effective detection mismatch: {frame_id}")
    if values["final_car_detection_count"] != effective_car_detection_count:
        raise IntensityDiagnosticError(f"final/effective Car detection mismatch: {frame_id}")


def validate_i0_reproduces_baseline(i0_report, baseline_batch_result, manifest):
    baseline_by_id = {frame.frame_id: frame for frame in baseline_batch_result.frame_results}
    mismatches = []

    for frame in i0_report["frames"]:
        frame_id = str(frame["frame_id"]).zfill(6)
        baseline = baseline_by_id.get(frame_id)
        if baseline is None:
            mismatches.append(f"{frame_id}: missing from A0 prime source run")
            continue
        summary = frame["failure_evidence"]["summary"]
        for iou_key in REQUIRED_IOU_KEYS:
            observed = deepcopy(summary["metrics_by_iou"][iou_key])
            observed.pop("effective_car_detection_count", None)
            expected_metrics = baseline.metrics_by_iou[iou_key]
            expected = expected_metrics.to_dict() if hasattr(expected_metrics, "to_dict") else dict(expected_metrics)
            if observed != expected:
                mismatches.append(f"{frame_id}: metrics mismatch at IoU {iou_key}")

        count_pairs = {
            "num_positive_gt": baseline.num_positive_gt,
            "num_raw_detections": baseline.num_raw_detections,
            "num_car_detections_before_nms": baseline.num_car_detections_before_nms,
            "num_detections_after_nms": baseline.num_detections_after_nms,
        }
        for field, expected in count_pairs.items():
            if int(summary[field]) != int(expected):
                mismatches.append(f"{frame_id}: {field} mismatch")

    if [item["frame_id"] for item in i0_report["frames"]] != list(manifest["frame_ids"]):
        mismatches.append("I0 frame order does not match diagnostic manifest")
    if mismatches:
        raise IntensityDiagnosticError("I0 does not reproduce A0 prime per frame: " + "; ".join(mismatches[:10]))
    return {
        "passed": True,
        "num_frames_compared": len(manifest["frame_ids"]),
        "compared_iou_keys": list(REQUIRED_IOU_KEYS),
        "mismatch_count": 0,
    }


def build_intensity_comparison(
    declaration,
    manifest_gate,
    config_diff,
    i0_report,
    i1_report,
    i0_invariants,
    i1_invariants,
    i0_reproduction,
):
    return {
        "schema_version": INTENSITY_DIAGNOSTIC_SCHEMA_VERSION,
        "experiment": "I0_vs_I1_intensity_filter_diagnostic",
        "status": "passed",
        "diagnostic_only": True,
        "baseline_id": declaration["baseline_id"],
        "source_run_id": declaration["source_run"]["run_id"],
        "manifest_gate": manifest_gate,
        "config_difference_gate": {"passed": True, "differences": config_diff},
        "i0_reproduction_gate": i0_reproduction,
        "i0_invariants": i0_invariants,
        "i1_invariants": i1_invariants,
        "comparison": {
            "metrics_by_iou": metric_comparison(i0_invariants, i1_invariants),
            "stage_point_counts": value_comparison(
                i0_invariants["stage_point_counts"],
                i1_invariants["stage_point_counts"],
            ),
            "primary_reason_counts": value_comparison(
                i0_report["summary"]["primary_reason_counts"],
                i1_report["summary"]["primary_reason_counts"],
            ),
            "candidate_generation_totals": value_comparison(
                i0_invariants["candidate_generation_totals"],
                i1_invariants["candidate_generation_totals"],
            ),
            "gt_candidate_coverage_counts": value_comparison(
                i0_invariants["candidate_coverage"]["counts"],
                i1_invariants["candidate_coverage"]["counts"],
            ),
            "candidate_outcome_counts": value_comparison(
                i0_invariants["candidate_coverage"]["candidate_outcome_counts"],
                i1_invariants["candidate_coverage"]["candidate_outcome_counts"],
            ),
            "zero_car_candidate_gt_count": {
                "i0": int(i0_invariants["candidate_coverage"]["zero_car_candidate_gt_count"]),
                "i1": int(i1_invariants["candidate_coverage"]["zero_car_candidate_gt_count"]),
                "delta": int(i1_invariants["candidate_coverage"]["zero_car_candidate_gt_count"])
                - int(i0_invariants["candidate_coverage"]["zero_car_candidate_gt_count"]),
            },
        },
    }


def metric_comparison(i0_invariants, i1_invariants):
    return {
        iou_key: value_comparison(
            i0_invariants["metrics_by_iou"][iou_key],
            i1_invariants["metrics_by_iou"][iou_key],
        )
        for iou_key in REQUIRED_IOU_KEYS
    }


def value_comparison(i0_values, i1_values):
    keys = sorted(set(i0_values) | set(i1_values))
    return {
        key: {
            "i0": int(i0_values.get(key, 0)),
            "i1": int(i1_values.get(key, 0)),
            "delta": int(i1_values.get(key, 0)) - int(i0_values.get(key, 0)),
        }
        for key in keys
    }


def flatten_config(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    output = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        output.update(flatten_config(item, path))
    return output


def require_mapping(value, description):
    if not isinstance(value, dict):
        raise IntensityDiagnosticError(f"{description} config must be an object")
    return value
