"""Protocol helpers for the v15.3 Car-candidate conversion diagnosis.

This module only defines attribution semantics. It does not run clustering,
classification, NMS, or evaluation.
"""

from collections import Counter
import math
from numbers import Integral

import numpy as np

from bev_tracking.eval_policy import classify_gt_box, assign_det_indices, detection_sort_key, is_positive_detection
from bev_tracking.geometry import bev_iou
from bev_tracking.geometry_sanity import points_in_oriented_3d_box
from bev_tracking.result_types import CandidateConversionEvidence, CandidateConversionState


PRIMARY_IOU = 0.50
AUXILIARY_IOU = 0.25
CANDIDATE_CONVERSION_SCHEMA_VERSION = "15.3.1"
CANDIDATE_CONVERSION_SOURCE_OF_TRUTH = "candidate_conversion.evidence"
DEPRECATED_FRAME_REPORT_FIELDS = (
    "gt_candidate_records",
    "failure_evidence",
)


def derive_terminal_state(
    *,
    filtered_point_count,
    min_points,
    cluster_ids,
    raw_detection_ids,
    car_detection_ids_before_nms,
    car_detection_ids_after_nms,
    best_iou_after_nms,
    matched_at_primary_iou,
):
    """Assign exactly one terminal state using the frozen pipeline order."""
    if filtered_point_count == 0:
        return CandidateConversionState.NO_FILTERED_POINTS
    if filtered_point_count < min_points and not cluster_ids:
        return CandidateConversionState.INSUFFICIENT_FILTERED_POINTS_FOR_ASSOCIATION
    if not cluster_ids:
        return CandidateConversionState.NO_ASSOCIATED_CLUSTER
    if not raw_detection_ids:
        return CandidateConversionState.NO_ASSOCIATED_CLUSTER
    if not car_detection_ids_before_nms:
        return CandidateConversionState.REJECTED_BY_CAR_CLASSIFIER
    if not car_detection_ids_after_nms:
        return CandidateConversionState.REMOVED_BY_NMS
    if matched_at_primary_iou:
        return CandidateConversionState.MATCHED_AT_0_50
    if best_iou_after_nms < AUXILIARY_IOU:
        return CandidateConversionState.BOX_IOU_BELOW_0_25
    if best_iou_after_nms < PRIMARY_IOU:
        return CandidateConversionState.BOX_IOU_0_25_TO_0_50
    return CandidateConversionState.IOU_GE_0_50_BUT_UNMATCHED


def validate_terminal_assignments(records, expected_gt_ids):
    """Validate uniqueness and conservation of GT terminal assignments."""
    expected = {str(gt_id) for gt_id in expected_gt_ids}
    actual_ids = [str(record.gt_id) for record in records]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("duplicate GT terminal assignment")
    if set(actual_ids) != expected:
        raise ValueError("GT terminal assignments do not conserve positive GT set")
    counts = Counter(record.terminal_state.value for record in records)
    return {state.value: int(counts.get(state.value, 0)) for state in CandidateConversionState}


def validate_min_points(min_points):
    if isinstance(min_points, bool) or not isinstance(min_points, Integral) or min_points <= 0:
        raise ValueError("min_points must be a positive integer")
    return int(min_points)


def effective_min_points_for_gt(gt_box, min_points, clustering_policy=None):
    """Resolve the audit threshold from the GT's own distance bin."""
    fallback = validate_min_points(min_points)
    if clustering_policy is None:
        return fallback
    range_xy = float(np.hypot(float(gt_box["x"]), float(gt_box["y"])))
    return validate_min_points(clustering_policy.params_for_range(range_xy)["min_points"])


def gt_cluster_association_indices(filtered_points, gt_box, clusters):
    """Return all associated cluster indices using the frozen v15.2 gate."""
    gt_point_count = int(points_in_oriented_3d_box(filtered_points, gt_box).sum())
    if gt_point_count < 3:
        return gt_point_count, []
    minimum = max(3, int(math.ceil(0.10 * gt_point_count)))
    associated = []
    for index, cluster in enumerate(clusters):
        if int(points_in_oriented_3d_box(cluster, gt_box).sum()) >= minimum:
            associated.append(index)
    return gt_point_count, associated


def _axis_features(cluster):
    x_span = float(max(cluster[:, 0].max() - cluster[:, 0].min(), 0.1))
    y_span = float(max(cluster[:, 1].max() - cluster[:, 1].min(), 0.1))
    z_span = float(max(cluster[:, 2].max() - cluster[:, 2].min(), 0.0))
    return {
        "num_points": int(len(cluster)),
        "axis_length": x_span,
        "axis_width": y_span,
        "height_span": z_span,
        "range_xy_m": float(np.hypot(float(cluster[:, 0].mean()), float(cluster[:, 1].mean()))),
        "point_density_xy": float(len(cluster) / max(x_span * y_span, 1e-6)),
    }


def _branch_record(cluster_index, cluster, raw_detection, kept_ids, gt_box):
    detection_id = str(raw_detection.get("id", f"cluster_{cluster_index + 1}"))
    is_kept = detection_id in kept_ids
    return {
        "cluster_id": f"cluster_{cluster_index + 1}",
        "raw_detection_id": detection_id,
        "class_name": str(raw_detection.get("class_name", "")),
        "is_car_candidate": bool(is_positive_detection(raw_detection)),
        "is_after_nms": bool(is_kept),
        "cluster_features": _axis_features(cluster),
        "pca_length": float(raw_detection.get("length", 0.0)),
        "pca_width": float(raw_detection.get("width", 0.0)),
        "candidate_iou": float(bev_iou(gt_box, raw_detection)),
    }


def build_candidate_conversion_evidence(
    *,
    frame_id,
    gt_box,
    stages,
    clusters,
    raw_detections,
    detections_after_nms,
    evaluation,
    variant,
    min_points,
    clustering_policy=None,
    source_run_id=None,
):
    """Build one GT's complete C0/C1 candidate-conversion lineage."""
    min_points = effective_min_points_for_gt(gt_box, min_points, clustering_policy)
    filtered_points = stages["intensity_filter"]
    filtered_count, associated_indices = gt_cluster_association_indices(filtered_points, gt_box, clusters)
    associated_raw = [raw_detections[index] for index in associated_indices]
    kept_ids = {str(det.get("id")) for det in detections_after_nms}
    car_before = [det for det in associated_raw if is_positive_detection(det)]
    car_after = [det for det in car_before if str(det.get("id")) in kept_ids]
    branches = [
        _branch_record(index, clusters[index], raw_detections[index], kept_ids, gt_box)
        for index in associated_indices
    ]
    associated_after_ids = {str(det.get("id")) for det in car_after}
    associated_after = [det for det in car_after if str(det.get("id")) in associated_after_ids]
    nms_trace = trace_nms_suppression(raw_detections, detections_after_nms)
    best_before = max((float(bev_iou(gt_box, det)) for det in car_before), default=0.0)
    best_after = max((float(bev_iou(gt_box, det)) for det in associated_after), default=0.0)
    primary_iou_key = f'{float(evaluation.get("iou_threshold", PRIMARY_IOU)):.2f}'
    matched_ids_by_iou = {
        primary_iou_key: {
            str(item.get("det_id"))
            for item in evaluation.get("matches", [])
            if str(item.get("gt_id")) == str(gt_box.get("id"))
        }
    }
    for iou_key, auxiliary in evaluation.get("auxiliary", {}).items():
        matched_ids_by_iou[iou_key] = {
            str(item.get("det_id"))
            for item in auxiliary.get("matches", [])
            if str(item.get("gt_id")) == str(gt_box.get("id"))
        }
    matched_ids = matched_ids_by_iou.get(primary_iou_key, set())
    matched = bool(matched_ids & associated_after_ids)
    downstream = build_downstream_attribution(
        gt_box=gt_box,
        associated_after=associated_after,
        associated_before=car_before,
        nms_trace=nms_trace,
        evaluation=evaluation,
        matched_ids=matched_ids,
    )
    terminal_state = derive_terminal_state(
        filtered_point_count=filtered_count,
        min_points=min_points,
        cluster_ids=[f"cluster_{index + 1}" for index in associated_indices],
        raw_detection_ids=[str(det.get("id")) for det in associated_raw],
        car_detection_ids_before_nms=[str(det.get("id")) for det in car_before],
        car_detection_ids_after_nms=[str(det.get("id")) for det in car_after],
        best_iou_after_nms=best_after,
        matched_at_primary_iou=matched,
    )
    frame_id = str(frame_id).zfill(6)
    range_xy = float(np.hypot(float(gt_box["x"]), float(gt_box["y"])))
    distance_bin = "near_0_15" if range_xy < 15.0 else "mid_15_30" if range_xy < 30.0 else "far_30_inf"
    return CandidateConversionEvidence(
        frame_id=frame_id,
        gt_id=gt_box["id"],
        terminal_state=terminal_state,
        variant=variant,
        distance_bin=distance_bin,
        stage_point_counts={
            name: int(points_in_oriented_3d_box(stage_points, gt_box).sum())
            for name, stage_points in stages.items()
        },
        cluster_ids=[f"cluster_{index + 1}" for index in associated_indices],
        raw_detection_ids=[str(det.get("id")) for det in associated_raw],
        car_detection_ids_before_nms=[str(det.get("id")) for det in car_before],
        car_detection_ids_after_nms=[str(det.get("id")) for det in car_after],
        candidate_branches=branches,
        downstream_attribution=downstream,
        best_iou_before_nms=best_before,
        best_iou_after_nms=best_after,
        matched_by_iou={key: bool(ids & associated_after_ids) for key, ids in matched_ids_by_iou.items()},
        matched_at_primary_iou=matched,
        source_run_id=source_run_id,
    )


def build_candidate_conversion_report(
    *,
    frame_id,
    gt_boxes,
    stages,
    clusters,
    raw_detections,
    detections_after_nms,
    evaluation,
    variant,
    min_points,
    clustering_policy=None,
    source_run_id=None,
):
    """Build evidence for every positive GT and summarize terminal states."""
    positive_gt = [box for box in gt_boxes if classify_gt_box(box) == "positive"]
    records = [
        build_candidate_conversion_evidence(
            frame_id=frame_id,
            gt_box=gt_box,
            stages=stages,
            clusters=clusters,
            raw_detections=raw_detections,
            detections_after_nms=detections_after_nms,
            evaluation=evaluation,
            variant=variant,
            min_points=min_points,
            clustering_policy=clustering_policy,
            source_run_id=source_run_id,
        )
        for gt_box in positive_gt
    ]
    counts = validate_terminal_assignments(records, [box["id"] for box in positive_gt])
    return {
        "schema_version": CANDIDATE_CONVERSION_SCHEMA_VERSION,
        "source_of_truth": CANDIDATE_CONVERSION_SOURCE_OF_TRUTH,
        "deprecated_fields": list(DEPRECATED_FRAME_REPORT_FIELDS),
        "frame_id": str(frame_id).zfill(6),
        "variant": str(variant),
        "num_positive_gt": len(positive_gt),
        "terminal_state_counts": counts,
        "evidence": [record.to_dict() for record in records],
    }


def validate_candidate_conversion_payload(payload):
    """Validate the frozen 15.3.1 canonical evidence contract."""
    if not isinstance(payload, dict):
        raise ValueError("candidate conversion payload must be a dictionary")
    if payload.get("schema_version") != CANDIDATE_CONVERSION_SCHEMA_VERSION:
        raise ValueError("unsupported candidate conversion schema version")
    if payload.get("source_of_truth") != CANDIDATE_CONVERSION_SOURCE_OF_TRUTH:
        raise ValueError("candidate conversion source_of_truth is invalid")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("candidate conversion evidence must be a list")
    keys = []
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("candidate conversion evidence item must be a dictionary")
        key = (str(item.get("frame_id")), str(item.get("gt_id")))
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("candidate conversion evidence contains duplicate frame/GT")
    expected = payload.get("num_positive_gt")
    if expected != len(evidence):
        raise ValueError("candidate conversion evidence count does not match num_positive_gt")
    return True


def _yaw_error(yaw_a, yaw_b):
    return float(abs((float(yaw_a) - float(yaw_b) + np.pi / 2.0) % np.pi - np.pi / 2.0))


def trace_nms_suppression(boxes, kept_boxes, iou_threshold=0.3):
    """Reconstruct NMS suppressor relationships without changing NMS behavior."""
    pending = sorted(assign_det_indices(boxes), key=detection_sort_key)
    expected_kept_ids = {str(box.get("id")) for box in kept_boxes}
    kept = []
    suppressed = []
    while pending:
        current = pending.pop(0)
        kept.append(current)
        remaining = []
        for box in pending:
            same_class = box.get("class_name") == current.get("class_name")
            iou = float(bev_iou(current, box)) if same_class else 0.0
            if same_class and iou > iou_threshold:
                suppressed.append({
                    "suppressed_id": str(box.get("id")),
                    "suppressor_id": str(current.get("id")),
                    "iou": iou,
                })
            else:
                remaining.append(box)
        pending = remaining

    actual_kept_ids = {str(box.get("id")) for box in kept}
    if actual_kept_ids != expected_kept_ids:
        raise ValueError("NMS trace does not match supplied kept detections")
    return {
        "iou_threshold": float(iou_threshold),
        "kept_ids": sorted(actual_kept_ids),
        "suppressed": suppressed,
    }


def build_downstream_attribution(*, gt_box, associated_after, associated_before, nms_trace, evaluation, matched_ids):
    """Summarize NMS, geometry, and evaluation competition for one GT."""
    associated_ids = {str(det.get("id")) for det in associated_before}
    suppression_records = [
        item for item in nms_trace.get("suppressed", [])
        if item["suppressed_id"] in associated_ids
    ]
    geometry = []
    for det in associated_after:
        dx = float(det["x"] - gt_box["x"])
        dy = float(det["y"] - gt_box["y"])
        geometry.append({
            "detection_id": str(det.get("id")),
            "dx_m": dx,
            "dy_m": dy,
            "center_error_m": float(np.hypot(dx, dy)),
            "length_error_m": float(det["length"] - gt_box["length"]),
            "width_error_m": float(det["width"] - gt_box["width"]),
            "yaw_error_rad": _yaw_error(det["yaw"], gt_box["yaw"]),
            "iou": float(bev_iou(gt_box, det)),
        })
    associated_ids_after = {str(det.get("id")) for det in associated_after}
    matched_associated = sorted(associated_ids_after & set(matched_ids))
    competing_matches = [
        {
            "det_id": str(item.get("det_id")),
            "gt_id": str(item.get("gt_id")),
            "iou": float(item.get("iou", 0.0)),
        }
        for item in evaluation.get("matches", [])
        if str(item.get("det_id")) in associated_ids_after
        and str(item.get("gt_id")) != str(gt_box.get("id"))
    ]
    best_geometry = max(geometry, key=lambda item: (item["iou"], item["detection_id"]), default=None)
    return {
        "nms": {
            "associated_suppressed_count": len(suppression_records),
            "suppression_records": suppression_records,
        },
        "geometry": {
            "candidates": geometry,
            "best_iou_candidate": best_geometry,
        },
        "evaluation": {
            "matched_associated_detection_ids": matched_associated,
            "matched_by_associated_candidate": bool(matched_associated),
            "competition_matches": competing_matches,
            "iou_ge_0_50_but_unmatched": bool(best_geometry and best_geometry["iou"] >= PRIMARY_IOU and not matched_associated),
        },
    }


WATERFALL_FIELDS = (
    "num_positive_gt",
    "tp_iou_0_50",
    "fn_iou_0_50",
    "tp_iou_0_25",
    "fn_iou_0_25",
    "gt_with_filtered_points",
    "gt_with_associated_cluster",
    "gt_with_car_before_nms",
    "gt_with_car_after_nms",
    "gt_with_associated_iou_ge_0_25",
    "gt_with_associated_iou_ge_0_50",
    "gt_matched_at_iou_0_50",
    "zero_detection_with_gt",
)
DISTANCE_BINS = ("near_0_15", "mid_15_30", "far_30_inf", "total")


def _empty_waterfall():
    return {field: 0 for field in WATERFALL_FIELDS}


def _ratio_counts(counts):
    denominator = counts["num_positive_gt"]
    return {
        field: None if denominator == 0 else float(value / denominator)
        for field, value in counts.items()
        if field != "num_positive_gt"
    }


def _add_waterfall_record(target, record):
    target["num_positive_gt"] += 1
    matched = record.get("matched_by_iou") or {}
    target["tp_iou_0_50"] += int(bool(matched.get("0.50", record.get("matched_at_primary_iou", False))))
    target["tp_iou_0_25"] += int(bool(matched.get("0.25", False)))
    stage_counts = record.get("stage_point_counts") or {}
    if int(stage_counts.get("intensity_filter", 0)) > 0:
        target["gt_with_filtered_points"] += 1
    if record.get("cluster_ids"):
        target["gt_with_associated_cluster"] += 1
    if record.get("car_detection_ids_before_nms"):
        target["gt_with_car_before_nms"] += 1
    if record.get("car_detection_ids_after_nms"):
        target["gt_with_car_after_nms"] += 1
    target["zero_detection_with_gt"] += int(not record.get("car_detection_ids_after_nms"))
    if float(record.get("best_iou_after_nms", 0.0)) >= AUXILIARY_IOU:
        target["gt_with_associated_iou_ge_0_25"] += 1
    if float(record.get("best_iou_after_nms", 0.0)) >= PRIMARY_IOU:
        target["gt_with_associated_iou_ge_0_50"] += 1
    if record.get("matched_at_primary_iou"):
        target["gt_matched_at_iou_0_50"] += 1
    target["fn_iou_0_50"] = target["num_positive_gt"] - target["tp_iou_0_50"]
    target["fn_iou_0_25"] = target["num_positive_gt"] - target["tp_iou_0_25"]


def _feature_summary(branches):
    numeric = {
        "num_points": [],
        "axis_max_dim": [],
        "axis_length": [],
        "axis_width": [],
        "height_span": [],
        "range_xy_m": [],
        "point_density_xy": [],
    }
    gate_counts = {
        "associated_branch_count": 0,
        "axis_size_pass_count": 0,
        "point_count_pass_count": 0,
        "both_classifier_gates_pass_count": 0,
        "neither_classifier_gate_pass_count": 0,
        "car_candidate_count": 0,
    }
    for branch in branches:
        features = branch.get("cluster_features") or {}
        axis_length = float(features.get("axis_length", 0.0))
        axis_width = float(features.get("axis_width", 0.0))
        num_points = int(features.get("num_points", 0))
        axis_pass = max(axis_length, axis_width) >= 2.0
        points_pass = num_points >= 500
        gate_counts["associated_branch_count"] += 1
        gate_counts["axis_size_pass_count"] += int(axis_pass)
        gate_counts["point_count_pass_count"] += int(points_pass)
        gate_counts["both_classifier_gates_pass_count"] += int(axis_pass or points_pass)
        gate_counts["neither_classifier_gate_pass_count"] += int(not (axis_pass or points_pass))
        gate_counts["car_candidate_count"] += int(branch.get("is_car_candidate", False))
        numeric["num_points"].append(num_points)
        numeric["axis_max_dim"].append(max(axis_length, axis_width))
        for field in ("axis_length", "axis_width", "height_span", "range_xy_m", "point_density_xy"):
            numeric[field].append(float(features.get(field, 0.0)))

    distributions = {}
    for field, values in numeric.items():
        distributions[field] = {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": float(sum(values) / len(values)) if values else None,
        }
    return {"gate_counts": gate_counts, "distributions": distributions}


def summarize_candidate_conversion_records(records):
    """Summarize read-only waterfall and classifier features by distance bin."""
    grouped = {distance_bin: [] for distance_bin in DISTANCE_BINS}
    for record in records:
        distance_bin = record.get("distance_bin")
        if distance_bin not in DISTANCE_BINS[:-1]:
            raise ValueError(f"invalid distance_bin: {distance_bin}")
        grouped[distance_bin].append(record)
        grouped["total"].append(record)

    waterfall = {}
    classification = {}
    terminal_states = {}
    for distance_bin in DISTANCE_BINS:
        counts = _empty_waterfall()
        for record in grouped[distance_bin]:
            _add_waterfall_record(counts, record)
        waterfall[distance_bin] = {"counts": counts, "ratios": _ratio_counts(counts)}
        branches = [branch for record in grouped[distance_bin] for branch in record.get("candidate_branches", [])]
        classification[distance_bin] = _feature_summary(branches)
        state_counts = Counter(record.get("terminal_state", "unknown") for record in grouped[distance_bin])
        terminal_states[distance_bin] = dict(sorted(state_counts.items()))

    return {
        "distance_bins": list(DISTANCE_BINS),
        "waterfall": waterfall,
        "classification_audit": classification,
        "terminal_state_counts": terminal_states,
    }


def build_candidate_conversion_delta(reports_by_variant, base_variant="C0", candidate_variant="C1"):
    """Compare canonical GT evidence and explain newly associated candidate GTs."""
    def index_evidence(reports, variant):
        indexed = {}
        for report in reports:
            conversion = report.get("candidate_conversion") or {}
            records = conversion.get("evidence")
            if not isinstance(records, list):
                raise ValueError(f"missing canonical evidence for {variant}: {report.get('frame_id')}")
            for record in records:
                key = (str(record.get("frame_id")).zfill(6), str(record.get("gt_id")))
                if key in indexed:
                    raise ValueError(f"duplicate canonical GT evidence: {variant} {key}")
                indexed[key] = record
        return indexed

    if base_variant not in reports_by_variant or candidate_variant not in reports_by_variant:
        raise ValueError("delta cohort requires base and candidate variants")
    base = index_evidence(reports_by_variant[base_variant], base_variant)
    candidate = index_evidence(reports_by_variant[candidate_variant], candidate_variant)
    if set(base) != set(candidate):
        raise ValueError("base and candidate GT evidence sets differ")

    added_keys = sorted(
        key for key in candidate
        if bool(candidate[key].get("cluster_ids")) and not bool(base[key].get("cluster_ids"))
    )
    terminal_counts = Counter(candidate[key].get("terminal_state", "unknown") for key in added_keys)
    stage_counts = {
        "car_classification_pass": 0,
        "rejected_by_car_classifier": 0,
        "removed_by_nms": 0,
        "iou_below_0_25": 0,
        "iou_0_25_to_0_50": 0,
        "iou_ge_0_50": 0,
        "matched_at_0_50": 0,
    }
    cohort = []
    for frame_id, gt_id in added_keys:
        before = base[(frame_id, gt_id)]
        after = candidate[(frame_id, gt_id)]
        terminal = after.get("terminal_state", "unknown")
        has_car_before = bool(after.get("car_detection_ids_before_nms"))
        stage_counts["car_classification_pass"] += int(has_car_before)
        stage_counts["rejected_by_car_classifier"] += int(terminal == "rejected_by_car_classifier")
        stage_counts["removed_by_nms"] += int(terminal == "removed_by_nms")
        stage_counts["iou_below_0_25"] += int(terminal == "box_iou_below_0_25")
        stage_counts["iou_0_25_to_0_50"] += int(terminal == "box_iou_0_25_to_0_50")
        stage_counts["iou_ge_0_50"] += int(terminal in {"iou_ge_0_50_but_unmatched", "matched_at_0_50"})
        stage_counts["matched_at_0_50"] += int(terminal == "matched_at_0_50")
        cohort.append({
            "frame_id": frame_id,
            "gt_id": gt_id,
            "base_variant": {
                "terminal_state": before.get("terminal_state"),
                "cluster_ids": list(before.get("cluster_ids", [])),
            },
            "candidate_variant": {
                "terminal_state": terminal,
                "cluster_ids": list(after.get("cluster_ids", [])),
                "car_detection_ids_before_nms": list(after.get("car_detection_ids_before_nms", [])),
                "car_detection_ids_after_nms": list(after.get("car_detection_ids_after_nms", [])),
                "best_iou_after_nms": float(after.get("best_iou_after_nms", 0.0)),
                "matched_by_iou": dict(after.get("matched_by_iou", {})),
            },
        })
    return {
        "base_variant": base_variant,
        "candidate_variant": candidate_variant,
        "base_associated_gt_count": sum(bool(item.get("cluster_ids")) for item in base.values()),
        "candidate_associated_gt_count": sum(bool(item.get("cluster_ids")) for item in candidate.values()),
        "new_associated_gt_count": len(cohort),
        "terminal_state_counts": dict(sorted(terminal_counts.items())),
        "conversion_stage_counts": stage_counts,
        "records": cohort,
    }
